"""Main training script for DNA-Text fusion models."""

import os
import argparse
import yaml
import time
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
import wandb

# Import from our modular structure
from ..configs.default_config import Config, default_config
from ..data.dataset import get_data_loaders
from ..models.dna_text_fusion import DNATextFusion, load_models
from ..losses.contrastive import DNATextContrastiveLoss
from ..training.evaluation import evaluate_model
from ..utils.optimization import create_optimizer_and_scheduler

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path=None):
    """Load configuration from YAML file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            # Convert dict to our Config dataclass
            # This is a simplified approach - you might want to use a more robust conversion
            config = default_config
            for section, values in config_dict.items():
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
            return config
    return default_config

def train_model(config=None):
    """Main training function for DNA-Text fusion models."""
    # Load configuration
    config = config or default_config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config={
            "training": vars(config.training),
            "model": vars(config.model),
            "data": vars(config.data)
        }
    )
    
    # Load tokenizers and models
    dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
        dna_model_path=config.model.dna_model_path,
        text_model_path=config.model.text_model_path,
        device=device
    )
    
    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        dna_tokenizer=dna_tokenizer,
        text_tokenizer=text_tokenizer,
        config=config
    )
    
    # Initialize model
    model = DNATextFusion(
        dna_model=dna_model,
        text_model=text_model,
        cross_attn_interval=config.model.cross_attn_interval
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=config.training.learning_rate,
        steps_per_epoch=len(train_loader),
        num_epochs=config.training.num_epochs,
        grad_accum_steps=config.training.grad_accum_steps
    )
    
    # Initialize contrastive loss
    contrastive_loss_fn = DNATextContrastiveLoss(
        initial_temp=0.5,
        min_temp=0.07,
        total_epochs=config.training.num_epochs,
        weight=config.training.semantic_loss_weight
    )
    
    # Initialize cross-entropy loss
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    
    
    # Training tracking variables
    best_bleu = 0
    epochs_no_improve = 0
    
    # Start training loop
    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0.0
        grad_norms = []
        
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        epoch_start_time = time.time()
        
        # Training steps
        for step, batch in enumerate(progress_bar):
            # Move inputs to device
            dna_inputs = {
                "input_ids": batch["dna_input_ids"].to(device),
                "attention_mask": batch["dna_attention_mask"].to(device),
            }
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
                "attention_mask": batch["text_attention_mask"].to(device),
            }

            # Text → DNA prediction (exact match)
            outputs = model(dna_inputs=dna_inputs, text_inputs=text_inputs)
            
            # Cross-entropy loss for prediction
            text2dna_loss = ce_loss_fn(
                outputs["logits"].view(-1, outputs["logits"].size(-1)),
                batch["dna_input_ids"].to(device).view(-1)
            )
            
            # Contrastive loss between embeddings
            contrastive_loss, contrastive_metrics = contrastive_loss_fn(
                outputs["joint_embedding"],
                outputs["fused_representation"],
                epoch=epoch
            )
            
            # Combine losses
            loss = text2dna_loss + contrastive_loss
            loss = loss / config.training.grad_accum_steps
            
            # Backward pass with gradient accumulation
            loss.backward()
            
            # Step optimizer every grad_accum_steps
            if (step + 1) % config.training.grad_accum_steps == 0:
                # Gradient clipping without unscaling
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norms.append(grad_norm.item())
                
                # Regular optimizer step without scaler
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            
            # Track loss
            total_loss += loss.item()
            
            # Calculate remaining training time (ETA)
            steps_done = step + 1 + epoch * len(train_loader)
            total_training_steps = config.training.num_epochs * len(train_loader)
            if step > 0:
                steps_per_sec = step / ((time.time() - epoch_start_time) or 1)
                remaining_steps = total_training_steps - steps_done
                eta_seconds = remaining_steps / steps_per_sec
                eta_minutes = eta_seconds / 60
            else:
                eta_minutes = None
                
            # Log step metrics
            step_metrics = {
                "step_loss": loss.item(),
                "text2dna_loss": text2dna_loss.item(),
                **contrastive_metrics,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "step": steps_done,
                "eta_min": eta_minutes
            }
            
            if len(grad_norms) > 0:
                step_metrics["grad_norm"] = grad_norms[-1]
                
            wandb.log(step_metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "t2d": text2dna_loss.item(),
                "contr": contrastive_loss.item(),
                "temp": contrastive_metrics["contrastive_temperature"]
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
        epoch_time = time.time() - epoch_start_time
        
        # Evaluate model on test set
        eval_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            text_tokenizer=text_tokenizer,
            dna_tokenizer=dna_tokenizer,
            device=device,
            current_epoch=epoch+1
        )
        
        bleu_score = eval_metrics["bleu_score"]
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_grad_norm": avg_grad_norm,
            "epoch_time_sec": epoch_time,
            **eval_metrics
        })
        
        # Print epoch summary
        print(f"[Epoch {epoch+1}/{config.training.num_epochs}] "
              f"Loss: {avg_loss:.4f} | GradNorm: {avg_grad_norm:.4f} | "
              f"Val BLEU: {bleu_score:.4f} | Time: {epoch_time:.1f}s")
        
        # Early stopping logic
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            epochs_no_improve = 0
            print("✨ New best model found!")
            torch.save(model.state_dict(), "best_model.pt")
            wandb.log({"best_model_saved": True})
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.training.early_stop_patience:
                print("⏹️ Early stopping triggered.")
                wandb.log({"early_stopped": True})
                break
    
    # Final evaluation
    print("\nTraining complete. Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pt"))
    final_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        text_tokenizer=text_tokenizer,
        dna_tokenizer=dna_tokenizer,
        device=device,
        current_epoch="final"
    )
    
    # Log final metrics
    wandb.log({
        "final_bleu": final_metrics["bleu_score"],
        "final_rouge1": final_metrics["rouge1"],
        "final_rouge2": final_metrics["rouge2"],
        "final_rougeL": final_metrics["rougeL"],
        "training_completed": True
    })
    
    print(f"Final BLEU Score: {final_metrics['bleu_score']:.4f}")
    wandb.finish()
    
    return model, final_metrics

def main():
    parser = argparse.ArgumentParser(description="Train DNA-Text fusion model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(config)

if __name__ == "__main__":
    main()