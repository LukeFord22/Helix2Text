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
from ..utils.logging import log_batch_examples

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
        max_step_times = 100
        step_times = []
        
        
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        epoch_start_time = time.time()
        
        # Training steps
        for step, batch in enumerate(progress_bar):
            
            #log time
            step_start_time = time.time()
            steps_done = step + 1 + epoch * len(train_loader)
            total_training_steps = config.training.num_epochs * len(train_loader)

            # Move inputs to device
            dna_inputs = {
                "input_ids": batch["dna_input_ids"].to(device),
                "attention_mask": batch["dna_attention_mask"].to(device),
            }
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
                "attention_mask": batch["text_attention_mask"].to(device),
            }

            outputs = model(dna_inputs=dna_inputs, text_inputs=text_inputs)
            
            # Cross-entropy loss for Text (predict text from DNA)
            text_ce_loss = ce_loss_fn(
                outputs["text_logits"].view(-1, outputs["text_logits"].size(-1)),
                batch["text_input_ids"].to(device).view(-1)
            )

            # Cross-entropy loss for DNA (predict DNA from text)
            dna_ce_loss = ce_loss_fn(
                outputs["dna_logits"].view(-1, outputs["dna_logits"].size(-1)),
                batch["dna_input_ids"].to(device).view(-1)
            )
            
            # Contrastive loss between embeddings
            contrastive_loss, contrastive_metrics = contrastive_loss_fn(
                outputs["joint_embedding"],
                outputs["fused_representation"],
                epoch=epoch
            )
            
            # Weighted loss combination
            dna_weight = config.training.dna_loss_weight 
            text_weight = config.training.text_loss_weight  
            contrastive_weight = config.training.contrastive_loss_weight 

            # Combine losses with weights
            loss = (
                dna_weight * dna_ce_loss +
                text_weight * text_ce_loss +
                contrastive_weight * contrastive_loss
            ) / config.training.grad_accum_steps
            
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
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            # Update step times list
            step_times.append(step_duration)
            if len(step_times) > max_step_times:
                step_times.pop(0)  # Remove oldest timing

            # Calculate ETA based on recent average
            avg_step_time = sum(step_times) / len(step_times)
            remaining_steps = total_training_steps - steps_done
            eta_seconds = remaining_steps * avg_step_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
                
            # Log step metrics
            step_metrics = {
                "step_loss": loss.item(),
                "DNA": dna_ce_loss.item(),
                "Text": text_ce_loss.item(),
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
                "Loss": loss.item(),
                "DNA": dna_ce_loss.item(),
                "Text": text_ce_loss.item(),
                "Cont": contrastive_loss.item(),
                "temp": contrastive_metrics["contrastive_temperature"],
                "eta_min": eta_minutes
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
        
        # Get a batch of examples from the test loader to visualize
        example_batch = next(iter(test_loader))
        
        # Move batch to device
        example_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in example_batch.items()}
        
        # Generate text predictions from DNA sequences
        with torch.no_grad():
            model.eval()
            outputs = model.generate(
                input_ids=example_batch["dna_input_ids"],
                attention_mask=example_batch["dna_attention_mask"],
                max_new_tokens=100,  # Adjust based on your expected output length
                num_beams=4
            )
            
            # Decode the generated text
            generated_texts = [text_tokenizer.decode(out, skip_special_tokens=True) 
                              for out in outputs]
            
            # Decode the reference text
            reference_texts = [text_tokenizer.decode(ref, skip_special_tokens=True) 
                              for ref in example_batch["text_input_ids"]]
        
            # Log the examples
            log_batch_examples(
                batch=example_batch,
                generated_texts=generated_texts,
                reference_texts=reference_texts,
                tokenizer=dna_tokenizer,  # For decoding DNA sequences
                prefix=f"epoch_{epoch+1}",
                max_examples=3  # Number of examples to log
            )

        # Generate DNA sequences from text inputs (reverse direction)
        with torch.no_grad():
            model.eval()
            # Your model must support generating DNA from text
            dna_outputs = model.generate_dna(
                input_ids=example_batch["text_input_ids"],
                attention_mask=example_batch["text_attention_mask"],
                max_new_tokens=150,  # DNA sequences might be longer
                num_beams=4
            )
            
            # Decode the generated DNA sequences
            generated_dna = [dna_tokenizer.decode(out, skip_special_tokens=True) 
                            for out in dna_outputs]
            
            # Decode the reference DNA sequences
            reference_dna = [dna_tokenizer.decode(ref, skip_special_tokens=True) 
                            for ref in example_batch["dna_input_ids"]]
            
            # Create a modified batch with swapped fields for Text→DNA direction
            # This works with your existing log_batch_examples implementation
            text2dna_batch = {
                "dna_input_ids": example_batch["text_input_ids"],  # Text is the "input" for this direction
                "text_input_ids": example_batch["dna_input_ids"],  # DNA is the "target" for this direction
                # Include any other required fields
                "dna_attention_mask": example_batch["text_attention_mask"],
                "text_attention_mask": example_batch["dna_attention_mask"]
            }
            
            # Log the Text→DNA examples
            log_batch_examples(
                batch=text2dna_batch,
                generated_texts=generated_dna,  # Generated DNA sequences
                reference_texts=reference_dna,  # Reference DNA sequences
                tokenizer=text_tokenizer,  # For decoding text sequences (the "DNA" in this direction)
                prefix=f"epoch_{epoch+1}_text2dna",  # Different prefix to distinguish direction
                max_examples=3  # Number of examples to log
            )

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