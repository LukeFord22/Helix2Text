"""Main training script for DNA-Text fusion models."""

import os
import argparse
import yaml
import time
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
import sys
import io
from datetime import datetime

# Import from our modular structure
from ..configs.default_config import Config, default_config
from ..data.dataset import get_data_loaders
from ..models.dna_text_fusion import DNATextFusion, load_models
from ..losses.contrastive import DNATextContrastiveLoss
from ..training.evaluation import evaluate_model
from ..utils.optimization import create_optimizer_and_scheduler
from ..utils.logging import log_batch_examples_bidirectional

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

def create_ddp_compatible_progress_bar(iterator, desc, is_main_process, local_rank):
    """
    Create a progress bar that's compatible with distributed training.
    Only shows on the main process with proper configuration.
    
    Args:
        iterator: Iterable to track progress over
        desc: Description for the progress bar
        is_main_process: Whether this is the main process (rank 0)
        local_rank: The rank of the current process
    
    Returns:
        A tqdm progress bar or the original iterator
    """
    if not is_main_process:
        return iterator
    
    # Create a log file for this specific rank's output
    # This prevents terminal corruption when multiple processes write simultaneously
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up tqdm with appropriate settings for DDP
    progress_bar = tqdm(
        iterator,
        desc=desc,
        ncols=100,
        leave=True,
        position=0,  # Main process always has position 0
        dynamic_ncols=False,
        disable=False,
        mininterval=1.0,  # Less frequent updates to prevent output corruption
        file=sys.stdout,  # Output to stdout for main process
        smoothing=0.1
    )
    
    # Clear the current line before starting the progress bar (helps with DDP)
    print("\033[K", end="", flush=True)
    
    return progress_bar

def train_model(config=None):
    """Main training function for DNA-Text fusion models."""
    # Load configuration
    print("Loading configuration...")
    config = config or default_config
    
    # Check which distributed mode we're using
    using_deepspeed = hasattr(config, 'deepspeed') and config.deepspeed.use_deepspeed
    using_ddp = hasattr(config, 'ddp_runtime') and config.ddp_runtime is not None
    using_distributed = hasattr(config, 'distributed') and config.distributed.get('enabled', False)
    
    # Set up DeepSpeed config path
    deepspeed_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "configs", "ds.config.json")
    
    # Determine device and distributed settings
    if using_deepspeed:
        # Extract DeepSpeed configuration
        model_engine = config.deepspeed_runtime.get('model_engine')
        local_rank = config.deepspeed_runtime.get('local_rank', -1)
        is_main_process = local_rank <= 0
        device = model_engine.device if model_engine else torch.device('cuda')
        print(f"Using DeepSpeed with device: {device}, local_rank: {local_rank}")
    elif using_ddp or using_distributed:
        # Extract DDP configuration
        if using_ddp:
            # If passing the model directly via ddp_runtime
            model = config.ddp_runtime.get('model')
            local_rank = config.ddp_runtime.get('rank', -1)
            device = config.ddp_runtime.get('device', torch.device('cuda'))
        else:
            # If using the distributed flag only
            local_rank = config.distributed.get('rank', -1)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        is_main_process = local_rank == 0
        print(f"Using PyTorch DDP with device: {device}, rank: {local_rank}")
    else:
        # Standard single-GPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
        local_rank = -1
        print(f"Using standard training with device: {device}")
    
    # Set random seed for reproducibility
    set_seed(42 + (local_rank if local_rank >= 0 else 0))
    
    # Initialize wandb only on main process with DDP-compatible settings
    if is_main_process:
        # Create a unique run group ID that can be shared across processes if needed
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Set the WANDB_RUN_GROUP environment variable for potential process coordination
        os.environ["WANDB_RUN_GROUP"] = f"ddp_training_{run_id}"
        
        # Make sure wandb doesn't reinitialize if a run already exists
        if not wandb.run:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config={
                    "training": vars(config.training),
                    "model": vars(config.model),
                    "data": vars(config.data),
                    "distributed": {
                        "enabled": using_ddp or using_distributed,
                        "world_size": config.ddp_runtime.get('world_size', 1) if using_ddp 
                                    else config.distributed.get('world_size', 1) if using_distributed 
                                    else 1
                    }
                }
            )
    else:
        # For non-main processes, disable wandb completely
        os.environ["WANDB_MODE"] = "disabled"
    
    # For DeepSpeed, we extract from runtime config
    if using_deepspeed:
        model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        optimizer = model_engine.optimizer
        scheduler = None  # DeepSpeed handles this
        
        # Get data loaders from the config
        dna_tokenizer = config.deepspeed_runtime.get('dna_tokenizer')
        text_tokenizer = config.deepspeed_runtime.get('text_tokenizer')
        train_loader = config.deepspeed_runtime.get('train_loader')
        test_loader = config.deepspeed_runtime.get('test_loader')
    
    # For DDP, we extract from runtime config
    elif using_ddp:
        # The model is already wrapped with DDP
        model = config.ddp_runtime.get('model')
        dna_tokenizer = config.ddp_runtime.get('dna_tokenizer')
        text_tokenizer = config.ddp_runtime.get('text_tokenizer')
        
        # Create data loaders with distributed sampler
        from torch.utils.data.distributed import DistributedSampler
        
        # Create data loaders
        train_loader, test_loader = get_data_loaders(
            dna_tokenizer=dna_tokenizer,
            text_tokenizer=text_tokenizer,
            config=config,
            rank=local_rank,
            world_size=config.ddp_runtime.get('world_size', 1),
            use_distributed_sampler=True
        )
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            lr=config.training.learning_rate,
            steps_per_epoch=len(train_loader),
            num_epochs=config.training.num_epochs,
            grad_accum_steps=config.training.grad_accum_steps
        )
    
    # Standard training setup
    else:
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
    print("Starting training...")
    for epoch in range(config.training.num_epochs):
        # Set sampler epoch for distributed training
        if using_ddp and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        # Set model to training mode
        if using_deepspeed:
            model_engine.train()
        else:
            model.train()
            
        total_loss = 0.0
        grad_norms = []
        max_step_times = 100
        step_times = []
        
        # Initialize progress bar (only on main process)
        progress_bar = create_ddp_compatible_progress_bar(
            iterator=train_loader,
            desc=f"Epoch {epoch+1}/{config.training.num_epochs}",
            is_main_process=is_main_process,
            local_rank=local_rank
        )
        
        epoch_start_time = time.time()
        
        # Training steps
        for step, batch in enumerate(progress_bar):
            # log time
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

            # Forward pass - different for different distributed modes
            if using_deepspeed:
                outputs = model_engine(dna_inputs=dna_inputs, text_inputs=text_inputs)
            else:
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
            )
            
            # Apply gradient accumulation for standard and DDP training
            if not using_deepspeed:
                # Scale loss for gradient accumulation
                loss = loss / config.training.grad_accum_steps
                loss.backward()
                
                # Step optimizer every grad_accum_steps
                if (step + 1) % config.training.grad_accum_steps == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_norms.append(grad_norm.item())
                    
                    # Regular optimizer step
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            else:
                # DeepSpeed handles gradient accumulation internally
                model_engine.backward(loss)
                model_engine.step()
                # Store gradient norm for logging if available
                if hasattr(model_engine, 'get_grad_norm'):
                    grad_norm = model_engine.get_grad_norm()
                    grad_norms.append(grad_norm)
            
            # Track loss
            total_loss += loss.item()
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            # Update timing and metrics only on main process
            if is_main_process:
                # Update step times list
                step_times.append(step_duration)
                if len(step_times) > max_step_times:
                    step_times.pop(0)  # Remove oldest timing

                # Calculate ETA based on recent average step time
                avg_step_time = sum(step_times) / len(step_times)
                
                # Calculate total steps and remaining steps for the entire training process
                total_training_steps = config.training.num_epochs * len(train_loader)
                steps_done = (epoch * len(train_loader)) + step + 1
                remaining_steps = total_training_steps - steps_done
                
                # Calculate ETA for the entire training process
                eta_seconds = remaining_steps * avg_step_time
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                # Format ETA for display
                if eta_hours > 1:
                    eta_display = f"{eta_hours:.1f}h"
                else:
                    eta_display = f"{eta_minutes:.1f}m"

                # Log step metrics
                if wandb.run:
                    step_metrics = {
                        "step_loss": loss.item(),
                        "DNA": dna_ce_loss.item(),
                        "Text": text_ce_loss.item(),
                        **contrastive_metrics,
                        "step": steps_done,
                        "eta_min": eta_minutes,
                        "total_progress": steps_done / total_training_steps,
                        "progress_percent": (steps_done / total_training_steps) * 100
                    }
                    
                    # Add learning rate if available
                    if not using_deepspeed and optimizer:
                        step_metrics["learning_rate"] = optimizer.param_groups[0]['lr']
                    
                    if len(grad_norms) > 0:
                        step_metrics["grad_norm"] = grad_norms[-1]
                        
                    wandb.log(step_metrics)
                
                # Update progress bar
                # Add indicator for gradient accumulation progress
                accum_step = (step % config.training.grad_accum_steps) + 1
                accum_percent = accum_step / config.training.grad_accum_steps * 100
                
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "DNA": f"{dna_ce_loss.item():.4f}",
                    "Text": f"{text_ce_loss.item():.4f}",
                    "Cont": f"{contrastive_loss.item():.4f}",
                    "Accum": f"{accum_step}/{config.training.grad_accum_steps} ({accum_percent:.0f}%)",
                    "ETA": eta_display,
                    "Progress": f"{steps_done}/{total_training_steps} ({(steps_done/total_training_steps)*100:.1f}%)"
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1) if grad_norms else 0
        epoch_time = time.time() - epoch_start_time
        
        # Synchronize distributed processes before evaluation
        if using_ddp or using_distributed:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        # Only evaluate on main process
        if is_main_process:
            # Get model for evaluation (unwrap from distributed wrappers if needed)
            if using_deepspeed:
                eval_model = model_engine.module
            elif using_ddp:
                # For DDP, we need to use the module attribute to get the base model
                if hasattr(model, 'module'):
                    eval_model = model.module
                else:
                    eval_model = model
            else:
                eval_model = model
            
            # Evaluate model on test set
            eval_metrics = evaluate_model(
                model=eval_model,
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
                eval_model.eval()
                outputs = eval_model.generate(
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
                if wandb.run:
                    log_batch_examples_bidirectional(
                        batch=example_batch,
                        generated_texts=generated_texts,
                        reference_texts=reference_texts,
                        tokenizer=dna_tokenizer,
                        prefix=f"epoch_{epoch+1}",
                        max_examples=3,
                        direction="dna2text"
                    )

            # Generate DNA sequences from text inputs (reverse direction)
            with torch.no_grad():
                eval_model.eval()
                dna_outputs = eval_model.generate_dna(
                    input_ids=example_batch["text_input_ids"],
                    attention_mask=example_batch["text_attention_mask"],
                    max_new_tokens=150,
                    num_beams=4
                )
                
                # Decode the generated DNA sequences
                generated_dna = [dna_tokenizer.decode(out, skip_special_tokens=True) 
                                for out in dna_outputs]
                
                # Decode the reference DNA sequences
                reference_dna = [dna_tokenizer.decode(ref, skip_special_tokens=True) 
                                for ref in example_batch["dna_input_ids"]]
                
                # Create a modified batch with swapped fields for Text→DNA direction
                text2dna_batch = {
                    "dna_input_ids": example_batch["text_input_ids"],
                    "text_input_ids": example_batch["dna_input_ids"],
                    "dna_attention_mask": example_batch["text_attention_mask"],
                    "text_attention_mask": example_batch["dna_attention_mask"]
                }
                
                # Log the Text→DNA examples
                if wandb.run:
                    log_batch_examples_bidirectional(
                        batch=text2dna_batch,
                        generated_texts=generated_dna,
                        reference_texts=reference_dna,
                        tokenizer=text_tokenizer,
                        prefix=f"epoch_{epoch+1}_text2dna",
                        max_examples=3,
                        direction="text2dna"
                    )

            # Log epoch metrics
            if wandb.run:
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
                
                # Save model differently depending on training mode
                if using_deepspeed:
                    # Save using DeepSpeed engine's save_checkpoint method
                    if hasattr(model_engine, 'save_checkpoint'):
                        checkpoint_path = os.path.dirname(config.output_path)
                        model_engine.save_checkpoint(checkpoint_path)
                    else:
                        # Fallback to standard save for the module
                        torch.save(model_engine.module.state_dict(), config.output_path)
                elif using_ddp:
                    # For DDP, save the unwrapped model
                    if hasattr(model, 'module'):
                        torch.save(model.module.state_dict(), config.output_path)
                    else:
                        torch.save(model.state_dict(), config.output_path)
                else:
                    # Standard PyTorch save
                    torch.save(model.state_dict(), config.output_path)
                    
                if wandb.run:
                    wandb.log({"best_model_saved": True})
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.training.early_stop_patience:
                    print("⏹️ Early stopping triggered.")
                    if wandb.run:
                        wandb.log({"early_stopped": True})
                    break
    
    # Synchronize processes before final evaluation
    if using_ddp or using_distributed:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    # Final evaluation - only on main process
    if is_main_process:
        print("\nTraining complete. Loading best model for final evaluation...")
        
        # Get the appropriate model for evaluation based on training mode
        if using_deepspeed:
            eval_model = model_engine.module
        elif using_ddp:
            eval_model = model.module if hasattr(model, 'module') else model
        else:
            eval_model = model
        
        try:
            if using_deepspeed:
                # For DeepSpeed, we try to load from the saved checkpoint
                checkpoint_path = os.path.dirname(config.output_path)
                checkpoint_tag = os.path.basename(config.output_path).split('.')[0]
                success = model_engine.load_checkpoint(
                    checkpoint_dir=checkpoint_path,
                    tag=checkpoint_tag
                )
                if not success:
                    print("Warning: Failed to load DeepSpeed checkpoint, using standard loading")
                    eval_model.load_state_dict(torch.load(config.output_path))
            else:
                # Standard PyTorch load
                eval_model.load_state_dict(torch.load(config.output_path))
        except Exception as e:
            print(f"Error loading best model: {str(e)}")
            print("Continuing with current model state")
        
        final_metrics = evaluate_model(
            model=eval_model,
            test_loader=test_loader,
            text_tokenizer=text_tokenizer,
            dna_tokenizer=dna_tokenizer,
            device=device,
            current_epoch="final"
        )
        
        # Log final metrics
        if wandb.run:
            wandb.log({
                "final_bleu": final_metrics["bleu_score"],
                "final_rouge1": final_metrics["rouge1"],
                "final_rouge2": final_metrics["rouge2"],
                "final_rougeL": final_metrics["rougeL"],
                "training_completed": True
            })
            wandb.finish()
        
        print(f"Final BLEU Score: {final_metrics['bleu_score']:.4f}")
        
        return eval_model, final_metrics
    
    # For non-main processes in distributed training
    if using_deepspeed:
        return model_engine.module, {}
    elif using_ddp:
        return model.module if hasattr(model, 'module') else model, {}
    else:
        return model, {}

def main():
    parser = argparse.ArgumentParser(description="Train DNA-Text fusion model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(config)

if __name__ == "__main__":
    main()