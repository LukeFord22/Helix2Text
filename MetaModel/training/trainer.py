"""Trainer class for DNA-Text fusion models."""

import time
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import wandb

from ..losses.criterion import DNATextCriterion
from .evaluation import evaluate_model

class DNATextTrainer:
    """
    Trainer class for DNA-Text fusion models.
    
    This trainer handles the training loop, gradient accumulation,
    mixed precision, logging, and evaluation.
    """
    def __init__(self, model, optimizer, scheduler, config, device):
        """
        Initialize the trainer.
        
        Args:
            model: The DNATextFusion model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            config: Configuration object
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Initialize scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Initialize criterion
        self.criterion = DNATextCriterion(
            semantic_weight=config.training.semantic_loss_weight,
            total_epochs=config.training.num_epochs
        )
        
        # Training tracking variables
        self.best_score = 0
        self.epochs_no_improve = 0
        
    def train(self, train_loader, val_loader, text_tokenizer, dna_tokenizer):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            text_tokenizer: Tokenizer for text sequences
            dna_tokenizer: Tokenizer for DNA sequences
            
        Returns:
            The trained model
        """
        # Track global training start time for overall ETA
        global_start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            # Training epoch
            epoch_metrics = self.train_epoch(
                train_loader=train_loader, 
                epoch=epoch, 
                global_start_time=global_start_time,
                total_epochs=self.config.training.num_epochs
            )
            
            # Evaluation
            eval_metrics = evaluate_model(
                model=self.model,
                test_loader=val_loader,
                text_tokenizer=text_tokenizer,
                dna_tokenizer=dna_tokenizer,
                device=self.device,
                current_epoch=epoch+1
            )
            
            # Log combined metrics
            wandb.log({
                "epoch": epoch + 1,
                **epoch_metrics,
                **eval_metrics
            })
            
            # Print epoch summary
            print(f"[Epoch {epoch+1}/{self.config.training.num_epochs}] "
                  f"Loss: {epoch_metrics['avg_loss']:.4f} | "
                  f"Val BLEU: {eval_metrics['bleu_score']:.4f} | "
                  f"Time: {epoch_metrics['epoch_time']:.1f}s")
            
            # Early stopping logic
            if eval_metrics["bleu_score"] > self.best_score:
                self.best_score = eval_metrics["bleu_score"]
                self.epochs_no_improve = 0
                print("✨ New best model found!")
                self.save_checkpoint("best_model.pt")
                wandb.log({"best_model_saved": True})
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.training.early_stop_patience:
                    print("⏹️ Early stopping triggered.")
                    wandb.log({"early_stopped": True})
                    break
        
        # Final evaluation using best model
        print("\nTraining complete. Loading best model for final evaluation...")
        self.load_checkpoint("best_model.pt")
        final_metrics = evaluate_model(
            model=self.model,
            test_loader=val_loader,
            text_tokenizer=text_tokenizer,
            dna_tokenizer=dna_tokenizer,
            device=self.device,
            current_epoch="final"
        )
        
        return self.model, final_metrics
    
    def train_epoch(self, train_loader, epoch, global_start_time, total_epochs):
        """
        Run one training epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            global_start_time: Training start time for overall ETA calculation
            total_epochs: Total number of epochs for ETA calculation
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        grad_norms = []
        
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        epoch_start_time = time.time()
        
        # Reset gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Training steps
        for step, batch in enumerate(progress_bar):
            # Move inputs to device
            dna_inputs = {k: v.to(self.device) for k, v in batch.items() if "dna" in k}
            text_inputs = {k: v.to(self.device) for k, v in batch.items() if "text" in k}
            
            # Targets for loss calculation
            targets = {
                "target_ids": batch["dna_input_ids"].to(self.device)
            }
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(dna_inputs=dna_inputs, text_inputs=text_inputs)
                loss, loss_dict = self.criterion(outputs, targets, epoch=epoch)
                loss = loss / self.config.training.grad_accum_steps
                
            # Backward pass with gradient accumulation
            self.scaler.scale(loss).backward()
            
            # Step optimizer every grad_accum_steps
            if (step + 1) % self.config.training.grad_accum_steps == 0 or (step + 1 == len(train_loader)):
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_norms.append(grad_norm.item())
                
                # Step optimizer and scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Track loss
            total_loss += loss.item() * self.config.training.grad_accum_steps
            
            # Calculate ETA for entire training process
            current_time = time.time()
            elapsed_time = current_time - global_start_time
            
            # Calculate progress through entire training
            total_steps = total_epochs * len(train_loader)
            current_global_step = epoch * len(train_loader) + step + 1
            progress_fraction = current_global_step / total_steps
            
            # Calculate ETA based on global progress
            if progress_fraction > 0:
                estimated_total_time = elapsed_time / progress_fraction
                remaining_time = estimated_total_time - elapsed_time
                eta_hours = remaining_time / 3600
                eta_minutes = (remaining_time % 3600) / 60
                eta_display = f"{int(eta_hours)}h {int(eta_minutes)}m"
            else:
                eta_display = "Calculating..."
            
            # Prepare logging metrics
            step_metrics = {
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "step": current_global_step,
                "total_eta_minutes": remaining_time / 60 if progress_fraction > 0 else None,
                "total_progress_percent": progress_fraction * 100,
                **{f"step_{k}": v for k, v in loss_dict.items()}
            }
            
            if len(grad_norms) > 0:
                step_metrics["grad_norm"] = grad_norms[-1]
            
            # Log step metrics
            wandb.log(step_metrics)
            
            # Update progress bar with both current loss and global ETA
            progress_bar.set_postfix({
                "loss": loss_dict["total_loss"],
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "ETA": eta_display
            })
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
        
        return {
            "avg_loss": avg_loss,
            "avg_grad_norm": avg_grad_norm,
            "epoch_time": epoch_time
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "config": self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_score = checkpoint["best_score"]