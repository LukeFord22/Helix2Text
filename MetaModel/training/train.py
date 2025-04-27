# Core modules
import torch
import wandb
import os, time
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed import get_rank, is_initialized

# Helix2Text modules
from .contrastive import DNATextContrastiveLoss
from .evaluation import evaluate_model
from .setup import setup_training
from ..utils.logging import log_batch_examples_bidirectional


def train_model(config=None):
    """Main training function for DNA-Text fusion models."""

    # Load configuration
    model_engine, optimizer, scheduler, train_loader, test_loader, device, local_rank, dna_tokenizer, text_tokenizer = setup_training(config)

    # Initialize loss
    contrastive_loss_fn = DNATextContrastiveLoss(
        initial_temp=0.5,
        min_temp=0.07,
        total_epochs=config.training.num_epochs,
        weight=config.training.semantic_loss_weight
    )
    
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training tracking variables
    best_bleu = 0
    epochs_no_improve = 0
    
    # Start training loop
    print("Starting training...")
    for epoch in range(config.training.num_epochs):
            
        # Set model to training mode
        model_engine.train()
        epoch_start_time = time.time()

        # Initialize variables
        max_step_times = 100
        total_loss = 0.0
        step_times = []
        grad_norms = []

        #progress bar for training
        is_main_process = not is_initialized() or get_rank() == 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}") if is_main_process else train_loader


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

            outputs = model_engine(dna_inputs=dna_inputs, text_inputs=text_inputs)

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
            
            # DeepSpeed handles gradient accumulation internally
            model_engine.backward(loss)
            model_engine.step()
            grad_norm = model_engine.get_grad_norm()
            if grad_norm is None:
                grad_norm = 0.0
            
            # Track loss
            total_loss += loss.item()
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            # Update step times list
            step_times.append(step_duration)
            if len(step_times) > max_step_times:
                step_times.pop(0)  # Remove oldest timing

            # Calculate ETA based on recent average step time
            avg_step_time = sum(step_times) / len(step_times)
            total_training_steps = config.training.num_epochs * len(train_loader)
            steps_done = (epoch * len(train_loader)) + step + 1
            remaining_steps = total_training_steps - steps_done
            eta_seconds = remaining_steps * avg_step_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            # Format ETA for display
            if eta_hours > 1:
                eta_display = f"{eta_hours:.1f}h"
            else:
                eta_display = f"{eta_minutes:.1f}m"

            # Log step metrics
            if is_main_process and wandb.run:
                wandb.log({
                "step_loss": loss.item(),
                "DNA": dna_ce_loss.item(),
                "Text": text_ce_loss.item(),
                **contrastive_metrics,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "step": steps_done,
                "grad_norm": grad_norm,
                "eta_min": eta_minutes,
                "total_progress": steps_done / total_training_steps,
                "progress_percent": (steps_done / total_training_steps) * 100
                })  
            
            
            # Add indicator for gradient accumulation progress
            accum_step = (step % config.training.grad_accum_steps) + 1
            accum_percent = accum_step / config.training.grad_accum_steps * 100
            
            # Update progress bar
            if is_main_process:
                accum_step = (step % config.training.grad_accum_steps) + 1
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "DNA": f"{dna_ce_loss.item():.4f}",
                    "Text": f"{text_ce_loss.item():.4f}",
                    "Cont": f"{contrastive_loss.item():.4f}",
                    "Accum": f"{accum_step}/{config.training.grad_accum_steps} ({accum_percent:.0f}%)",
                    "ETA": eta_display,
                    "Progress": f"{steps_done}/{total_training_steps} ({(steps_done/total_training_steps)*100:.1f}%)"
                })
            
        
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1) if grad_norms else 0
        epoch_time = time.time() - epoch_start_time
        

        # Evaluate model on test set
        eval_model = model_engine.module
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
            outputs = eval_model.generate_text_from_dna(
                dna_inputs={
                    "input_ids": example_batch["dna_input_ids"],
                    "attention_mask": example_batch["dna_attention_mask"]
                },
                max_length=100
            )
            
            # Decode the generated text
            generated_texts = [text_tokenizer.decode(out, skip_special_tokens=True) 
                            for out in outputs]
            
            # Decode the reference text
            reference_texts = [text_tokenizer.decode(ref, skip_special_tokens=True) 
                            for ref in example_batch["text_input_ids"]]
        
            # Log the examples
            if is_main_process and wandb.run:
                log_batch_examples_bidirectional(
                    batch=example_batch,
                    generated_texts=generated_texts,
                    reference_texts=reference_texts,
                    tokenizer=dna_tokenizer,
                    prefix=f"epoch_{epoch+1}_dna2text",
                    max_examples=3,
                    direction="dna2text"
                )

        # Generate DNA sequences from text inputs (reverse direction)
        with torch.no_grad():
            eval_model.eval()
            dna_outputs = eval_model.generate_dna_from_text(
                text_inputs={
                    "input_ids": example_batch["text_input_ids"],
                    "attention_mask": example_batch["text_attention_mask"]
                },
                max_length=100
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
        if is_main_process and wandb.run:
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
        
        # Early stopping
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            epochs_no_improve = 0
            print("✨ New best model found!")
            if hasattr(model_engine, 'save_checkpoint'):
                if dist.is_initialized():
                    dist.barrier()
                checkpoint_path = os.path.dirname(config.output_path)
                model_engine.save_checkpoint(checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.training.early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break


    if dist.is_initialized():
        dist.barrier()
    
    eval_model = model_engine.module

    # Load best checkpoint for final evaluation
    if is_main_process:
        print("\nTraining complete. Loading best checkpoint...")
        checkpoint_path = os.path.dirname(config.output_path)
        checkpoint_tag = os.path.basename(config.output_path).split('.')[0]
        success = model_engine.load_checkpoint(checkpoint_dir=checkpoint_path, tag=checkpoint_tag)
        if not success:
            print("⚠️ Warning: Failed to load checkpoint!")

    
        final_metrics = evaluate_model(
            model=model_engine.module,
            test_loader=test_loader,
            text_tokenizer=text_tokenizer,
            dna_tokenizer=dna_tokenizer,
            device=device,
            current_epoch="final"
        )
        

        wandb.log({
            "final_bleu": final_metrics["bleu_score"],
            "final_rouge1": final_metrics["rouge1"],
            "final_rouge2": final_metrics["rouge2"],
            "final_rougeL": final_metrics["rougeL"],
            "training_completed": True
        })
        if wandb.run:
          wandb.finish()
    
        print(f"Final BLEU Score: {final_metrics['bleu_score']:.4f}")
        
        return eval_model, final_metrics
    
    return model_engine.module, {}

