"""Evaluation metrics and visualization for DNA-text fusion models."""

import torch
import random
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_model(model, test_loader, text_tokenizer, dna_tokenizer, device, current_epoch=None, show_examples=5, batch_size=4):
    """
    Evaluate the model on DNA→Text and Text→DNA generation tasks with memory optimizations.
    
    Args:
        model: The DNATextFusion model to evaluate
        test_loader: DataLoader with test data
        text_tokenizer: Tokenizer for text sequences 
        dna_tokenizer: Tokenizer for DNA sequences
        device: Device to run evaluation on
        current_epoch: Current epoch number for logging
        show_examples: Number of examples to visualize
        batch_size: Maximum batch size for memory efficiency
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_text_preds, all_text_refs = [], []  # For DNA→Text
    dna_from_text_preds, dna_from_text_refs = [], []  # For Text→DNA
    example_pairs_dna_to_text = []  # DNA→Text examples
    example_pairs_text_to_dna = []  # Text→DNA examples

    # Import garbage collection
    import gc
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Process in smaller sub-batches for memory efficiency if needed
            actual_batch_size = batch["dna_input_ids"].size(0)
            
            if actual_batch_size > batch_size:
                # Process large batches in chunks
                sub_batches = []
                for i in range(0, actual_batch_size, batch_size):
                    end_idx = min(i + batch_size, actual_batch_size)
                    sub_batch = {
                        k: v[i:end_idx] for k, v in batch.items()
                    }
                    sub_batches.append(sub_batch)
            else:
                # Process as a single batch
                sub_batches = [batch]
            
            # Process each sub-batch
            for sub_batch in sub_batches:
                # Prepare inputs
                dna_inputs = {
                    "input_ids": sub_batch["dna_input_ids"].to(device),
                    "attention_mask": sub_batch["dna_attention_mask"].to(device),
                }
                text_inputs = {
                    "input_ids": sub_batch["text_input_ids"].to(device),
                    "attention_mask": sub_batch["text_attention_mask"].to(device),
                }
                
                # 1. Evaluate DNA→Text generation
                try:
                    generated_text_ids = model.generate_text_from_dna(
                        dna_inputs, 
                        max_length=128,
                        top_k=50,
                        top_p=0.9
                    )
                    generated_texts = text_tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)
                    reference_texts = text_tokenizer.batch_decode(sub_batch["text_input_ids"], skip_special_tokens=True)
                    
                    all_text_preds.extend(generated_texts)
                    all_text_refs.extend(reference_texts)
                    example_pairs_dna_to_text.extend(list(zip(reference_texts, generated_texts)))
                    
                    # Free memory
                    del generated_text_ids, generated_texts, reference_texts
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"Error during DNA→Text generation: {e}")
                
                # 2. Evaluate Text→DNA generation 
                try:
                    generated_dna_ids = model.generate_dna_from_text(
                        text_inputs, 
                        max_length=512,
                        top_k=50,
                        top_p=0.9
                    )
                    generated_dnas = dna_tokenizer.batch_decode(generated_dna_ids, skip_special_tokens=True)
                    reference_dnas = dna_tokenizer.batch_decode(sub_batch["dna_input_ids"], skip_special_tokens=True)
                    
                    dna_from_text_preds.extend(generated_dnas)
                    dna_from_text_refs.extend(reference_dnas)
                    example_pairs_text_to_dna.extend(list(zip(reference_dnas, generated_dnas)))
                    
                    # Free memory
                    del generated_dna_ids, generated_dnas, reference_dnas
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"Error during Text→DNA generation: {e}")
                
                # Clean up batch tensors
                del dna_inputs, text_inputs
                torch.cuda.empty_cache()
            
            # Process only first few batches for faster evaluation
            if batch_idx >= 5:
                break

    # Create and log tables with examples for wandb
    if example_pairs_dna_to_text:
        try:
            dna_to_text_table = wandb.Table(columns=["Reference Text", "Generated Text"])
            for ref, pred in random.sample(example_pairs_dna_to_text, k=min(show_examples, len(example_pairs_dna_to_text))):
                dna_to_text_table.add_data(ref, pred)
            wandb.log({f"dna_to_text_examples/epoch_{current_epoch}": dna_to_text_table})
        except Exception as e:
            print(f"Error logging DNA→Text examples to wandb: {e}")
    
    if example_pairs_text_to_dna:
        try:
            text_to_dna_table = wandb.Table(columns=["Reference DNA", "Generated DNA"])
            for ref, pred in random.sample(example_pairs_text_to_dna, k=min(show_examples, len(example_pairs_text_to_dna))):
                # Truncate long DNA sequences for better visualization
                ref_display = ref[:100] + "..." if len(ref) > 100 else ref
                pred_display = pred[:100] + "..." if len(pred) > 100 else pred
                text_to_dna_table.add_data(ref_display, pred_display)
            wandb.log({f"text_to_dna_examples/epoch_{current_epoch}": text_to_dna_table})
        except Exception as e:
            print(f"Error logging Text→DNA examples to wandb: {e}")

    # Calculate metrics for DNA→Text
    metrics = {}
    
    # BLEU score
    try:
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        for ref, pred in zip(all_text_refs, all_text_preds):
            try:
                score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                bleu_scores.append(score)
            except Exception as e:
                print(f"BLEU calculation error: {e}")
        
        if bleu_scores:
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            metrics["bleu_score"] = avg_bleu
    except Exception as e:
        print(f"Error calculating BLEU scores: {e}")
    
    # ROUGE scores - process in smaller chunks to save memory
    try:
        # Process ROUGE in batches to save memory
        max_rouge_batch = 50
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []
        
        for i in range(0, len(all_text_refs), max_rouge_batch):
            batch_refs = all_text_refs[i:i+max_rouge_batch]
            batch_preds = all_text_preds[i:i+max_rouge_batch]
            
            rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(batch_refs, batch_preds)]
            
            all_rouge1.extend([score['rouge1'].fmeasure for score in rouge_scores])
            all_rouge2.extend([score['rouge2'].fmeasure for score in rouge_scores])
            all_rougeL.extend([score['rougeL'].fmeasure for score in rouge_scores])
            
            # Clean up batch
            del batch_refs, batch_preds, rouge_scores
            gc.collect()
        
        if all_rouge1:
            metrics["rouge1"] = sum(all_rouge1) / len(all_rouge1)
            metrics["rouge2"] = sum(all_rouge2) / len(all_rouge2)
            metrics["rougeL"] = sum(all_rougeL) / len(all_rougeL)
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
    
    # DNA similarity metrics - process in memory-efficient way
    if dna_from_text_preds:
        try:
            # Process in batches for memory efficiency
            max_dna_batch = 50
            dna_exact_matches = 0
            nuc_matches = []
            
            for i in range(0, len(dna_from_text_refs), max_dna_batch):
                batch_refs = dna_from_text_refs[i:i+max_dna_batch]
                batch_preds = dna_from_text_preds[i:i+max_dna_batch]
                
                # Count exact matches
                dna_exact_matches += sum(pred == ref for pred, ref in zip(batch_preds, batch_refs))
                
                # Calculate nucleotide-level accuracy
                for pred, ref in zip(batch_preds, batch_refs):
                    # Trim to same length for comparison
                    min_len = min(len(pred), len(ref))
                    if min_len > 0:
                        matches = sum(p == r for p, r in zip(pred[:min_len], ref[:min_len]))
                        nuc_matches.append(matches / min_len)
                    else:
                        nuc_matches.append(0)
                
                # Clean up batch
                del batch_refs, batch_preds
                gc.collect()
            
            if dna_from_text_refs:
                metrics["dna_exact_match"] = dna_exact_matches / len(dna_from_text_refs)
            
            if nuc_matches:
                metrics["nucleotide_accuracy"] = sum(nuc_matches) / len(nuc_matches)
        except Exception as e:
            print(f"DNA metrics calculation error: {e}")
    
    # Free memory
    del all_text_preds, all_text_refs, dna_from_text_preds, dna_from_text_refs
    del example_pairs_dna_to_text, example_pairs_text_to_dna
    gc.collect()
    torch.cuda.empty_cache()
    
    # Log metrics with epoch prefix if provided
    if current_epoch is not None:
        try:
            epoch_metrics = {f"{k}/epoch_{current_epoch}": v for k, v in metrics.items()}
            wandb.log(epoch_metrics)
        except Exception as e:
            print(f"Error logging epoch metrics: {e}")
    
    # Also log without prefix for nice curves
    try:
        wandb.log(metrics)
    except Exception as e:
        print(f"Error logging metrics: {e}")
    
    # Print summary
    print(f"🧠 Evaluation Scores:")
    print(f"  BLEU: {metrics.get('bleu_score', 0):.4f}")
    print(f"  ROUGE-1: {metrics.get('rouge1', 0):.4f}")
    print(f"  ROUGE-L: {metrics.get('rougeL', 0):.4f}")
    if 'nucleotide_accuracy' in metrics:
        print(f"  DNA Nucleotide Accuracy: {metrics.get('nucleotide_accuracy', 0):.4f}")
    
    return metrics