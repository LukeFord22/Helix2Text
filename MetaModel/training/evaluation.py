"""Evaluation metrics and visualization for DNA-text fusion models."""

import torch
import random
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_model(model, test_loader, text_tokenizer, dna_tokenizer, device, current_epoch=None, show_examples=5):
    """
    Evaluate the model on DNA→Text and Text→DNA generation tasks.
    
    Args:
        model: The DNATextFusion model to evaluate
        test_loader: DataLoader with test data
        text_tokenizer: Tokenizer for text sequences 
        dna_tokenizer: Tokenizer for DNA sequences
        device: Device to run evaluation on
        current_epoch: Current epoch number for logging
        show_examples: Number of examples to visualize
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_text_preds, all_text_refs = [], []  # For DNA→Text
    dna_from_text_preds, dna_from_text_refs = [], []  # For Text→DNA
    example_pairs_dna_to_text = []  # DNA→Text examples
    example_pairs_text_to_dna = []  # Text→DNA examples

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Prepare inputs
            dna_inputs = {
                "input_ids": batch["dna_input_ids"].to(device),
                "attention_mask": batch["dna_attention_mask"].to(device),
            }
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
                "attention_mask": batch["text_attention_mask"].to(device),
            }
            
            # 1. Evaluate DNA→Text generation
            generated_text_ids = model.generate_text_from_dna(dna_inputs, max_length=128)
            generated_texts = text_tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)
            reference_texts = text_tokenizer.batch_decode(batch["text_input_ids"], skip_special_tokens=True)
            
            all_text_preds.extend(generated_texts)
            all_text_refs.extend(reference_texts)
            example_pairs_dna_to_text.extend(list(zip(reference_texts, generated_texts)))
            
            # 2. Evaluate Text→DNA generation 
            generated_dna_ids = model.generate_dna_from_text(text_inputs, max_length=512)
            generated_dnas = dna_tokenizer.batch_decode(generated_dna_ids, skip_special_tokens=True)
            reference_dnas = dna_tokenizer.batch_decode(batch["dna_input_ids"], skip_special_tokens=True)
            
            dna_from_text_preds.extend(generated_dnas)
            dna_from_text_refs.extend(reference_dnas)
            example_pairs_text_to_dna.extend(list(zip(reference_dnas, generated_dnas)))
            
            # Process only first few batches for faster evaluation
            if batch_idx >= 5:
                break

    # Create and log tables with examples for wandb
    if example_pairs_dna_to_text:
        dna_to_text_table = wandb.Table(columns=["Reference Text", "Generated Text"])
        for ref, pred in random.sample(example_pairs_dna_to_text, k=min(show_examples, len(example_pairs_dna_to_text))):
            dna_to_text_table.add_data(ref, pred)
        wandb.log({f"dna_to_text_examples/epoch_{current_epoch}": dna_to_text_table})
    
    if example_pairs_text_to_dna:
        text_to_dna_table = wandb.Table(columns=["Reference DNA", "Generated DNA"])
        for ref, pred in random.sample(example_pairs_text_to_dna, k=min(show_examples, len(example_pairs_text_to_dna))):
            # Truncate long DNA sequences for better visualization
            ref_display = ref[:100] + "..." if len(ref) > 100 else ref
            pred_display = pred[:100] + "..." if len(pred) > 100 else pred
            text_to_dna_table.add_data(ref_display, pred_display)
        wandb.log({f"text_to_dna_examples/epoch_{current_epoch}": text_to_dna_table})

    # Calculate metrics for DNA→Text
    metrics = {}
    
    # BLEU score
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
    
    # ROUGE scores
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(all_text_refs, all_text_preds)]
        
        avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
        
        metrics["rouge1"] = avg_rouge1
        metrics["rouge2"] = avg_rouge2
        metrics["rougeL"] = avg_rougeL
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
    
    # DNA similarity metrics
    if dna_from_text_preds:
        dna_exact_matches = sum(pred == ref for pred, ref in zip(dna_from_text_preds, dna_from_text_refs))
        dna_match_ratio = dna_exact_matches / len(dna_from_text_refs)
        metrics["dna_exact_match"] = dna_match_ratio
        
        # Nucleotide-level accuracy
        nuc_matches = []
        for pred, ref in zip(dna_from_text_preds, dna_from_text_refs):
            # Trim to same length for comparison
            min_len = min(len(pred), len(ref))
            matches = sum(p == r for p, r in zip(pred[:min_len], ref[:min_len]))
            nuc_matches.append(matches / min_len if min_len > 0 else 0)
        
        avg_nuc_accuracy = sum(nuc_matches) / len(nuc_matches) if nuc_matches else 0
        metrics["nucleotide_accuracy"] = avg_nuc_accuracy
    
    # Log metrics with epoch prefix if provided
    if current_epoch is not None:
        epoch_metrics = {f"{k}/epoch_{current_epoch}": v for k, v in metrics.items()}
        wandb.log(epoch_metrics)
    
    # Also log without prefix for nice curves
    wandb.log(metrics)
    
    # Print summary
    print(f"🧠 Evaluation Scores:")
    print(f"  BLEU: {metrics.get('bleu_score', 0):.4f}")
    print(f"  ROUGE-1: {metrics.get('rouge1', 0):.4f}")
    print(f"  ROUGE-L: {metrics.get('rougeL', 0):.4f}")
    if 'nucleotide_accuracy' in metrics:
        print(f"  DNA Nucleotide Accuracy: {metrics.get('nucleotide_accuracy', 0):.4f}")
    
    return metrics