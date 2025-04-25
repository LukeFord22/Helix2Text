"""Logging utilities for DNA-text fusion models."""

import wandb
import random

def setup_wandb(config):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        config: Configuration object containing wandb settings
    """
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config={
            "training": vars(config.training),
            "model": vars(config.model),
            "data": vars(config.data)
        }
    )
    
    return wandb.run

def log_batch_examples_bidirectional(batch, generated_texts, reference_texts, tokenizer, 
                                    prefix="train", max_examples=3, direction="dna2text"):
    """
    Log examples of generated text to wandb.
    
    Args:
        batch: Batch of data
        generated_texts: List of generated texts
        reference_texts: List of reference texts
        tokenizer: Tokenizer for decoding
        prefix: Prefix for logging table name
        max_examples: Maximum number of examples to log
        direction: Either "dna2text" or "text2dna"
    """
    if not wandb.run:
        return
    
    # Create table with direction-appropriate column names
    if direction == "dna2text":
        columns = ["DNA Sequence", "Reference Text", "Generated Text"]
    else:  # text2dna
        columns = ["Text Description", "Reference DNA", "Generated DNA"]
    
    table = wandb.Table(columns=columns)
    
    # Get a sample of examples
    indices = random.sample(range(len(generated_texts)), min(max_examples, len(generated_texts)))
    
    for idx in indices:
        # Get source sequence (DNA or text)
        if direction == "dna2text":
            source_seq = tokenizer.decode(batch["dna_input_ids"][idx], skip_special_tokens=True)
            # Truncate long DNA sequence for readability
            if len(source_seq) > 100:
                source_seq = source_seq[:100] + "..."
        else:
            source_seq = tokenizer.decode(batch["dna_input_ids"][idx], skip_special_tokens=True)
            
        # Add row to table
        table.add_data(source_seq, reference_texts[idx], generated_texts[idx])
    
    # Log table to wandb
    wandb.log({f"{prefix}_examples": table})