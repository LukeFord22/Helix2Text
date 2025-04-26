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
    Log examples of generated sequences to wandb with direction-specific formatting.
    
    Args:
        batch: Batch dictionary containing input_ids
        generated_texts: List of model-generated sequences
        reference_texts: List of ground truth sequences
        tokenizer: Tokenizer for decoding input sequences
        prefix: Prefix for the wandb table name (default: "train")
        max_examples: Maximum number of examples to log (default: 3)
        direction: Translation direction - either "dna2text" or "text2dna" (default: "dna2text")
    
    Returns:
        None - logs data to wandb if wandb.run is active
    """
    # Early return if wandb isn't initialized
    if not wandb.run:
        print("Warning: wandb.run not found. Skipping example logging.")
        return
    
    try:
        # Set up direction-specific column headers for better readability
        if direction == "dna2text":
            columns = ["DNA Input", "Expected Text Output", "Model Generated Text"]
            input_key = "dna_input_ids"
        elif direction == "text2dna":
            columns = ["Text Input", "Expected DNA Output", "Model Generated DNA"]
            input_key = "dna_input_ids"  # In text2dna case, this contains text IDs
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'dna2text' or 'text2dna'")
        
        # Create wandb table with our columns
        table = wandb.Table(columns=columns)
        
        # Safety check for batch sizes
        if len(generated_texts) == 0:
            print(f"Warning: No generated texts to log for {direction}")
            return
            
        # Select random examples (or all if fewer than max_examples)
        num_samples = min(max_examples, len(generated_texts))
        indices = random.sample(range(len(generated_texts)), num_samples)
        
        # Process each selected example
        for idx in indices:
            # Decode the input sequence 
            source_seq = tokenizer.decode(batch[input_key][idx], skip_special_tokens=True)
            
            # Truncate long sequences for better display
            max_display_len = 150
            if len(source_seq) > max_display_len:
                source_seq = source_seq[:max_display_len] + "..."
                
            # Add all data to the table
            table.add_data(source_seq, reference_texts[idx], generated_texts[idx])
        
        # Log to wandb with a clear, descriptive name
        direction_name = "DNA_to_Text" if direction == "dna2text" else "Text_to_DNA"
        table_name = f"{prefix}_{direction_name}_examples"
        wandb.log({table_name: table})
        
    except Exception as e:
        print(f"Error logging examples to wandb: {str(e)}")