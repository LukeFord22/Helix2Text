"""Text and DNA generation script for DNA-Text fusion models."""

import argparse
import torch
import yaml
import os
import pandas as pd

from ..configs.default_config import load_config
from ..models.dna_text_fusion import load_models, DNATextFusion

def generate_text_from_dna(model, dna_tokenizer, text_tokenizer, dna_sequence, device, max_length=128):
    """
    Generate text description from a DNA sequence.
    
    Args:
        model: DNATextFusion model
        dna_tokenizer: DNA tokenizer
        text_tokenizer: Text tokenizer
        dna_sequence: DNA sequence as string
        device: Device to run generation on
        max_length: Maximum length of generated text
        
    Returns:
        Generated text description
    """
    model.eval()
    with torch.no_grad():
        # Tokenize DNA
        dna_inputs = dna_tokenizer(
            dna_sequence, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        dna_inputs = {k: v.to(device) for k, v in dna_inputs.items()}
        
        # Generate text
        generated_ids = model.generate_text_from_dna(
            dna_inputs, 
            max_length=max_length
        )
        
        # Decode generated text
        generated_text = text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text

def generate_dna_from_text(model, dna_tokenizer, text_tokenizer, text_description, device, max_length=512):
    """
    Generate DNA sequence from a text description.
    
    Args:
        model: DNATextFusion model
        dna_tokenizer: DNA tokenizer
        text_tokenizer: Text tokenizer
        text_description: Text description as string
        device: Device to run generation on
        max_length: Maximum length of generated DNA
        
    Returns:
        Generated DNA sequence
    """
    model.eval()
    with torch.no_grad():
        # Tokenize text
        text_inputs = text_tokenizer(
            text_description, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # Generate DNA
        generated_ids = model.generate_dna_from_text(
            text_inputs, 
            max_length=max_length
        )
        
        # Decode generated DNA
        generated_dna = dna_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_dna

def main():
    """Run generation script."""
    parser = argparse.ArgumentParser(description="Generate text/DNA using DNA-Text fusion model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--mode", type=str, choices=["dna2text", "text2dna", "batch"], default="dna2text", 
                        help="Generation mode: dna2text, text2dna, or batch")
    parser.add_argument("--input", type=str, help="Input DNA sequence or text description")
    parser.add_argument("--input_file", type=str, help="CSV file with DNA or text for batch processing")
    parser.add_argument("--output_file", type=str, help="Output file to save the results")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated sequence")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers and models
    dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
        dna_model_path=config.model.dna_model_path,
        text_model_path=config.model.text_model_path,
        device=device
    )
    
    # Initialize model
    model = DNATextFusion(
        dna_model=dna_model,
        text_model=text_model,
        cross_attn_interval=config.model.cross_attn_interval
    ).to(device)
    
    # Load trained weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Batch mode: process multiple inputs from a file
    if args.mode == "batch":
        if not args.input_file or not os.path.exists(args.input_file):
            raise ValueError("Input file not found. Please provide a valid CSV file path.")
        
        # Load the input file
        df = pd.read_csv(args.input_file)
        results = []
        
        # Check for required columns based on mode
        if "DNA_Sequence" in df.columns:
            print(f"Processing {len(df)} DNA sequences to generate text...")
            for idx, row in df.iterrows():
                dna_seq = row["DNA_Sequence"]
                generated_text = generate_text_from_dna(
                    model, dna_tokenizer, text_tokenizer, dna_seq, device, args.max_length
                )
                results.append({"DNA_Sequence": dna_seq, "Generated_Text": generated_text})
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} sequences")
                    
        elif "text" in df.columns or "Text" in df.columns:
            text_col = "text" if "text" in df.columns else "Text"
            print(f"Processing {len(df)} text descriptions to generate DNA...")
            for idx, row in df.iterrows():
                text = row[text_col]
                generated_dna = generate_dna_from_text(
                    model, dna_tokenizer, text_tokenizer, text, device, args.max_length
                )
                results.append({text_col: text, "Generated_DNA": generated_dna})
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} descriptions")
        else:
            raise ValueError("Input file must contain either 'DNA_Sequence' or 'text'/'Text' column.")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = args.output_file or "generation_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
    # Single input mode
    elif args.input:
        if args.mode == "dna2text":
            print(f"Generating text from DNA sequence: {args.input[:50]}...")
            generated_text = generate_text_from_dna(
                model, dna_tokenizer, text_tokenizer, args.input, device, args.max_length
            )
            print(f"\nGenerated Text:\n{'-' * 50}\n{generated_text}\n{'-' * 50}")
            
        elif args.mode == "text2dna":
            print(f"Generating DNA from text: {args.input[:50]}...")
            generated_dna = generate_dna_from_text(
                model, dna_tokenizer, text_tokenizer, args.input, device, args.max_length
            )
            print(f"\nGenerated DNA:\n{'-' * 50}\n{generated_dna[:100]}{'...' if len(generated_dna) > 100 else ''}\n{'-' * 50}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()