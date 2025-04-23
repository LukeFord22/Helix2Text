"""Standalone evaluation script for trained DNA-Text fusion models."""

import argparse
import torch
import yaml
import os

from ..configs.default_config import load_config
from ..models.dna_text_fusion import load_models, DNATextFusion
from ..data.dataset import get_data_loaders
from ..training.evaluation import evaluate_model
from ..utils.logging import setup_wandb

def main():
    """Run evaluation on a trained model."""
    parser = argparse.ArgumentParser(description="Evaluate DNA-Text fusion model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples to display")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    setup_wandb(config)
    
    # Load tokenizers and models
    dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
        dna_model_path=config.model.dna_model_path,
        text_model_path=config.model.text_model_path,
        device=device
    )
    
    # Create data loaders (only need test loader for evaluation)
    _, test_loader = get_data_loaders(
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
    
    # Load trained weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        text_tokenizer=text_tokenizer,
        dna_tokenizer=dna_tokenizer,
        device=device,
        current_epoch="evaluation",
        show_examples=args.examples
    )
    
    # Print metrics
    print("\n✨ Evaluation Results:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()