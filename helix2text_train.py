#!/usr/bin/env python3
"""
Helix2Text Training Script

A simplified script that loads configuration and starts model training.
All parameters are defined at the top of the file for easy modification.
"""

import os
import sys
import torch
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Helix2Text.MetaModel.configs.default_config import default_config
from Helix2Text.MetaModel.scripts.train import train_model

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Modify these parameters according to your needs

# Path Configuration
DATA_PATH = "protein_to_dna_sequences.csv"  # Path to your training data
OUTPUT_MODEL_PATH = "best_model.pt"  # Where to save the trained model

# Training Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
GRAD_ACCUM_STEPS = 4
EARLY_STOP_PATIENCE = 3

# Data Splitting Options
SPLIT_DATA = True  # Whether to split data into train/eval
TRAIN_DATA_FRACTION = 1.0  # Fraction of data to use (1.0 = use all data)
TRAIN_SPLIT_FRACTION = 0.9  # Fraction for training vs. evaluation

# Experiment Tracking
USE_WANDB = True  # Whether to use Weights & Biases for tracking
PROJECT_NAME = "Helix2Text"  # Same project name as your other files
EXPERIMENT_NAME = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Device Configuration
CUDA_DEVICE = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0+ for GPU

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Configure and start model training."""
    print(f"Starting Helix2Text training with data: {DATA_PATH}")
    
    # Set up configuration
    config = default_config
    
    # Update with our parameters
    config.project_name = PROJECT_NAME
    config.experiment_name = EXPERIMENT_NAME
    
    # Training configuration
    config.training.num_epochs = NUM_EPOCHS
    config.training.batch_size = BATCH_SIZE
    config.training.learning_rate = LEARNING_RATE
    config.training.early_stop_patience = EARLY_STOP_PATIENCE
    config.training.grad_accum_steps = GRAD_ACCUM_STEPS
    config.training.train_data_fraction = TRAIN_DATA_FRACTION
    config.training.train_split_fraction = TRAIN_SPLIT_FRACTION
    
    # Data configuration
    config.data.local_path = DATA_PATH
    config.data.split_data = SPLIT_DATA
    
    # Set output model path
    config.output_path = OUTPUT_MODEL_PATH
    
    # Disable wandb if not requested
    if not USE_WANDB:
        os.environ["WANDB_MODE"] = "disabled"
        print("Weights & Biases tracking disabled")
    else:
        print(f"Weights & Biases tracking enabled for project: {PROJECT_NAME}")
    
    # Set device
    if CUDA_DEVICE >= 0 and torch.cuda.is_available():
        device = f'cuda:{CUDA_DEVICE}'
        print(f"Using GPU: {device}")
    else:
        device = 'cpu'
        print("Using CPU for training")
    
    # Start training
    print(f"Starting training with {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
    model, metrics = train_model(config)
    
    print("\n============= Training Complete =============")
    print(f"Final BLEU Score: {metrics.get('bleu_score', 0):.4f}")
    print(f"Model saved to: {OUTPUT_MODEL_PATH}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())