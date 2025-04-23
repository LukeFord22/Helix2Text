"""
Directory structure for modular Helix2Text implementation:

Helix2Text/
├── MetaModel/                      # Main package directory
│   ├── __init__.py                 # Package initialization
│   ├── configs/                    # Configuration files
│   │   ├── __init__.py
│   │   ├── default_config.py       # Default configuration values 
│   │   └── training_config.yaml    # YAML configuration for experiments
│   ├── data/                       # Data processing modules
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset creation and processing
│   │   └── tokenizers.py           # Tokenizer handling and utilities
│   ├── models/                     # Model architecture definitions
│   │   ├── __init__.py
│   │   ├── dna_text_fusion.py      # Main fusion model architecture
│   │   ├── attention.py            # Cross-attention mechanisms
│   │   └── components.py           # Reusable model components
│   ├── losses/                     # Loss function implementations
│   │   ├── __init__.py
│   │   ├── contrastive.py          # InfoNCE and dual softmax losses
│   │   └── criterion.py            # Combined loss functions
│   ├── training/                   # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop implementation
│   │   └── evaluation.py           # Evaluation metrics and visualization
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py              # WandB logging utilities
│   │   └── optimization.py         # Scheduler and optimizer utilities
│   └── scripts/                    # Runnable scripts
│       ├── train.py                # Main training script
│       ├── evaluate.py             # Standalone evaluation script
│       └── generate.py             # Text/DNA generation script

# Usage:
# 1. Define your configuration in configs/training_config.yaml
# 2. Run training with: python -m MetaModel.scripts.train
# 3. Generate text with: python -m MetaModel.scripts.generate --model_path best_model.pt
"""