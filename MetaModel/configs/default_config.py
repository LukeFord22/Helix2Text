"""Default configuration parameters for DNA-Text fusion models."""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 2
    num_epochs: int = 8
    learning_rate: float = 3e-5
    max_seq_length: int = 512
    grad_accum_steps: int = 5
    train_data_fraction: float = 1  
    train_split_fraction: float = 0.9  # Default: 90% for training, 10% for evaluation
    semantic_loss_weight: float = 0.5
    frozenGPT_layers: int = 0
    frozenDNA_layers: int = 0
    early_stop_patience: int = 3

    # Loss weights
    dna_loss_weight: 0.5     
    text_loss_weight: 0.5      
    contrastive_loss_weight: 1 
    
@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""
    dna_model_path: str = "metagene-ai/METAGENE-1-BnB-4Bit"
    text_model_path: str = "microsoft/biogpt-large-pubmedqa"
    cross_attn_interval: int = 4
    hidden_dim: int = 768
    
@dataclass
class DataConfig:
    """Data processing configuration parameters."""
    local_path: str = "protein_to_dna_sequence_descriptions.csv"
    eval_path: str = ""  # Path to evaluation dataset (if empty, split from local_path)
    split_data: bool = True  # Whether to split data from the original file
    
@dataclass
class Config:
    """Combined configuration for the entire project."""
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    experiment_name: str = "biogpt-dna-fusion"
    project_name: str = "dna-text-fusion"
    
# Default configuration
default_config = Config()