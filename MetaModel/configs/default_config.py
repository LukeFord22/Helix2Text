"""Default configuration parameters for DNA-Text fusion models."""

from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 1
    num_epochs: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 512
    grad_accum_steps: int = 3
    train_data_fraction: float = 1.0
    train_split_fraction: float = 0.9  # Default: 90% for training, 10% for evaluation
    semantic_loss_weight: float = 0.5
    frozenGPT_layers: int = 0
    frozenDNA_layers: int = 0
    early_stop_patience: int = 3

    # Loss weights
    dna_loss_weight: float = 0.5
    text_loss_weight: float = 0.5      
    contrastive_loss_weight: float = 1 
    
@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""
    dna_model_path: str = "metagene-ai/METAGENE-1"
    text_model_path: str = "microsoft/biogpt-large"
    cross_attn_interval: int = 4
    hidden_dim: int = 768
    
@dataclass
class DataConfig:
    """Data processing configuration parameters."""
    local_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 "MetaModel/data/protein_to_dna_sequence_descriptions.csv")
    eval_path: str = ""  # Path to evaluation dataset (if empty, split from local_path)
    split_data: bool = True  # Whether to split data from the original file

@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration parameters."""
    num_gpus: int = 4
    zero_stage: int = 3
    fp16_enabled: bool = True
    offload_optimizer: bool = True
    offload_param: bool = True
    zero3_init_flag: bool = True
    zero3_save_16bit_model: bool = True
    sub_group_size: float = 1e9
    contiguous_gradients: bool = True
    overlap_comm: bool = True

@dataclass
class Config:
    """Combined configuration for the entire project."""
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    deepspeed: DeepSpeedConfig = DeepSpeedConfig()
    experiment_name: str = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project_name: str = "Helix2Text"
    output_path: str = "best_model.pt"
    
# Default configuration
default_config = Config()