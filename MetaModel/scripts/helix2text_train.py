#!/usr/bin/env python3
"""
Helix2Text Training Script

A simplified script that loads configuration and starts model training
using PyTorch's DistributedDataParallel (DDP) for multi-GPU training.
"""

import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
import random
import numpy as np
import subprocess


# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MetaModel.configs.default_config import default_config
from MetaModel.training.train import train_model
from MetaModel.models.dna_text_fusion import DNATextFusion, load_models
from MetaModel.data.dataset import get_data_loaders

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Modify these parameters according to your needs

# Path Configuration
DATA_PATH = "\MetaModel\data\protein_to_dna_sequence_descriptions.csv"  # Path to your training data
OUTPUT_MODEL_PATH = "best_model.pt"  # Where to save the trained model

# Training Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
GRAD_ACCUM_STEPS = 3  # Reduced from 100 for more frequent updates
EARLY_STOP_PATIENCE = 3

# Data Splitting Options
SPLIT_DATA = True  # Whether to split data into train/eval
TRAIN_DATA_FRACTION = 1.0  # Fraction of data to use (1.0 = use all data)
TRAIN_SPLIT_FRACTION = 0.9  # Fraction for training vs. evaluation

# Experiment Tracking
USE_WANDB = True  # Whether to use Weights & Biases for tracking
PROJECT_NAME = "Helix2Text"  # Same project name as your other files
EXPERIMENT_NAME = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Distributed Training Configuration
USE_DISTRIBUTED = True  # Re-enabled with fixed progress visualization
NUM_GPUS = 4  # Number of GPUs to use for distributed training
DISTRIBUTED_BACKEND = "nccl"  # Backend for distributed training (nccl is best for CUDA)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_available_gpus():
    """Count the number of available NVIDIA GPUs."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               check=True)
        gpu_list = result.stdout.strip().split('\n')
        return len(gpu_list)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not query GPUs with nvidia-smi. Assuming 1 GPU.")
        return 1 if torch.cuda.is_available() else 0

def setup_distributed(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(random.randint(10000, 20000)))
    
    # Initialize the process group
    dist.init_process_group(backend=DISTRIBUTED_BACKEND, rank=rank, world_size=world_size)

def cleanup_distributed():
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# TRAINING FUNCTION FOR EACH PROCESS
# =============================================================================

def train_process(rank, world_size):
    """Training process for a single GPU in distributed training."""
    try:
        # Set up device for this process
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize the distributed process group
        setup_distributed(rank, world_size)
        
        # Only the main process prints information
        is_main_process = (rank == 0)
        
        if is_main_process:
            print(f"Starting Helix2Text training with data: {DATA_PATH}")
            print(f"Distributed training: Enabled, {world_size} GPUs")
            print(f"Process rank: {rank}, World size: {world_size}, Device: {device}")
        
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
        
        # Update DeepSpeed configuration to use ZeRO-3 with offloading
        config.deepspeed.use_deepspeed = True
        config.deepspeed.zero_stage = 3  # Optimizer + parameter sharding and offloading
        config.deepspeed.fp16_enabled = True  # Enable mixed precision

        # Add memory-efficient configuration for DeepSpeed
        config.deepspeed.offload_optimizer = True
        config.deepspeed.offload_param = True
        config.deepspeed.zero3_init_flag = True  # Enable ZeRO-3 initialization optimization
        config.deepspeed.zero3_save_16bit_model = True  # Save in FP16 format
        config.deepspeed.sub_group_size = 1e9  # Large value to reduce communication
        config.deepspeed.contiguous_gradients = True  # More efficient gradient handling
        config.deepspeed.overlap_comm = True  # Overlap communication with computation

        if is_main_process:
            print("DeepSpeed ZeRO-3 enabled with parameter/optimizer offloading for maximum memory efficiency")
        
        # Set random seed for reproducibility (adding rank for process-specific randomness)
        set_seed(42 + rank)
        
        # Configure distributed attributes
        config.distributed = {
            "enabled": True,
            "rank": rank,
            "world_size": world_size,
            "is_main_process": is_main_process
        }
        
        # Configure wandb for DDP - disable on non-main processes
        # Set environment variables before any possible wandb import/init
        if not is_main_process:
            # Completely disable wandb on non-main processes
            os.environ["WANDB_MODE"] = "disabled"
            os.environ["WANDB_SILENT"] = "true"
        elif not USE_WANDB:
            # If wandb is disabled by user config
            os.environ["WANDB_MODE"] = "disabled"
            if is_main_process:
                print("Weights & Biases tracking disabled")
        else:
            # Set a unique group ID for this DDP run
            os.environ["WANDB_RUN_GROUP"] = f"ddp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if is_main_process:
                print(f"Weights & Biases tracking enabled for project: {PROJECT_NAME}")
        
        # Load model and tokenizers
        dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
            dna_model_path=config.model.dna_model_path,
            text_model_path=config.model.text_model_path,
            device=device
        )
        
        # Enable gradient checkpointing in the models to save memory
        dna_model.gradient_checkpointing_enable()
        text_model.gradient_checkpointing_enable()
        
        if is_main_process:
            print("Gradient checkpointing enabled to reduce memory usage")
        
        # Create the fusion model
        model = DNATextFusion(
            dna_model=dna_model,
            text_model=text_model,
            cross_attn_interval=config.model.cross_attn_interval
        ).to(device)
        

        model = model.half()

        # Wrap model with DDP
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        # Add DDP integration info to config
        config.ddp_runtime = {
            "model": model,
            "rank": rank,
            "is_distributed": True,
            "world_size": world_size,
            "dna_tokenizer": dna_tokenizer,
            "text_tokenizer": text_tokenizer,
            "device": device
        }
        
        if is_main_process:
            print(f"Starting training with DDP, {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
        
        # Train the model with DDP integration
        model, metrics = train_model(config)
        
        # Only the main process should print final results
        if is_main_process:
            print("\n============= Training Complete =============")
            print(f"Final BLEU Score: {metrics.get('bleu_score', 0):.4f}")
            print(f"Model saved to: {OUTPUT_MODEL_PATH}")
        
    except Exception as e:
        import traceback
        error_msg = f"Process {rank} encountered an error: {str(e)}"
        error_traceback = traceback.format_exc()
        
        print(f"\n{'='*50}")
        print(error_msg)
        print(f"Traceback:\n{error_traceback}")
        
        # Log to wandb if this is the main process and wandb is enabled
        if rank == 0 and USE_WANDB:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"error": str(e), "traceback": error_traceback})
                    wandb.finish(exit_code=1)
            except:
                pass
        
        # Raise the exception to ensure the process exits with an error
        # This will be caught by the spawner and signal other processes to terminate
        raise
    finally:
        # Always clean up the distributed environment, even if an error occurred
        try:
            cleanup_distributed()
        except Exception as cleanup_error:
            print(f"Process {rank} had an error during cleanup: {str(cleanup_error)}")
            # Don't raise this error as it would mask the original error

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Configure and start model training with DDP."""
    global USE_DISTRIBUTED
    
    try:
        # Set random seed for reproducibility
        set_seed(42)
        
        # Check if we should use distributed training
        if USE_DISTRIBUTED and torch.cuda.is_available():
            # Check available GPUs
            available_gpus = count_available_gpus()
            world_size = min(NUM_GPUS, available_gpus)
            
            if world_size > 1:
                print(f"Launching distributed training across {world_size} GPUs")
                # Launch processes for distributed training
                mp.spawn(
                    train_process,
                    args=(world_size,),
                    nprocs=world_size,
                    join=True
                )
                return 0
            else:
                print("Only 1 GPU available, falling back to single-GPU training")
                USE_DISTRIBUTED = False
        
        # If we're using a single GPU or CPU, run standard training
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
        
        # Update DeepSpeed configuration to use ZeRO-3 with offloading
        config.deepspeed.use_deepspeed = True
        config.deepspeed.zero_stage = 3  # Optimizer + parameter sharding and offloading
        config.deepspeed.fp16_enabled = True  # Enable mixed precision

        # Add memory-efficient configuration for DeepSpeed
        config.deepspeed.offload_optimizer = True
        config.deepspeed.offload_param = True
        config.deepspeed.zero3_init_flag = True  # Enable ZeRO-3 initialization optimization
        config.deepspeed.zero3_save_16bit_model = True  # Save in FP16 format
        config.deepspeed.sub_group_size = 1e9  # Large value to reduce communication
        config.deepspeed.contiguous_gradients = True  # More efficient gradient handling
        config.deepspeed.overlap_comm = True  # Overlap communication with computation

        print("DeepSpeed ZeRO-3 enabled with parameter/optimizer offloading for maximum memory efficiency")
        
        # Configure distributed attributes (single-process mode)
        config.distributed = {
            "enabled": False,
            "rank": 0,
            "world_size": 1,
            "is_main_process": True
        }
        
        # Disable wandb if not requested
        if not USE_WANDB:
            os.environ["WANDB_MODE"] = "disabled"
            print("Weights & Biases tracking disabled")
        else:
            print(f"Weights & Biases tracking enabled for project: {PROJECT_NAME}")
        
        # Set device for single-GPU/CPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using standard training with device: {device}")
        
        # Load model and tokenizers
        dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
            dna_model_path=config.model.dna_model_path,
            text_model_path=config.model.text_model_path,
            device=device
        )
        
        # Enable gradient checkpointing in the models to save memory
        dna_model.gradient_checkpointing_enable()
        text_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled to reduce memory usage")
        
        # Create the fusion model
        print("Loading models...")
        model = DNATextFusion(
            dna_model=dna_model,
            text_model=text_model,
            cross_attn_interval=config.model.cross_attn_interval
        ).to(device)
        
        # Standard training without DDP
        print(f"Starting standard training, {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
        model, metrics = train_model(config)
        
        print("\n============= Training Complete =============")
        print(f"Final BLEU Score: {metrics.get('bleu_score', 0):.4f}")
        print(f"Model saved to: {OUTPUT_MODEL_PATH}")
        
        return 0
    
    except Exception as e:
        # Log the error
        import traceback
        error_msg = f"\n\n{'='*50}\nERROR: Training failed with the following error:\n{str(e)}\n\n"
        error_traceback = traceback.format_exc()
        
        print(error_msg)
        print(f"Traceback:\n{error_traceback}")
        
        # If using wandb, log the error there too
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"error": str(e), "traceback": error_traceback})
                wandb.finish(exit_code=1)
        except:
            pass
        
        # Return error code 1 to indicate failure
        return 1

if __name__ == "__main__":
    sys.exit(main())