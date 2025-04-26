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
from datetime import datetime, timedelta
import deepspeed
import random
import numpy as np
import subprocess
import wandb

# Add the parent directory to the path so we can import the modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from MetaModel.configs.default_config import default_config
from MetaModel.training.train import train_model
from MetaModel.models.dna_text_fusion import DNATextFusion, load_models
from MetaModel.data.dataset import get_data_loaders

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Modify these parameters according to your needs

# Path Configuration
DATA_PATH = os.path.join(parent_dir, "MetaModel/data/protein_to_dna_sequence_descriptions.csv")  # Path to your training data
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
PROJECT_NAME = "Helix2Text"  # Same project name as your other files
EXPERIMENT_NAME = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Distributed Training Configuration
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
    # Use a fixed port instead of random
    os.environ['MASTER_PORT'] = '29500'
    
    # Add timeout to avoid hanging indefinitely
    dist.init_process_group(
        backend=DISTRIBUTED_BACKEND, 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=30)  
    )

def cleanup_distributed():
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def train_process(rank, world_size):
    """Training process for a single GPU in distributed training."""
    try:
        # Create Triton autotune directory if it doesn't exist
        os.makedirs("/root/.triton/autotune", exist_ok=True)
        
        print(f"Process {rank}: Setting up device")
        # Set up device for this process
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize the distributed process group
        if world_size > 1:
            print(f"Process {rank}: Initializing distributed process group")
            setup_distributed(rank, world_size)
            print(f"Process {rank}: Process group initialized successfully")
        
        # Only the main process prints information
        is_main_process = (rank == 0)
        
        if is_main_process:
            print(f"Starting Helix2Text training")
            if world_size > 1:
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
            "enabled": world_size > 1,
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
        print(f"Process {rank}: Loading models")
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
        if is_main_process:
            torch.cuda.empty_cache()  # Free up any cached memory
            print(f"GPU memory before fusion: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Create the fusion model
        model = DNATextFusion(
            dna_model=dna_model,
            text_model=text_model,
            cross_attn_interval=config.model.cross_attn_interval
        ).to(device)
        
        if is_main_process:
            print(f"GPU memory after fusion: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Model loaded on device: {device}")

        print(f"Process {rank}: Initializing with DeepSpeed")
        # Create a deepspeed config dict from your config object
        ds_config = {
            "train_batch_size": config.training.batch_size * world_size,
            "fp16": {"enabled": config.deepspeed.fp16_enabled},
            "zero_optimization": {
                "stage": config.deepspeed.zero_stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "contiguous_gradients": config.deepspeed.contiguous_gradients,
                "overlap_comm": config.deepspeed.overlap_comm
            }
        }

        # Initialize with DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        print(f"Process {rank}: DeepSpeed initialization successful")

        # Update config for training
        config.ddp_runtime = {
            "model": model_engine,  # Use model_engine instead of model
            "rank": rank,
            "is_distributed": world_size > 1,
            "world_size": world_size,
            "dna_tokenizer": dna_tokenizer,
            "text_tokenizer": text_tokenizer,
            "device": device
        }
        
        if is_main_process:
            print(f"Starting training with {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
        
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
        
        # Log to wandb if this is the main process
        try:
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
        if world_size > 1:
            try:
                cleanup_distributed()
            except Exception as cleanup_error:
                print(f"Process {rank} had an error during cleanup: {str(cleanup_error)}")
                # Don't raise this error as it would mask the original error



def main():
    """Configure and start model training with DDP."""
    
    try:
        # Create Triton autotune directory if it doesn't exist
        os.makedirs("/root/.triton/autotune", exist_ok=True)
        
        # Set random seed for reproducibility
        set_seed(42)
        
        available_gpus = count_available_gpus()
        world_size = min(NUM_GPUS, available_gpus)
        
        print(f"Launching distributed training across {world_size} GPUs")

        # Use multiprocessing to launch multiple processes
        mp.spawn(
            train_process,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )

            
    except Exception as e:
        print(f"ERROR: Training failed with the following error:\n\n{str(e)}")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())