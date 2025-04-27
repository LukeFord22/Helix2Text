#!/usr/bin/env python3
"""Helix2Text Training Script — Optimized."""

import os
import sys
import torch
import wandb
import random
import traceback
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from datetime import datetime, timedelta


# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from MetaModel.data.dataset import get_data_loaders
from MetaModel.configs.default_config import default_config
from MetaModel.training.train import train_model
from MetaModel.models.load_models import setup_fusion_and_deepspeed

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_gpus():
    return torch.cuda.device_count()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )

def cleanup():
    dist.destroy_process_group()


# =============================================================================
# TRAINING PROCESS
# =============================================================================

def train_process(rank, world_size):
    """Main training loop for a single rank."""

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["NCCL_DEBUG"] = "WARN"
    
    try:

        setup(rank, world_size)

        device = torch.device(f"cuda:{rank}")

        config = default_config

        config.experiment_name = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


        print(f"[{rank}] Loading models...")
        model_engine, dna_tokenizer, text_tokenizer = setup_fusion_and_deepspeed(rank, world_size, config)

        train_loader, test_loader = get_data_loaders(
        dna_tokenizer, text_tokenizer, config,
        rank=rank, world_size=world_size
        )

        # Update config runtime references
        config.deepspeed_runtime = {
            "model_engine": model_engine,
            "device": device,
            "rank": rank,
            "world_size": world_size,
            "dna_tokenizer": dna_tokenizer,
            "text_tokenizer": text_tokenizer,
            "train_loader": train_loader,  # Add this!
            "test_loader": test_loader     # Add this!
        }
        
        print(f"[{rank}] Starting training: {config.training.num_epochs} epochs")


        train_model(config)



    except Exception as e:
        # Clean error message
        print(f"\n[Rank {rank}] ❌ Exception during training: {e}\n{traceback.format_exc()}")

        # Attempt WANDB logging
        try:
            if wandb.run is not None:
                wandb.log({
                    "error_message": str(e),
                    "error_traceback": traceback.format_exc()
                })
                wandb.finish(exit_code=1)
        except Exception as wandb_error:
            print(f"[Rank {rank}] ⚠️ Failed to log to wandb: {wandb_error}")

        # Flush any pending stdout/stderr (important for multi-GPU clusters)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass

        # Always re-raise the original exception
        raise

    finally:
        # Final distributed cleanup
        if world_size > 1:
            try:
                cleanup()
                print(f"[Rank {rank}] ✅ Cleanup successful.")
            except Exception as cleanup_error:
                print(f"[Rank {rank}] ⚠️ Cleanup warning: {cleanup_error}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function to launch training."""
    try:
        os.makedirs("/root/.triton/autotune", exist_ok=True)
        set_seed(42)

        available_gpus = count_gpus()
        world_size = min(default_config.deepspeed.num_gpus, available_gpus)

        print(f"Launching training with {world_size} GPUs")

        mp.spawn(
            train_process,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )

    except Exception as e:
        print(f"Training launch failed:\n{str(e)}\n{traceback.format_exc()}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
