"""Optimization utilities for DNA-text fusion models."""

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def create_optimizer_and_scheduler(model, lr=3e-5, steps_per_epoch=100, num_epochs=8, grad_accum_steps=1):
    """
    Create optimizer and learning rate scheduler with memory optimizations.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        steps_per_epoch: Number of steps per epoch
        num_epochs: Number of epochs
        grad_accum_steps: Gradient accumulation steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Check if we can use fused implementations for better performance
    fused_available = False
    try:
        # Check if PyTorch supports fused optimizers
        fused_available = hasattr(torch.optim, 'AdamW') and hasattr(torch.optim.AdamW, 'fused')
    except:
        pass
    
    # Create optimizer with memory-efficient options
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=fused_available  # Use fused implementation if available
    )
    
    # Calculate scheduler parameters
    total_steps = (num_epochs * steps_per_epoch) // grad_accum_steps
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    # Create cosine annealing scheduler for main training
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps,
        eta_min=1e-6  # Don't go all the way to zero
    )
    
    # Combine into sequential scheduler
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    return optimizer, scheduler