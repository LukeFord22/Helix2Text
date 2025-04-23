"""Combined loss functions for DNA-text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import DNATextContrastiveLoss

class DNATextCriterion(nn.Module):
    """
    Combined loss function for DNA-Text fusion model training.
    
    This criterion combines cross-entropy for sequence generation and
    contrastive loss for alignment between modalities.
    """
    def __init__(self, 
                 semantic_weight=0.5, 
                 initial_temp=0.5, 
                 min_temp=0.07, 
                 total_epochs=8):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = DNATextContrastiveLoss(
            initial_temp=initial_temp,
            min_temp=min_temp,
            total_epochs=total_epochs,
            weight=semantic_weight
        )
        self.semantic_weight = semantic_weight
        
    def forward(self, outputs, targets, epoch=0):
        """
        Calculate combined loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target tensors dictionary
            epoch: Current epoch for temperature annealing
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Cross-entropy loss for sequence generation
        logits = outputs["logits"]
        target_ids = targets["target_ids"]
        
        ce_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Contrastive loss between joint embeddings
        contrastive_loss, contrastive_metrics = self.contrastive_loss(
            outputs["joint_embedding"],
            outputs["fused_representation"],
            epoch=epoch
        )
        
        # Calculate dynamic weight based on training progress
        alpha = self.semantic_weight * max(0.1, min(1.0, epoch / (0.3 * self.contrastive_loss.total_epochs)))
        
        # Combine losses
        total_loss = alpha * contrastive_loss + (1 - alpha) * ce_loss
        
        # Return loss and metrics
        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "alpha": alpha,
            **contrastive_metrics
        }
        
        return total_loss, loss_dict