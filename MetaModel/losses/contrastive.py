"""Contrastive loss functions for multimodal learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.07, batch_size=None):
    """
    InfoNCE contrastive loss for better representation learning.
    
    Args:
        features: Tensor of shape [2*batch_size, feature_dim] containing the features of the positive pairs.
            The first batch_size elements are anchors, and the second batch_size elements are positives.
        temperature: Temperature parameter to scale the similarity scores.
        batch_size: Size of the batch. If None, assumes features has shape [2*B, D] and infers batch_size=B.
    
    Returns:
        InfoNCE loss
    """
    if batch_size is None:
        batch_size = features.shape[0] // 2
        
    # Normalize the features
    features = F.normalize(features, dim=1)
    
    # Create similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create labels
    labels = torch.cat([
        torch.arange(batch_size, 2*batch_size),  # Positives for first half (anchors)
        torch.arange(0, batch_size)              # Positives for second half
    ]).to(features.device)
    
    # Create mask for the positive pairs (diagonal blocks)
    mask = torch.zeros_like(similarity_matrix)
    mask.scatter_(1, labels.unsqueeze(1), 1)
    
    # InfoNCE loss calculation
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
    
    return -mean_log_prob.mean()

def dual_softmax_loss(similarity_matrix, temperature=0.07):
    """
    Dual softmax loss for bidirectional similarity optimization.
    Helps with cross-modal retrieval by optimizing in both directions.
    
    Args:
        similarity_matrix: Similarity matrix of shape [batch_size, batch_size]
        temperature: Temperature parameter
    
    Returns:
        Dual softmax loss
    """
    sim_row_scaled = similarity_matrix / temperature
    sim_col_scaled = similarity_matrix.t() / temperature
    
    # Row-wise softmax (text-to-dna retrieval)
    row_labels = torch.arange(sim_row_scaled.size(0)).to(sim_row_scaled.device)
    row_loss = F.cross_entropy(sim_row_scaled, row_labels)
    
    # Column-wise softmax (dna-to-text retrieval)
    col_labels = torch.arange(sim_col_scaled.size(0)).to(sim_col_scaled.device)
    col_loss = F.cross_entropy(sim_col_scaled, col_labels)
    
    # Average the losses
    return (row_loss + col_loss) / 2.0

def mean_pool(hidden_states, attention_mask):
    """
    Pool hidden states using attention mask for proper weighting.
    
    Args:
        hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
        attention_mask: Tensor of shape [batch_size, seq_len]
        
    Returns:
        Pooled embeddings of shape [batch_size, hidden_dim]
    """
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    summed_mask = mask.sum(dim=1).clamp(min=1e-9)
    return summed / summed_mask

class DNATextContrastiveLoss(nn.Module):
    """
    Combined contrastive losses for DNA-Text fusion.
    
    This module combines InfoNCE and Dual Softmax losses with temperature annealing.
    """
    def __init__(self, initial_temp=0.5, min_temp=0.07, total_epochs=8, weight=1.0):
        super().__init__()
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.total_epochs = total_epochs
        self.weight = weight
        
    def forward(self, pred_embeddings, target_embeddings, epoch=0):
        """
        Calculate combined contrastive loss with temperature annealing.
        
        Args:
            pred_embeddings: Embeddings of predicted texts
            target_embeddings: Embeddings of target texts
            epoch: Current epoch for temperature annealing
            
        Returns:
            Combined contrastive loss and metrics dict
        """
        # Apply temperature annealing
        temp = max(self.min_temp, self.initial_temp - (epoch / self.total_epochs) * 
                   (self.initial_temp - self.min_temp))
        
        # InfoNCE loss
        features = torch.cat([pred_embeddings, target_embeddings], dim=0)
        info_nce = info_nce_loss(features, temperature=temp)
        
        # Dual softmax loss
        similarity_matrix = torch.matmul(
            F.normalize(pred_embeddings, dim=1), 
            F.normalize(target_embeddings, dim=1).T
        )
        dual_softmax = dual_softmax_loss(similarity_matrix, temperature=temp)
        
        # Combine losses
        combined_loss = (info_nce + dual_softmax) * self.weight
        
        # Return loss and metrics for logging
        metrics = {
            "info_nce_loss": info_nce.item(),
            "dual_softmax_loss": dual_softmax.item(),
            "contrastive_loss": combined_loss.item(),
            "contrastive_temperature": temp
        }
        
        return combined_loss, metrics