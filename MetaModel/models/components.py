"""Reusable components for DNA-text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableFusionBlock(nn.Module):
    """
    Fusion block with learnable gating for combining modality embeddings.
    
    This block learns to combine embeddings from different modalities (DNA and text)
    using a learnable gating mechanism that adapts to the importance of each modality.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, text_hidden, dna_hidden):
        """
        Combine text and DNA embeddings with learned gating.
        
        Args:
            text_hidden: Text modality embeddings of shape [batch_size, hidden_dim]
            dna_hidden: DNA modality embeddings of shape [batch_size, hidden_dim]
            
        Returns:
            Fused embeddings of shape [batch_size, hidden_dim]
        """
        combined = torch.cat([text_hidden, dna_hidden], dim=-1)
        gate = self.gate(combined)
        fused = gate * text_hidden + (1 - gate) * dna_hidden
        return self.linear(torch.cat([fused, dna_hidden], dim=-1))

class ModalityAdapter(nn.Module):
    """
    Adapter module for modality-specific transformations.
    
    This adapter helps bridge the gap between different modality representations
    by applying a modality-specific transformation before fusion.
    """
    def __init__(self, dim, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        """Apply modality-specific adaptation."""
        return x + self.adapter(x)

class ContrastiveTokenHead(nn.Module):
    """
    Projection head for token-level contrastive learning.
    
    Projects token embeddings to a space suitable for contrastive learning.
    """
    def __init__(self, hidden_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, hidden, mask):
        """
        Project and pool token embeddings.
        
        Args:
            hidden: Token embeddings of shape [batch_size, seq_len, hidden_dim]
            mask: Attention mask of shape [batch_size, seq_len]
            
        Returns:
            Projected embeddings of shape [batch_size, proj_dim]
        """
        mask = mask.unsqueeze(-1).expand_as(hidden).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.projector(pooled)