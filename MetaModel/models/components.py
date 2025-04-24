"""Reusable components for DNA-text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableFusionBlock(nn.Module):
    """Block that fuses two representations with learnable weights."""
    
    def __init__(self, hidden_dim, input_dim_x=None, input_dim_y=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Store input dimensions or use hidden_dim as default
        self.input_dim_x = input_dim_x if input_dim_x is not None else hidden_dim
        self.input_dim_y = input_dim_y if input_dim_y is not None else hidden_dim
        
        # Calculate the total input dimension for the gate
        self.combined_dim = self.input_dim_x + self.input_dim_y
        
        # Create projection layers if dimensions don't match
        self.project_x = None
        self.project_y = None
        
        if self.input_dim_x != hidden_dim:
            self.project_x = nn.Linear(self.input_dim_x, hidden_dim)
            
        if self.input_dim_y != hidden_dim:
            self.project_y = nn.Linear(self.input_dim_y, hidden_dim)
        
        # Gate mechanism controls how much of each representation to use
        self.gate = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion layer combines the two representations
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x, y):
        """
        Fuse two representations with adaptive weights.
        
        Args:
            x: First representation tensor (batch_size, seq_len, input_dim_x)
            y: Second representation tensor (batch_size, seq_len, input_dim_y)
            
        Returns:
            Fused representation (batch_size, seq_len, hidden_dim)
        """
        # Project inputs if needed
        if self.project_x is not None:
            x_proj = self.project_x(x)
        else:
            x_proj = x
            
        if self.project_y is not None:
            y_proj = self.project_y(y)
        else:
            y_proj = y
        
        # Calculate gate values based on concatenated input
        combined = torch.cat([x, y], dim=-1)  # Use original inputs for gate calculation
        gate = self.gate(combined)
        
        # Apply gating mechanism
        gated = (gate * x_proj) + ((1 - gate) * y_proj)
        
        # Final fusion
        combined_proj = torch.cat([x_proj, y_proj], dim=-1)
        fused = self.fusion(combined_proj)
        
        # Residual connection with the gated output
        output = gated + fused
        
        return output

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