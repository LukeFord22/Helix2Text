"""Reusable components for DNA-text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableFusionBlock(nn.Module):
    """Efficient fusion block for two modalities with adaptive gating."""
    
    def __init__(self, hidden_dim, input_dim_x=None, input_dim_y=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.input_dim_x = input_dim_x or hidden_dim
        self.input_dim_y = input_dim_y or hidden_dim
        
        # Projections to hidden_dim if needed
        self.project_x = nn.Linear(self.input_dim_x, hidden_dim, bias=False) if self.input_dim_x != hidden_dim else nn.Identity()
        self.project_y = nn.Linear(self.input_dim_y, hidden_dim, bias=False) if self.input_dim_y != hidden_dim else nn.Identity()
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-5)
        )
    
    def forward(self, x, y):
        x_proj = self.project_x(x)
        y_proj = self.project_y(y)
        
        combined_proj = torch.cat([x_proj, y_proj], dim=-1)
        
        gate_val = self.gate(combined_proj)
        
        fused = self.fusion(combined_proj)
        
        output = gate_val * x_proj + (1 - gate_val) * y_proj + fused
        
        return output

class ModalityAdapter(nn.Module):
    """Efficient modality adaptation block (bottleneck MLP)."""
    
    def __init__(self, dim, expansion_factor=1.5, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.LayerNorm(dim, eps=1e-5)
        )

    def forward(self, x):
        return x + self.adapter(x)

class ContrastiveTokenHead(nn.Module):
    """Token-level contrastive projection head with normalized pooling."""
    
    def __init__(self, hidden_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim, eps=1e-5)
        )

    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden).float()
        hidden = hidden * mask
        pooled = hidden.sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-6))
        pooled = F.normalize(pooled, dim=-1)
        return self.projector(pooled)