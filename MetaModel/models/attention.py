"""Cross-attention mechanisms for DNA-Text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedGatedCrossAttentionBlock(nn.Module):
    """
    Cross-attention block with gating and normalization.
    
    This block allows one modality to attend to another with a learnable
    gating parameter that controls the influence of the cross-attention.
    """
    def __init__(self, dim, cross_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            kdim=cross_dim, 
            vdim=cross_dim,
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # Learnable parameter to control the influence of cross-attention
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, context, context_mask=None):
        """
        Apply cross-attention between x and context.
        
        Args:
            x: Query tensor of shape (batch_size, seq_len, dim)
            context: Key/value tensor of shape (batch_size, context_len, cross_dim)
            context_mask: Optional mask for context of shape (batch_size, context_len)
            
        Returns:
            Updated tensor after cross-attention
        """
        # Convert mask from [0, 1] to boolean for nn.MultiheadAttention
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        
        # Apply layer normalization before attention
        x_norm = self.norm1(x)
        
        # Multihead cross-attention
        cross_output, _ = self.cross_attn(
            query=x_norm, 
            key=context, 
            value=context, 
            key_padding_mask=key_padding_mask
        )
        
        # Apply gated residual connection
        return self.norm2(x + self.alpha * self.dropout(cross_output))
    
class CrossModalityTransformerBlock(nn.Module):
    """
    Complete transformer block with cross-attention for fusing two modalities.
    
    This block combines self-attention, cross-attention, and feed-forward 
    layers to enable effective communication between modalities.
    """
    def __init__(self, dim, cross_dim, n_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or 4 * dim
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention
        self.cross_attn = NormalizedGatedCrossAttentionBlock(
            dim=dim,
            cross_dim=cross_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x, context, self_mask=None, context_mask=None):
        """
        Apply self-attention, cross-attention, and feed-forward to input.
        
        Args:
            x: Query tensor of shape (batch_size, seq_len, dim)
            context: Context tensor of shape (batch_size, context_len, cross_dim)
            self_mask: Optional mask for self-attention
            context_mask: Optional mask for context
            
        Returns:
            Updated tensor after all operations
        """
        # First self-attention with residual
        key_padding_mask = None if self_mask is None else ~self_mask.bool()
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask
        )
        x = x + self_attn_out
        
        # Then cross-attention with context
        x = self.cross_attn(x, context, context_mask)
        
        # Finally feed-forward with residual
        x = x + self.ff(self.norm3(x))
        
        return x