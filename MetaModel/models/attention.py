"""Cross-attention mechanisms for DNA-Text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedGatedCrossAttentionBlock(nn.Module):
    """Cross-attention block with layer normalization and gating."""
    
    def __init__(self, dim, cross_dim=None, n_heads=8):
        super().__init__()
        self.dim = dim
        self.cross_dim = cross_dim if cross_dim is not None else dim
        
        # Add projection if dimensions don't match
        self.needs_projection = self.cross_dim != self.dim
        if self.needs_projection:
            self.cross_projection = nn.Linear(self.cross_dim, self.dim)
        
        # Layer normalization for stability
        self.norm_self = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(self.cross_dim)
        
        # Cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Gating mechanism to control cross-modal influence
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, cross, cross_mask=None):
        """
        Forward pass through the cross-attention block.
        
        Args:
            x: Query tensor (batch_size, seq_len_q, dim)
            cross: Key/value tensor (batch_size, seq_len_kv, cross_dim)
            cross_mask: Attention mask (batch_size, seq_len_kv)
            
        Returns:
            Updated tensor after cross-attention (batch_size, seq_len_q, dim)
        """
        # Normalize inputs
        normed_x = self.norm_self(x)
        normed_cross = self.norm_cross(cross)
        
        # Project cross to matching dimension if needed
        if self.needs_projection:
            normed_cross = self.cross_projection(normed_cross)
        
        # Process attention mask - key_padding_mask expects shape (batch_size, seq_len_kv)
        # and value of True for positions to be masked out (ignored)
        key_padding_mask = None
        if cross_mask is not None:
            # Convert to boolean mask where True indicates masked positions
            key_padding_mask = ~cross_mask.bool()
            
            # Ensure the mask has shape (batch_size, seq_len_kv)
            if key_padding_mask.dim() > 2:
                key_padding_mask = key_padding_mask.squeeze(1)
        
        # Apply cross-attention - key_padding_mask will be directly passed to MultiheadAttention
        cross_output, _ = self.cross_attn(
            query=normed_x,
            key=normed_cross,
            value=normed_cross,
            key_padding_mask=key_padding_mask
        )
        
        # Calculate adaptive gate values
        gate_input = torch.cat([normed_x, cross_output], dim=-1)
        gate_val = self.gate(gate_input)
        
        # Apply gated residual connection
        out = x + gate_val * cross_output
        
        return out
    
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