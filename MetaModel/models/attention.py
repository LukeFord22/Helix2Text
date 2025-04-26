import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    FlashAttention = LlamaFlashAttention2
except ImportError:
    # Fallback if FlashAttention2 is not available
    FlashAttention = nn.MultiheadAttention


class NormalizedGatedCrossAttention(nn.Module):
    """Highly optimized cross-attention block with layernorm, projection, gating."""
    def __init__(self, query_dim, context_dim=None, n_heads=8, dropout=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim if context_dim is not None else query_dim
        self.n_heads = n_heads

        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(self.context_dim)

        if self.query_dim != self.context_dim:
            self.context_proj = nn.Linear(self.context_dim, query_dim)
        else:
            self.context_proj = nn.Identity()

        self.cross_attn = FlashAttention(
            hidden_size=query_dim,
            num_heads=n_heads,
            attention_dropout=dropout
        )

        self.gate = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.Sigmoid()
        )

    def forward(self, query, context, mask=None):
        q = self.norm_query(query)
        k = self.norm_context(context)
        k = self.context_proj(k)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask[:, None, None, :]
            elif mask.dim() == 3 and mask.size(1) == 1:
                attn_mask = mask.unsqueeze(1)
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.float()

        out = self.cross_attn(
            hidden_states=q,
            encoder_hidden_states=k,
            attention_mask=attn_mask,
            encoder_attention_mask=attn_mask,
            output_attentions=False,
            use_cache=False
        )[0]

        gate_val = self.gate(torch.cat([q, out], dim=-1))
        return query + gate_val * out

class CrossModalityTransformerBlock(nn.Module):
    """Transformer block with Flash Self-Attention + Cross-Attention."""
    def __init__(self, dim, cross_dim=None, n_heads=8, dropout=0.1, ff_expansion=4):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = FlashAttention(
            hidden_size=dim,
            num_heads=n_heads,
            attention_dropout=dropout
        )

        self.cross_attn = NormalizedGatedCrossAttention(
            query_dim=dim,
            context_dim=cross_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)

        ff_hidden = int(dim * ff_expansion)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, self_mask=None, context_mask=None):
        x_norm = self.norm1(x)

        x = x + self.self_attn(
            hidden_states=x_norm,
            attention_mask=self_mask,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        )

        x = self.cross_attn(x, context, mask=context_mask)

        x = x + self.mlp(self.norm2(x))
        return x
