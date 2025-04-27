import torch
import torch.nn as nn
import torch.nn.functional as F

class MQAFlashAttention(nn.Module):
    """Multi-Query Attention block with Flash-style speedup."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projections for queries
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # SHARED projections for keys/values (only ONE key, ONE value per position)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        q = self.q_proj(query)  # (B, Lq, H*D)
        k = self.k_proj(key)    # (B, Lk, D)
        v = self.v_proj(value)  # (B, Lk, D)

        # Reshape queries for multihead
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lq, D)

        # Expand k/v to match heads
        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, Lk, D)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, Lk, D)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, Lq, Lk)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = attn_probs @ v  # (B, H, Lq, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)

        return self.out_proj(attn_output)


class NormalizedGatedCrossAttention(nn.Module):
    """Highly optimized cross-attention block with LayerNorm, projection, MQA, gating."""
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

        # Always use MQA
        self.cross_attn = MQAFlashAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            dropout=dropout
        )
        
        self.gate = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.Sigmoid()
        )

    def forward(self, query, context, mask=None):
        q = self.norm_query(query)
        k = self.norm_context(context)
        k = self.context_proj(k)

        out = self.cross_attn(
            query=q,
            key=k,
            value=k,
            mask=mask
        )

        gate_val = self.gate(torch.cat([q, out], dim=-1))
        return query + gate_val * out


class CrossModalityTransformerBlock(nn.Module):
    """Transformer block with Flash Self-Attention + Cross-Attention."""
    def __init__(self, dim, cross_dim=None, n_heads=8, dropout=0.1, ff_expansion=4):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
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
            nn.Linear(dim, ff_hidden * 2),
            nn.SiLU(),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, self_mask=None, context_mask=None):
        x_norm = self.norm1(x)

        attn_output = self.self_attn(
            hidden_states=x_norm,
            attention_mask=self_mask,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        )[0]


        x = x + attn_output
        x = self.cross_attn(x, context, mask=context_mask)
        x = x + self.mlp(self.norm2(x))
        return x
