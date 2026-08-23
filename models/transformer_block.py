import torch.nn as nn


class LightweightTransformerBlock(nn.Module):
    """apply pre-normalized self-attention and feed-forward refinement.

    Args:
        embed_dim: feature size of every token.
        num_heads: number of self-attention heads.
        ff_dim: hidden size of the feed-forward network.
        dropout: dropout probability used in residual branches.
    """

    def __init__(self, embed_dim, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """refine a token sequence while preserving its dimensions.

        Args:
            x: tensor shaped ``[batch, tokens, embed_dim]``.

        Returns:
            Refined tensor with the same shape as ``x``.
        """
        residual = x
        normalized = self.norm1(x)
        attention_output, _ = self.self_attn(
            normalized, normalized, normalized
        )
        x = residual + self.dropout(attention_output)

        residual = x
        normalized = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(normalized))
        return x
