import torch
import torch.nn as nn

from .transformer_block import LightweightTransformerBlock


class FineGrainedAttentionModule(nn.Module):
    """refine spatial image tokens with intra-modal self-attention.

    Args:
        feature_dim: size of each image token.
        num_heads: number of self-attention heads.
        transformer_layers: number of lightweight transformer blocks.
    """

    def __init__(self, feature_dim, num_heads=8, transformer_layers=2):
        super().__init__()
        self.initial_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.transformer_layers = nn.ModuleList([
            LightweightTransformerBlock(
                embed_dim=feature_dim,
                num_heads=num_heads,
                ff_dim=feature_dim * 4,
                dropout=0.1,
            )
            for _ in range(transformer_layers)
        ])
        self.final_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """refine an image-token sequence without changing its shape.

        Args:
            x: image tokens shaped ``[batch, tokens, feature_dim]``.

        Returns:
            Refined image tokens with the same shape as ``x``.
        """
        residual = x
        attention_output, _ = self.initial_attention(x, x, x)
        x = self.norm1(residual + attention_output)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.norm2(x + self.final_ffn(x))


EnhancedFineGrainedAttentionModule = FineGrainedAttentionModule
