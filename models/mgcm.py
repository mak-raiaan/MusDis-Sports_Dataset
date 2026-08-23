import torch
import torch.nn as nn

from .transformer_block import LightweightTransformerBlock


class MultimodalGeometricCorrespondenceModule(nn.Module):
    """align 256 image tokens with 33 coordinate tokens.

    Args:
        img_token_dim: feature size of each FGAM image token.
        coord_token_dim: feature size of each coordinate token.
        hidden_dim: shared attention feature size.
        output_dim: size of the fused representation.
        transformer_layers: number of post-fusion transformer blocks.
    """

    def __init__(
        self,
        img_token_dim=128,
        coord_token_dim=128,
        hidden_dim=256,
        output_dim=512,
        transformer_layers=2,
    ):
        super().__init__()
        self.img_transform = nn.Sequential(
            nn.Linear(img_token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.coord_transform = nn.Sequential(
            nn.Linear(coord_token_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 8, batch_first=True
        )
        self.transformer_layers = nn.ModuleList([
            LightweightTransformerBlock(
                embed_dim=hidden_dim * 2,
                num_heads=4,
                ff_dim=hidden_dim * 4,
                dropout=0.1,
            )
            for _ in range(transformer_layers)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, img_features, coordinates):
        """perform visual-to-skeletal cross-attention.

        Args:
            img_features: image tokens shaped ``[batch, 256, img_token_dim]``.
            coordinates: joint tokens shaped ``[batch, 33, 2]`` or flattened
                coordinates shaped ``[batch, 66]``.

        Returns:
            Fused tokens shaped ``[batch, 256, output_dim]``.
        """
        if coordinates.ndim == 2:
            coordinates = coordinates.reshape(coordinates.shape[0], 33, 2)
        img_proj = self.img_transform(img_features)
        coord_proj = self.coord_transform(coordinates)
        attention_output, _ = self.cross_attention(
            img_proj,
            coord_proj,
            coord_proj,
        )
        fused = torch.cat([img_proj, attention_output], dim=-1)
        transformer_input = fused
        for layer in self.transformer_layers:
            transformer_input = layer(transformer_input)
        return self.fusion(transformer_input)


EnhancedMultimodalGeometricCorrespondenceModule = (
    MultimodalGeometricCorrespondenceModule
)
