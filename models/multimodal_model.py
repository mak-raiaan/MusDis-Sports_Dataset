import torch.nn as nn

from .fgam import FineGrainedAttentionModule
from .image_encoder import CustomImageEncoder
from .mgcm import MultimodalGeometricCorrespondenceModule
from .transformer_block import LightweightTransformerBlock


class MultimodalModel(nn.Module):
    """classify posture risk from image and skeletal features.

    Args:
        num_classes: number of posture-risk output classes.
        coord_dim: flattened coordinate width, expected to be 66.
        transformer_layers: number of FGAM and MGCM transformer blocks.
    """

    def __init__(self, num_classes=8, coord_dim=66, transformer_layers=2):
        super().__init__()
        self.img_feature_dim = 2048
        self.backbone = CustomImageEncoder(
            feature_dim=self.img_feature_dim,
            token_dim=128,
            grid_size=16,
        )
        self.fine_grained_module = FineGrainedAttentionModule(
            feature_dim=128,
            num_heads=8,
            transformer_layers=transformer_layers,
        )
        self.correspondence_module = MultimodalGeometricCorrespondenceModule(
            img_token_dim=128,
            coord_token_dim=2,
            transformer_layers=transformer_layers,
        )
        self.feature_refinement = LightweightTransformerBlock(
            embed_dim=512,
            num_heads=8,
            ff_dim=1024,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, coordinates):
        """return class logits for a batch of multimodal samples.

        Args:
            images: RGB image tensor shaped ``[batch, 3, height, width]``.
            coordinates: flattened 66-value or shaped 33-joint coordinates.

        Returns:
            Class logits shaped ``[batch, num_classes]``.
        """
        visual_features = self.backbone(images)
        visual_features = self.fine_grained_module(visual_features)
        fused_features = self.correspondence_module(
            visual_features, coordinates
        )
        refined_features = self.feature_refinement(
            fused_features
        )
        return self.classifier(refined_features.mean(dim=1))


EnhancedMultimodalModel = MultimodalModel
