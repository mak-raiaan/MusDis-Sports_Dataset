import torch
import torch.nn as nn

from .residual_block import ResidualBlock


class CustomImageEncoder(nn.Module):
    """encode an image as a sequence of 256 visual tokens.

    Args:
        feature_dim: number of channels in the final convolutional map.
        token_dim: feature size assigned to each image token.
        grid_size: spatial size used to create the token grid.
    """

    def __init__(self, feature_dim=2048, token_dim=128, grid_size=16):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.res_block1 = ResidualBlock(64, 128, 2)
        self.res_block2 = ResidualBlock(128, 256, 2)
        self.res_block3 = ResidualBlock(256, 512, 2)
        self.res_block4 = ResidualBlock(512, feature_dim // 2, 2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.token_projection = nn.Conv2d(
            feature_dim,
            token_dim,
            kernel_size=1,
        )

    def forward(self, x):
        """return image tokens with shape ``[batch, 256, 128]``.

        Args:
            x: RGB image tensor shaped ``[batch, 3, height, width]``.

        Returns:
            A sequence of projected image tokens.
        """
        x = self.conv_block1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.final_conv(x)
        x = self.token_pool(x)
        x = self.token_projection(x)
        return torch.flatten(x, 2).transpose(1, 2)
