"""Small shared encoder for local Japanese glyph-shape patches."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """A compact normalized convolution block."""

    def __init__(self, input_channels: int, output_channels: int, stride: int):
        super().__init__()
        groups = min(8, output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride,
                      padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class LocalShapeEncoder(nn.Module):
    """Five-layer CNN with explicit native-resolution conditioning.

    The image is a single-channel ink image in [0, 1], where 1 is dark ink.
    ``native_pixels`` is the effective source resolution before normalization.
    The embedding is L2-normalized and is suitable for cosine comparison.
    """

    def __init__(self, embedding_dim: int = 64, family_count: int = 0,
                 face_count: int = 0, weight_count: int = 0):
        super().__init__()
        channels = (32, 64, 96, 128, 160)
        blocks = []
        input_channels = 1
        for index, output_channels in enumerate(channels):
            blocks.append(ConvBlock(input_channels, output_channels,
                                    stride=2 if index < 4 else 1))
            input_channels = output_channels
        self.backbone = nn.Sequential(*blocks)
        self.resolution = nn.Sequential(
            nn.Linear(1, 8), nn.SiLU(inplace=True), nn.Linear(8, 8),
        )
        self.projection = nn.Sequential(
            nn.Linear(channels[-1] + 8, 128), nn.SiLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )
        self.family_head = nn.Linear(embedding_dim, family_count) if family_count else None
        self.face_head = nn.Linear(embedding_dim, face_count) if face_count else None
        self.weight_head = nn.Linear(embedding_dim, weight_count) if weight_count else None

    def forward(self, image: torch.Tensor, native_pixels: torch.Tensor,
                heads: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.backbone(image)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        resolution = torch.log(native_pixels.float().clamp_min(1).reshape(-1, 1) / 64.)
        features = torch.cat((features, self.resolution(resolution)), dim=1)
        raw = self.projection(features)
        embedding = F.normalize(raw, p=2, dim=1)
        if not heads:
            return embedding
        output = {"embedding": embedding, "raw": raw}
        for name in ("family", "face", "weight"):
            head = getattr(self, f"{name}_head")
            if head is not None:
                output[name] = head(embedding)
        return output


def parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)
