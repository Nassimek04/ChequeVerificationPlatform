"""Siamese network V1: ResNet18 backbone, shared weights, projection head.

- Both branches share a SINGLE backbone instance (tied weights by construction:
  `encode` is called on each input with the same module).
- The classification head of ResNet18 is removed.
- A projection MLP maps the 512-d penultimate features to `embedding_dim`.
- Embeddings are L2-normalized before distance/similarity computation.

Input convention: [3, H, W] ImageNet-normalized tensors (see preprocessing.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .config import ExperimentConfig


class SiameseResNet18(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg

        weights = None
        if cfg.pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)
        self._pretrained_loaded = weights is not None

        # Remove the ImageNet classification head.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection head (features -> embedding).
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cfg.embedding_dim),
        )

    @property
    def pretrained_loaded(self) -> bool:
        return self._pretrained_loaded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] -> L2-normalized embedding [B,D]."""
        f = self.backbone(x)            # [B,512]
        e = self.projection(f)          # [B,D]
        return nn.functional.normalize(e, p=2, dim=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """Shared-weight forward for a pair. Returns both embeddings."""
        return self.encode(img1), self.encode(img2)


def euclidean_distance(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Euclidean distance on L2-normalized embeddings (in [0, 2])."""
    return torch.norm(e1 - e2, p=2, dim=1)


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity (embeddings are already L2-normalized)."""
    return (e1 * e2).sum(dim=1)


def build_model(cfg: ExperimentConfig) -> SiameseResNet18:
    return SiameseResNet18(cfg)