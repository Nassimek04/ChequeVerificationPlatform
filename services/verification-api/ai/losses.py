"""Contrastive loss V1.

Label semantics (match convention, used everywhere):
    y = 1  -> same-writer / genuine pair
    y = 0  -> non-match

Mathematical definition (margin `m`, configurable; d = Euclidean distance):
    L = mean_over_batch( y * d^2 + (1 - y) * max(m - d, 0)^2 )

The verification threshold is NEVER part of this loss; it is selected later
from validation scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """y must be 0.0/1.0 floats (same convention as dataset labels)."""
        d = torch.norm(e1 - e2, p=2, dim=1)
        pos = y * d.pow(2)
        neg = (1.0 - y) * torch.clamp(self.margin - d, min=0.0).pow(2)
        loss = torch.mean(pos + neg)
        return loss