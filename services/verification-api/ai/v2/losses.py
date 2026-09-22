"""Triplet loss V2.

Distance metric: Euclidean distance d(a, b) = ||e_a - e_b||_2 on the
L2-normalized embeddings (so d in [0, 2]).

Triplet constraint:
    d(anchor, positive) + margin < d(anchor, negative)

Loss (margin m, configurable; mean over mined triplets):
    L = mean( max(0, d(a, p) - d(a, n) + m) )

Label/ownership semantics (used by mining, not by this loss):
    anchor  : genuine signature of writer W
    positive: genuine signature of the SAME writer W
    negative: skilled forgery of W OR genuine of another writer (impostor)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        d_ap: torch.Tensor,
        d_an: torch.Tensor,
    ) -> torch.Tensor:
        """d_ap, d_an: [B] Euclidean distances for mined triplets."""
        losses = torch.clamp(d_ap - d_an + self.margin, min=0.0)
        return losses.mean()


def triplet_violation(d_ap: torch.Tensor, d_an: torch.Tensor, margin: float) -> torch.Tensor:
    """Vector of per-triplet hinge values (for mining statistics)."""
    return torch.clamp(d_ap - d_an + margin, min=0.0)