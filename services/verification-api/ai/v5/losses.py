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


class SupConLoss(nn.Module):
    """Supervised Contrastive on L2 embeddings, genuine only, temperature 0.07."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [N, D] L2-normed, labels: [N] int writer id
        # Only genuine samples should be passed; forged are not used here (caller filters)
        if embeddings.size(0) < 2:
            return embeddings.sum()*0.0
        sim = embeddings @ embeddings.T / self.temperature  # [N,N]
        # numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim)
        # mask positives: same label, not self
        labels = labels.view(-1,1)
        mask_pos = (labels == labels.T).float()
        mask_pos.fill_diagonal_(0)
        # if no positive for an anchor, ignore
        pos_count = mask_pos.sum(1)
        valid = pos_count > 0
        if valid.sum()==0:
            return embeddings.sum()*0.0
        # log-softmax
        # denominator: sum over all except self
        mask_self = 1 - torch.eye(embeddings.size(0), device=embeddings.device)
        denom = (exp_sim * mask_self).sum(1, keepdim=True)
        log_prob = sim - torch.log(denom + 1e-12)
        # mean over positives per anchor
        loss = -(mask_pos * log_prob).sum(1) / (pos_count + 1e-12)
        return loss[valid].mean()


class WriterCELoss(nn.Module):
    """Auxiliary writer classification CE on pooled 512-D feat, genuine only, weight 0.2."""
    def __init__(self, in_dim: int = 512, num_classes: int = 50):
        super().__init__()
        self.classifier = torch.nn.Linear(in_dim, num_classes)
    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # feats: [N, 512] pooled before projection, labels: [N]
        if feats.size(0)==0:
            return feats.sum()*0.0
        logits = self.classifier(feats)
        return torch.nn.functional.cross_entropy(logits, labels)