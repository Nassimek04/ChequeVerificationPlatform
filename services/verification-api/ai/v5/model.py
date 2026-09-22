"""V2 model.

Backbone intentionally UNCHANGED from V1 (ResNet18, ImageNet pretrained,
128-D L2-normalized embeddings) so any improvement is attributable to the
metric-learning objective, not capacity. Reuses the exact V1 model class
(shared code, no modification).
"""

from __future__ import annotations

from ai.model import SiameseResNet18  # noqa: F401  (re-export, shared with V1)
from ai.model import build_model, cosine_similarity, euclidean_distance  # noqa: F401