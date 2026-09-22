"""V4 model — same as V2 (ResNet18, 128-D L2) — no architecture change."""
from ai.model import SiameseResNet18
from ai.model import build_model, cosine_similarity, euclidean_distance

__all__ = ["SiameseResNet18", "build_model", "cosine_similarity", "euclidean_distance"]
