"""V3 loss — Triplet loss reused from V2 (margin studied on validation)."""

from __future__ import annotations

from ai.v2.losses import TripletLoss, triplet_violation  # noqa: F401

__all__ = ["TripletLoss", "triplet_violation"]