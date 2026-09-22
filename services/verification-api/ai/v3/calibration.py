"""Enrollment-only calibration.

All calibration statistics are computed EXCLUSIVELY from the customer's
genuine enrollment references. No forgeries and no query labels are ever used.

Given K enrolled references with embeddings R = [r_1..r_K] (each L2-normalized)
and a query embedding q (L2-normalized), per-reference similarities are
    s_i(q) = q . r_i ,  i = 1..K

Aggregated raw score:        s_raw(q) = agg_i( s_i(q) )
Aggregation may be max/mean/median/top-2 mean or prototype:
    prototype = L2_normalize( (r_1+...+r_K)/K )
    s_proto(q) = q . prototype

Enrollment-only statistics (computed from the references themselves):

  pairwise ref similarities   P = { r_i . r_j : i<j }
  mu    = mean(P)                         (enrollment centroid similarity)
  sigma = std(P),  floored at eps
  med   = median(P)
  mad   = median(|P - med|),  floored at eps
  p_i   = r_i . prototype,  i = 1..K      (references vs their own prototype)
  mu_p  = mean(p_i), sigma_p = std(p_i) floored at eps

Calibration formulas (applied to an aggregated score s):

  raw            : s
  z              : (s - mu) / sigma
  mad            : (s - med) / (1.4826 * mad)
  proto_relative : (s - mu_p) / sigma_p
  centroid       : s - mu                  (shift only, no rescaling)

If sigma_p ~ 0 (K=1) proto_relative is undefined and excluded from candidates.
Near-zero dispersion is handled by the eps floor; a fully degenerate case
(eps floor still produces zero) falls back to the raw score.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .config import V3Config


def pairwise_ref_stats(refs: torch.Tensor, cfg: V3Config) -> dict:
    """Statistics from reference embeddings only. refs: [K, D] L2-normalized."""
    k = refs.shape[0]
    sims = refs @ refs.T
    idx = torch.triu_indices(k, k, offset=1)
    vals = sims[idx[0], idx[1]] if k > 1 else sims.flatten()
    mu = float(vals.mean()) if k > 1 else 1.0
    sigma = float(vals.std(unbiased=False)) if k > 1 else 0.0
    med = float(vals.median()) if k > 1 else 1.0
    mad = float((vals - med).abs().median()) if k > 1 else 0.0

    proto = torch.nn.functional.normalize(refs.mean(dim=0), p=2, dim=0)
    p_i = refs @ proto  # references vs their own prototype
    mu_p = float(p_i.mean())
    sigma_p = float(p_i.std(unbiased=False))

    return {
        "k": k,
        "mu": mu,
        "sigma": max(sigma, cfg.calib_eps),
        "med": med,
        "mad": max(mad, cfg.calib_eps),
        "mu_p": mu_p,
        "sigma_p": max(sigma_p, cfg.calib_eps),
        "prototype": proto,
    }


def calibrate(scores: torch.Tensor, stats: dict, method: str, cfg: V3Config) -> torch.Tensor:
    """Transform aggregated query scores using enrollment-only stats.

    scores: [Q] aggregated similarity scores (e.g. mean/max over references).
    Returns calibrated scores (float32 tensor).
    """
    s = scores.float()
    if method == "raw":
        return s
    if method == "z":
        return (s - stats["mu"]) / stats["sigma"]
    if method == "mad":
        return (s - stats["med"]) / (cfg.mad_scale * stats["mad"])
    if method == "proto_relative":
        return (s - stats["mu_p"]) / stats["sigma_p"]
    if method == "centroid":
        return s - stats["mu"]
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibration(
    queries: torch.Tensor,  # [Q, D] L2-normalized query embeddings
    refs: torch.Tensor,  # [K, D] L2-normalized
    agg: str,
    cal: str,
    cfg: V3Config,
) -> Tuple[torch.Tensor, dict]:
    """Full calibration pipeline for a query set against K references.

    Returns (calibrated_scores, stats). stats['prototype'] is included for the
    prototype aggregation path.
    """
    stats = pairwise_ref_stats(refs, cfg)
    if agg == "prototype":
        raw = queries @ stats["prototype"]  # query-to-prototype similarity
    else:
        sims = queries @ refs.T  # [Q, K] per-reference similarities
        raw = aggregate(sims, agg)
    return calibrate(raw, stats, cal, cfg), stats


def aggregate(sims: torch.Tensor, agg: str) -> torch.Tensor:
    if agg == "max":
        return sims.max(dim=1).values
    if agg == "mean":
        return sims.mean(dim=1)
    if agg == "median":
        return sims.median(dim=1).values
    if agg == "top2_mean":
        return sims.topk(2, dim=1).values.mean(dim=1)
    if agg == "prototype":
        raise ValueError("prototype aggregation handled by apply_calibration")
    raise ValueError(f"unknown aggregation: {agg}")


def is_valid_candidate(k: int, agg: str, cal: str, cfg: V3Config) -> bool:
    if k not in cfg.eval_ks:
        return False
    if agg not in cfg.aggregations:
        return False
    if cal not in cfg.calibrations:
        return False
    if agg == "top2_mean" and k < 3:
        return False
    if k < 3 and cal != "raw":
        # pairwise stats are degenerate for K<3
        return False
    if cal == "proto_relative" and k < 3:
        return False
    return True


def valid_candidates(cfg: V3Config):
    return [
        (k, agg, cal)
        for k in cfg.eval_ks
        for agg in cfg.aggregations
        for cal in cfg.calibrations
        if is_valid_candidate(k, agg, cal, cfg)
    ]