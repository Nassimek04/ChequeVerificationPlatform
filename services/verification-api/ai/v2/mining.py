"""Hard/semi-hard negative mining + writer-grouped batch construction.

THE MAIN V2 EXPERIMENT.

Batch design
------------
Each batch samples `P = writers_per_batch` writers. Every writer contributes
`K = genuines_per_writer` genuine signatures and `M = forgeries_per_writer`
skilled forgeries -> a batch has P*(K+M) images.

For every genuine anchor of writer W:
  - positive: a genuine signature of the SAME writer W (never the anchor image)
  - negatives: skilled forgeries of W (same claimed writer, forged) and
    genuine signatures of OTHER writers (random impostors), in a configurable
    ratio `skilled_negative_fraction`.

Mining
------
Negatives are selected by distance to the anchor (Euclidean on L2-normalized
embeddings):
  - semi-hard preferred:  d(a,p) < d(a,n) < d(a,p) + margin
  - if not enough semi-hard: fill with the hardest remaining negatives
    (d(a,n) <= d(a,p)), i.e. the largest margin violations.
  - easy negatives (d(a,n) >= d(a,p)+margin) are used only if nothing else
    exists, so training focuses on useful gradients.

Mining NEVER uses validation/test writers or labels. It only operates on the
writers present in the current training batch.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch

from ai.preprocessing import preprocess

from .config import V2Config
from .dataset import WriterSamples, rng_for


def build_batch(
    samples: Dict[int, WriterSamples],
    cfg: V2Config,
    epoch: int,
    batch_idx: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Build one writer-grouped training batch (with augmentation).

    Returns (images [N,C,H,W], writers [N], is_genuine [N] bool).
    """
    rng = rng_for(cfg.seed, "batch", epoch, batch_idx)
    train_writers = list(cfg.train_writers)
    assert len(train_writers) >= cfg.writers_per_batch
    writers = rng.sample(train_writers, cfg.writers_per_batch)

    entries: List[Tuple[int, bool, object]] = []  # (writer, is_genuine, path)
    for w in writers:
        ws = samples[w]
        g_idx = rng.sample(range(ws.n_genuine), cfg.genuines_per_writer)
        f_idx = rng.sample(range(ws.n_forgery), cfg.forgeries_per_writer)
        for i in g_idx:
            entries.append((w, True, ws.genuine_paths[i]))
        for i in f_idx:
            entries.append((w, False, ws.forgery_paths[i]))
    rng.shuffle(entries)

    images = []
    for pos, (w, gen, path) in enumerate(entries):
        seed_img = (epoch * 1_000_003 + batch_idx * 1000 + pos) % (2**31)
        images.append(preprocess(path, cfg, augment=cfg.augment, seed=seed_img))
    images = torch.stack(images)
    writers_arr = np.asarray([e[0] for e in entries], dtype=np.int64)
    is_genuine = np.asarray([e[1] for e in entries], dtype=bool)
    return images, writers_arr, is_genuine


def _ranked(dist: torch.Tensor, a: int, pool: List[int]) -> List[int]:
    return sorted(pool, key=lambda i: float(dist[a, i]))


def _category(d_ap: float, d_an: float, margin: float) -> str:
    if d_an <= d_ap:
        return "hard"
    if d_an < d_ap + margin:
        return "semi-hard"
    return "easy"


def select_triplets(
    embeddings: torch.Tensor,
    writers: np.ndarray,
    is_genuine: np.ndarray,
    cfg: V2Config,
    epoch: int,
) -> Dict:
    """Mine triplets inside a batch.

    Returns a dict with tensor arrays (anchor/positive/negative indices,
    category, neg_kind) plus mining statistics.
    """
    margin = float(cfg.margin)
    n_skilled = int(round(cfg.negatives_per_anchor * cfg.skilled_negative_fraction))
    n_random = cfg.negatives_per_anchor - n_skilled

    dist = torch.cdist(embeddings, embeddings, p=2)  # [N,N]
    dist_d = dist.detach()  # for selection; `dist` stays differentiable for the loss
    N = embeddings.shape[0]
    arange = torch.arange(N, device=embeddings.device)
    writers_t = torch.as_tensor(writers, device=embeddings.device)
    genuine_t = torch.as_tensor(is_genuine, device=embeddings.device)

    a_idx: List[int] = []
    p_idx: List[int] = []
    n_idx: List[int] = []
    cats: List[str] = []
    kinds: List[str] = []
    stats = Counter()
    anchors_with_neg = 0

    for a in range(N):
        if not bool(is_genuine[a]):
            continue  # anchors are genuine signatures only
        same_w = (writers_t == writers_t[a])
        pos_cand = torch.nonzero(same_w & genuine_t & (arange != a)).flatten().tolist()
        if not pos_cand:
            continue
        rng = rng_for(cfg.seed, "pos", epoch, a)
        p = int(pos_cand[rng.randrange(len(pos_cand))])
        d_ap = float(dist_d[a, p])

        skilled = torch.nonzero(same_w & ~genuine_t).flatten().tolist()
        random_pool = torch.nonzero((writers_t != writers_t[a]) & genuine_t).flatten().tolist()

        def pick(pool: List[int], n: int) -> List[int]:
            if not pool:
                return []
            ranked = _ranked(dist_d, a, pool)
            # prefer semi-hard in ascending distance order
            semi = [i for i in ranked if d_ap < float(dist_d[a, i]) < d_ap + margin]
            chosen = semi[:n]
            chosen_set = set(chosen)
            if len(chosen) < n:
                for i in ranked:
                    if i in chosen_set:
                        continue
                    chosen.append(i)
                    chosen_set.add(i)
                    if len(chosen) >= n:
                        break
            return chosen[:n]

        sel_skilled = pick(skilled, n_skilled)
        sel_random = pick(random_pool, n_random)
        if sel_skilled or sel_random:
            anchors_with_neg += 1

        for i in sel_skilled:
            cat = _category(d_ap, float(dist_d[a, i]), margin)
            a_idx.append(a); p_idx.append(p); n_idx.append(i)
            cats.append(cat); kinds.append("skilled")
            stats[("skilled", cat)] += 1
        for i in sel_random:
            cat = _category(d_ap, float(dist_d[a, i]), margin)
            a_idx.append(a); p_idx.append(p); n_idx.append(i)
            cats.append(cat); kinds.append("random")
            stats[("random", cat)] += 1

    total_neg = len(n_idx)
    hard = sum(v for (k, c), v in stats.items() if c == "hard")
    semi = sum(v for (k, c), v in stats.items() if c == "semi-hard")
    easy = sum(v for (k, c), v in stats.items() if c == "easy")
    skilled_n = sum(v for (k, c), v in stats.items() if k == "skilled")
    random_n = sum(v for (k, c), v in stats.items() if k == "random")

    mining_stats = {
        "anchors": int(anchors_with_neg),
        "triplets": total_neg,
        "skilled_negatives": int(skilled_n),
        "random_negatives": int(random_n),
        "skilled_fraction_mined": float(skilled_n / total_neg) if total_neg else 0.0,
        "hard": int(hard),
        "semi_hard": int(semi),
        "easy": int(easy),
        "hard_ratio": float(hard / total_neg) if total_neg else 0.0,
        "semi_hard_ratio": float(semi / total_neg) if total_neg else 0.0,
        "hard_plus_semi_ratio": float((hard + semi) / total_neg) if total_neg else 0.0,
    }
    return {
        "anchor": torch.as_tensor(a_idx, dtype=torch.long, device=embeddings.device),
        "positive": torch.as_tensor(p_idx, dtype=torch.long, device=embeddings.device),
        "negative": torch.as_tensor(n_idx, dtype=torch.long, device=embeddings.device),
        "category": cats,
        "neg_kind": kinds,
        "dist": dist,
        "stats": mining_stats,
    }
