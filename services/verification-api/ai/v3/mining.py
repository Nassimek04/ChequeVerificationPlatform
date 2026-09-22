"""V3 improved hard-negative mining.

Fixes V2's weakness: semi-hard ratio collapsed toward 0 as the model learned
the in-batch negatives. V3 introduces:

  1. A per-writer CANDIDATE BANK of hard skilled-forgery embeddings. Bank
     entries are collected during training (in-batch skilled negatives that
     were hard/semi-hard) and periodically REFRESHED by re-encoding their
     stored images with the current checkpoint, so stale/trivial candidates
     are replaced.

  2. A SCHEDULED mining policy (validation-agnostic; train writers only):

     Semi-hard negatives (d_ap < d(a,n) < d_ap + margin) are ALWAYS preferred
     (they produce useful gradients); hard negatives (d(a,n) <= d_ap) fill the
     remainder, largest violations first.

     early phase : explore within the semi-hard band (shuffle), fill with the
                   hardest remaining -> random + semi-hard mix
     mid phase   : prefer the hardest semi-hard, then the hardest hard
     late phase  : same as mid, and the bank is refreshed every epoch so the
                   hardest stable skilled forgeries stay fresh

  3. Per-anchor negatives are composed mostly of SAME-WRITER skilled forgeries
     (the attack model) with a configurable minority of other-writer genuine
     impostors.

Bank and schedule NEVER see validation/test writers or labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ai.preprocessing import preprocess

from .config import V3Config
from ai.v2.dataset import WriterSamples, rng_for


class CandidateBank:
    """Per-writer pool of hard skilled-forgery embeddings + source images.

    Images are stored as preprocessed tensors (no augmentation at refresh
    time) so the bank can be re-encoded with the current checkpoint without
    re-reading from disk.
    """

    def __init__(self, cfg: V3Config):
        self.cfg = cfg
        self.cap = cfg.bank_cap_per_writer
        self._entries: Dict[int, List[dict]] = {}

    def add(self, writer: int, img_cpu: torch.Tensor, emb_cpu: torch.Tensor) -> None:
        if writer not in self._entries:
            self._entries[writer] = []
        self._entries[writer].append({
            "img": img_cpu.detach().cpu(),
            "emb": emb_cpu.detach().cpu(),
        })
        if len(self._entries[writer]) > self.cap:
            self._entries[writer] = self._entries[writer][-self.cap:]

    def refresh(self, model, cfg: V3Config, device: torch.device) -> int:
        """Re-encode all bank images with the current model. Returns count."""
        if not self._entries:
            return 0
        model.eval()
        n = 0
        for w, entries in self._entries.items():
            imgs = torch.stack([e["img"] for e in entries]).to(device)
            with torch.inference_mode():
                embs = model.encode(imgs).cpu()
            for e, emb in zip(entries, embs):
                e["emb"] = emb
            n += len(entries)
        return n

    def bank_tensor(self, writer: int, device: torch.device) -> Optional[torch.Tensor]:
        entries = self._entries.get(writer)
        if not entries:
            return None
        return torch.stack([e["emb"] for e in entries]).to(device)

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())


def build_batch(
    samples: Dict[int, WriterSamples],
    cfg: V3Config,
    epoch: int,
    batch_idx: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List]:
    """Writer-grouped batch: P writers x (K genuines + M forgeries).

    Returns (images [N,C,H,W], writers, is_genuine, entries).
    entries is the ordered (writer, is_genuine, path) list used for metadata.
    """
    rng = rng_for(cfg.seed, "batch", epoch, batch_idx)
    train_writers = list(cfg.train_writers)
    assert len(train_writers) >= cfg.writers_per_batch
    writers = rng.sample(train_writers, cfg.writers_per_batch)
    entries = []
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
    return images, writers_arr, is_genuine, entries


def _category(d_ap: float, d_an: float, margin: float) -> str:
    if d_an <= d_ap:
        return "hard"
    if d_an < d_ap + margin:
        return "semi-hard"
    return "easy"


def select_triplets_v3(
    embeddings: torch.Tensor,
    images: torch.Tensor,
    writers: np.ndarray,
    is_genuine: np.ndarray,
    cfg: V3Config,
    epoch: int,
    bank: CandidateBank,
    device: torch.device,
) -> Dict:
    """Mine triplets for one batch.

    Returns:
      dist_ap / dist_an : [M] differentiable distances for the loss
      anchor / positive / negative : index metadata (negative index >= N marks a
        bank candidate; N = number of in-batch images)
      category / neg_kind : per-triplet labels
      stats : mining statistics (hard/semi/easy %, skilled %, mean d_ap/d_an,
              active loss fraction)
      bank_updates : [(writer, img_cpu, emb_cpu)] to feed back into the bank
    """
    margin = float(cfg.margin)
    n_skilled = int(round(cfg.negatives_per_anchor * cfg.skilled_negative_fraction))
    n_random = cfg.negatives_per_anchor - n_skilled
    phase = cfg.schedule_phase(epoch)

    dist = torch.cdist(embeddings, embeddings, p=2)
    dist_d = dist.detach()
    N = embeddings.shape[0]
    arange = torch.arange(N, device=embeddings.device)
    writers_t = torch.as_tensor(writers, device=embeddings.device)
    genuine_t = torch.as_tensor(is_genuine, device=embeddings.device)

    a_idx, p_idx, n_idx = [], [], []
    dist_ap_float, dist_an_float = [], []
    dist_ap_grad = []  # differentiable scalars, order matches a_idx
    dist_an_grad = []  # differentiable scalars, order matches a_idx
    cats, kinds = [], []
    stats = Counter()
    bank_updates = []
    anchors_with_neg = 0
    active_loss = 0

    for a in range(N):
        if not bool(is_genuine[a]):
            continue
        same_w = writers_t == writers_t[a]
        pos_cand = torch.nonzero(same_w & genuine_t & (arange != a)).flatten().tolist()
        if not pos_cand:
            continue
        rng = rng_for(cfg.seed, "pos", epoch, a)
        p = int(pos_cand[rng.randrange(len(pos_cand))])
        d_ap = float(dist_d[a, p])

        # --- skilled candidate pool (same-writer forgeries) ---
        inbatch_skilled = torch.nonzero(same_w & ~genuine_t).flatten().tolist()
        bank_emb = bank.bank_tensor(int(writers_t[a]), device)  # [B, D] or None
        bank_base = N

        cand_skilled = [(float(dist_d[a, i]), ("in", i)) for i in inbatch_skilled]
        bank_dist_full = None
        if bank_emb is not None:
            # differentiable wrt the anchor embedding; bank embeddings are frozen
            bank_dist_full = torch.cdist(embeddings[a : a + 1], bank_emb, p=2)[0]
            d_bank = bank_dist_full.detach()
            for j in range(bank_emb.shape[0]):
                cand_skilled.append((float(d_bank[j]), ("bank", bank_base + j)))

        random_pool = torch.nonzero((writers_t != writers_t[a]) & genuine_t).flatten().tolist()
        cand_random = [(float(dist_d[a, i]), ("in", i)) for i in random_pool]

        cand_skilled.sort(key=lambda t: t[0])
        cand_random.sort(key=lambda t: t[0])

        # Schedule-aware semi-hard-first selection (V2-compatible core, with
        # bank + schedule on top):
        #   semi-hard band : d_ap < d(a,n) < d_ap + margin   (the useful gradient)
        #   hard           : d(a,n) <= d_ap                   (largest violation)
        #   early : explore -> shuffle WITHIN the semi-hard band, then fill with
        #           the hardest remaining (hard first); a random+semi-hard mix
        #   mid   : prefer the hardest semi-hard, then the hardest hard
        #   late  : same as mid; the bank is additionally refreshed every epoch
        def pick(pairs, n, mode):
            if n <= 0 or not pairs:
                return []
            semi = [t for t in pairs if d_ap < t[0] < d_ap + margin]
            if mode == "early":
                chosen = list(semi)
                rng.shuffle(chosen)
            else:
                chosen = semi[:]  # ascending distance -> hardest semi-hard first
            chosen = chosen[:n]
            if len(chosen) < n:
                used = set(id(t) for t in chosen)
                rest = [t for t in pairs if id(t) not in used]
                chosen.extend(rest[: n - len(chosen)])
            return chosen[:n]

        sel_skilled = pick(cand_skilled, n_skilled, phase)
        sel_random = pick(cand_random, n_random, phase)

        if sel_skilled or sel_random:
            anchors_with_neg += 1

        def _grad_an(kind):
            """Differentiable distance for a chosen negative."""
            if kind[0] == "in":
                return dist[a, kind[1]]
            j = kind[1] - bank_base
            return bank_dist_full[j]

        for d_an, kind in sel_skilled:
            cat = _category(d_ap, d_an, margin)
            a_idx.append(a); p_idx.append(p); n_idx.append(kind[1])
            dist_ap_float.append(d_ap); dist_an_float.append(d_an)
            dist_ap_grad.append(dist[a, p])
            dist_an_grad.append(_grad_an(kind))
            cats.append(cat); kinds.append("skilled")
            stats[("skilled", cat)] += 1
            if cat != "easy" and kind[0] == "in":
                w = int(writers[a])
                bank_updates.append((w, images[kind[1]], embeddings[kind[1]]))
            if cat != "easy":
                active_loss += 1

        for d_an, kind in sel_random:
            cat = _category(d_ap, d_an, margin)
            a_idx.append(a); p_idx.append(p); n_idx.append(kind[1])
            dist_ap_float.append(d_ap); dist_an_float.append(d_an)
            dist_ap_grad.append(dist[a, p])
            dist_an_grad.append(_grad_an(kind))
            cats.append(cat); kinds.append("random")
            stats[("random", cat)] += 1
            if cat != "easy":
                active_loss += 1

    total_neg = len(n_idx)
    hard = sum(v for (k, c), v in stats.items() if c == "hard")
    semi = sum(v for (k, c), v in stats.items() if c == "semi-hard")
    easy = sum(v for (k, c), v in stats.items() if c == "easy")
    skilled_n = sum(v for (k, c), v in stats.items() if k == "skilled")
    random_n = sum(v for (k, c), v in stats.items() if k == "random")

    if a_idx:
        dist_ap_t = torch.stack(dist_ap_grad)
        dist_an_t = torch.stack(dist_an_grad)
    else:
        dist_ap_t = torch.zeros(0, device=embeddings.device)
        dist_an_t = torch.zeros(0, device=embeddings.device)

    skilled_dists = [d for d, k in zip(dist_an_float, kinds) if k == "skilled"]
    stats_out = {
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
        "mean_d_ap": float(torch.as_tensor(dist_ap_float).mean()) if dist_ap_float else 0.0,
        "mean_d_an_skilled": float(torch.as_tensor(skilled_dists).mean()) if skilled_dists else 0.0,
        "active_loss_fraction": float(active_loss / total_neg) if total_neg else 0.0,
        "phase": phase,
        "bank_size": len(bank),
    }

    return {
        "anchor": torch.as_tensor(a_idx, dtype=torch.long, device=embeddings.device),
        "positive": torch.as_tensor(p_idx, dtype=torch.long, device=embeddings.device),
        "negative": torch.as_tensor(n_idx, dtype=torch.long, device=embeddings.device),
        "dist_ap": dist_ap_t,
        "dist_an": dist_an_t,
        "category": cats,
        "neg_kind": kinds,
        "stats": stats_out,
        "bank_updates": bank_updates,
        "phase": phase,
    }