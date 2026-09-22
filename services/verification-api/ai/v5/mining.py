"""V5 mining — P6 K5 M8 with capping for low-sample (15:4) and only-genuine handling."""
from __future__ import annotations
import random
from typing import Dict, List, Tuple
import torch

def sample_batch_writers(all_train_writers: List[str], P: int, rng: random.Random) -> List[str]:
    return rng.sample(all_train_writers, P)

def per_writer_counts(writer_id: str, n_genuine: int, n_forgery: int, K: int, M: int):
    k = min(K, n_genuine)
    m = min(M, n_forgery) if n_forgery>0 else 0
    return k, m

def build_batch_indices(writer_samples: Dict[str, tuple], P: int, K: int, M: int, rng: random.Random) -> List[Tuple[str,int,bool]]:
    # writer_samples: id -> (genuine_paths, forgery_paths)
    writers = sample_batch_writers(list(writer_samples.keys()), P, rng)
    batch=[]
    for w in writers:
        gen, forg = writer_samples[w]
        k,m = per_writer_counts(w, len(gen), len(forg), K, M)
        # sample k genuines without replacement
        gen_idx = rng.sample(range(len(gen)), k) if k>0 else []
        for idx in gen_idx:
            batch.append((w, idx, True))  # True=genuine
        # sample m forgeries
        if m>0:
            forg_idx = rng.sample(range(len(forg)), m)
            for idx in forg_idx:
                batch.append((w, idx, False))
    return batch
