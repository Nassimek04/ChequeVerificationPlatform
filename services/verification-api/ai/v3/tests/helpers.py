import torch
import torch.nn.functional as F


def make_writer_emb(n_gen=24, n_forg=24, device=torch.device("cpu"), seed=0):
    """Genuines near (1,0,0), forgeries near (0,1,0); distinct per writer seed.

    Returns two [n, 3] L2-normalized embedding tensors (genuines, forgeries).
    """
    g = torch.Generator().manual_seed(seed)
    base_g = torch.tensor([1.0, 0.0, 0.0])
    base_f = torch.tensor([0.0, 1.0, 0.0])
    gemb = F.normalize(base_g + 0.1 * torch.randn(n_gen, 3, generator=g), p=2, dim=1)
    femb = F.normalize(base_f + 0.1 * torch.randn(n_forg, 3, generator=g), p=2, dim=1)
    return gemb.to(device), femb.to(device)


def fake_emb_map(writers, seed=0, device=torch.device("cpu")):
    return {w: make_writer_emb(seed=seed + w, device=device) for w in writers}


def load_v1_v2_hashes():
    """Recorded SHA256 before any V3 work; V1/V2 must never change."""
    return {
        "v1": "5b611b151aff332f4d64d144a6ce5924c552c38154f201e7858912a255daac20",
        "v2": "95fdc3f1120e93468c0c99387748d20d5321430eacb8c175fbfcda323973b98a",
    }