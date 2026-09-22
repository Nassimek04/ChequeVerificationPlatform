import torch
import torch.nn.functional as F


def make_embeddings(device=torch.device("cpu")):
    """Craft distinct cluster embeddings for a fake batch.

    Layout: 2 writers x (2 genuine + 2 forged) = 8 embeddings.
    Writer 1 genuines near angle 0; writer 1 forgeries offset +0.3;
    writer 2 gen/forg around angle 1.5.
    """
    base = [0.0, 0.02, 0.3, 0.32, 1.5, 1.52, 1.75, 1.78]
    d = torch.as_tensor(base, device=device)
    e = torch.stack([torch.cos(d), torch.sin(d)], dim=1)
    return F.normalize(e, p=2, dim=1)