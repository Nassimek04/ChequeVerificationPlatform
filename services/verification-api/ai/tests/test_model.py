import torch

from ai.model import cosine_similarity, euclidean_distance


def test_model_forward_pass(model):
    x1 = torch.randn(4, 3, 32, 64)
    x2 = torch.randn(4, 3, 32, 64)
    e1, e2 = model(x1, x2)
    assert e1.shape == e2.shape


def test_embedding_dimension(model, tiny_cfg):
    x = torch.randn(2, 3, 32, 64)
    e, _ = model(x, x)
    assert e.shape[1] == tiny_cfg.embedding_dim


def test_embeddings_l2_normalized(model):
    x = torch.randn(8, 3, 32, 64)
    e, _ = model(x, x)
    norms = e.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_shared_weights_single_backbone(model):
    # Both branches MUST share the same parameters: feeding the same image
    # through encode twice must give identical embeddings.
    x = torch.randn(4, 3, 32, 64)
    e_a = model.encode(x)
    e_b = model.encode(x)
    assert torch.allclose(e_a, e_b, atol=1e-6)


def test_encode_matches_forward(model):
    x1 = torch.randn(3, 3, 32, 64)
    x2 = torch.randn(3, 3, 32, 64)
    e1, e2 = model(x1, x2)
    assert torch.allclose(e1, model.encode(x1), atol=1e-6)
    assert torch.allclose(e2, model.encode(x2), atol=1e-6)


def test_distance_and_similarity_consistent(model):
    x1 = torch.randn(2, 3, 32, 64)
    x2 = torch.randn(2, 3, 32, 64)
    e1, e2 = model(x1, x2)
    d = euclidean_distance(e1, e2)
    s = cosine_similarity(e1, e2)
    assert torch.allclose(d, (2 - 2 * s).clamp(min=0).sqrt(), atol=1e-5)
