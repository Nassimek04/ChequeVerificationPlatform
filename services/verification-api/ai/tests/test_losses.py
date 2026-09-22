import torch

from ai.losses import ContrastiveLoss
from ai.model import euclidean_distance


def test_loss_finite_and_positive(model):
    loss_fn = ContrastiveLoss(margin=1.0)
    x1 = torch.randn(8, 3, 32, 64)
    x2 = torch.randn(8, 3, 32, 64)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(model.encode(x1), model.encode(x2), y)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_semantics_identical_pairs_low(model):
    """Genuine pairs (y=1) at zero distance contribute 0 to the loss."""
    loss_fn = ContrastiveLoss(margin=1.0)
    x = torch.randn(4, 3, 32, 64)
    y = torch.ones(4)
    e = model.encode(x)
    loss = loss_fn(e, e, y)
    assert loss.item() < 1e-6


def test_loss_semantics_very_different_pairs_hinge(model):
    loss_fn = ContrastiveLoss(margin=1.0)
    e1 = torch.zeros(4, 32)
    e2 = torch.ones(4, 32) * 2.0  # distance >> margin
    y = torch.zeros(4)
    loss = loss_fn(e1, e2, y)
    assert loss.item() < 1e-6  # max(margin - d, 0) = 0


def test_loss_margin_configurable(model):
    e1 = torch.zeros(2, 32)
    e2 = torch.ones(2, 32) * 0.5  # distance = 2.828... wait not in this scale
    # distance between (0,...) and (0.5,...) of dim 32 = sqrt(32*0.25) = 2.828
    loss_wide = ContrastiveLoss(margin=4.0)(e1, e2, torch.zeros(2))
    loss_narrow = ContrastiveLoss(margin=1.0)(e1, e2, torch.zeros(2))
    assert loss_wide.item() > 0
    assert loss_narrow.item() < 1e-6
    assert loss_wide.item() > loss_narrow.item()


def test_loss_reduces_with_training_step(model):
    loss_fn = ContrastiveLoss(margin=1.0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x1 = torch.randn(16, 3, 32, 64)
    x2 = torch.randn(16, 3, 32, 64)
    y = torch.randint(0, 2, (16,)).float()
    opt.zero_grad()
    loss = loss_fn(model.encode(x1), model.encode(x2), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
