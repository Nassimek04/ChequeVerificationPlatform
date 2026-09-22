import numpy as np
import pytest
import torch

from ai.v2.losses import TripletLoss, triplet_violation


def test_zero_when_satisfied():
    loss = TripletLoss(margin=0.3)
    d_ap = torch.tensor([0.5, 0.5])
    d_an = torch.tensor([1.0, 1.5])
    assert loss(d_ap, d_an).item() == pytest.approx(0.0)


def test_positive_when_violated():
    loss = TripletLoss(margin=0.3)
    d_ap = torch.tensor([0.5])
    d_an = torch.tensor([0.6])
    assert loss(d_ap, d_an).item() == pytest.approx(0.5 - 0.6 + 0.3)


def test_mean_over_batch():
    loss = TripletLoss(margin=0.3)
    d_ap = torch.tensor([0.5, 0.5])
    d_an = torch.tensor([0.6, 1.5])
    expected = ((0.5 - 0.6 + 0.3) + 0.0) / 2
    assert loss(d_ap, d_an).item() == pytest.approx(expected)


def test_margin_configurable():
    l1 = TripletLoss(margin=0.2)
    l2 = TripletLoss(margin=0.5)
    d_ap = torch.tensor([0.5])
    d_an = torch.tensor([0.6])
    assert l1(d_ap, d_an).item() == pytest.approx(0.1)
    assert l2(d_ap, d_an).item() == pytest.approx(0.4)


def test_backward_passes():
    loss = TripletLoss(margin=0.3)
    d_ap = torch.tensor([0.5], requires_grad=True)
    d_an = torch.tensor([0.6], requires_grad=True)
    loss(d_ap, d_an).backward()
    assert d_ap.grad is not None and d_ap.grad.item() == pytest.approx(1.0)
    assert d_an.grad is not None and d_an.grad.item() == pytest.approx(-1.0)


def test_triplet_violation_vector():
    v = triplet_violation(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.6]), 0.3)
    assert v[0].item() == pytest.approx(0.0)
    assert v[1].item() == pytest.approx(0.2)