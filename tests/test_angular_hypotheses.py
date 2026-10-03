import torch
import pytest
from audit.training import loss_parts


def legacy_loss(pred, clean, noisy):
    p = pred.reshape(-1)
    y = noisy[:, None].expand_as(pred).reshape(-1)
    mse = (torch.where(y.abs() > .001, .5, 1.) * (p-y).square()).mean()
    p, y = p-p.mean(), y-y.mean()
    angular = 1-(p*y).sum()/(p.norm()+1e-8)/(y.norm()+1e-8)
    return mse+angular, mse, angular


def test_default_control_matches_old_loss_and_gradient_exactly():
    torch.manual_seed(9)
    p = torch.randn(11, 4, requires_grad=True)
    clean = torch.randn(11)*.003
    noisy = clean+torch.randn(11)*.005
    old = legacy_loss(p, clean, noisy)
    new = loss_parts(p, clean, noisy, weight_target='noisy')
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(old[0], p, retain_graph=True)[0]
    gb = torch.autograd.grad(new[0], p)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_clean_angular_preserves_mse_and_removes_angular_noise_dependence():
    p = torch.tensor([[.003,.002],[-.004,-.003],[.001,.002],[.006,.004]], requires_grad=True)
    clean = torch.tensor([.002,-.003,0.,.004])
    noisy = torch.tensor([-.004,.005,.003,.001])
    control = loss_parts(p, clean, noisy, 'noisy')
    h1 = loss_parts(p, clean, noisy, 'noisy', angular_target='clean')
    other_noise = loss_parts(p, clean, noisy*2, 'noisy', angular_target='clean')
    torch.testing.assert_close(control[1], h1[1], rtol=0, atol=0)
    g0 = torch.autograd.grad(control[1], p, retain_graph=True)[0]
    g1 = torch.autograd.grad(h1[1], p, retain_graph=True)[0]
    torch.testing.assert_close(g0, g1, rtol=0, atol=0)
    torch.testing.assert_close(h1[2], other_noise[2], rtol=0, atol=0)
    assert not torch.isclose(control[2], h1[2])
    assert torch.isfinite(torch.autograd.grad(h1[0], p)[0]).all()


def test_mean_angular_uses_inference_mean_but_keeps_individual_mse():
    y = torch.tensor([-.003,.001,.004,-.002])
    p = torch.stack([y,y], dim=1)
    offsets = torch.tensor([.002,-.002])
    changed = p+offsets
    a = loss_parts(p,y,y,'noisy',angular_aggregation='mean')
    b = loss_parts(changed,y,y,'noisy',angular_aggregation='mean')
    torch.testing.assert_close(a[2],b[2],rtol=0,atol=1e-7)
    assert b[1] > a[1]
    assert loss_parts(changed,y,y,'noisy')[2] > b[2]


@pytest.mark.parametrize('kwargs', [{'angular_target':'invalid'}, {'angular_aggregation':'invalid'}])
def test_invalid_loss_switch_is_rejected(kwargs):
    with pytest.raises(ValueError):
        loss_parts(torch.ones(3,2),torch.ones(3),torch.ones(3),**kwargs)
