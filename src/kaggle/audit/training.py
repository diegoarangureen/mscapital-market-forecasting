"""Backend-independent training math; device-specific RNG is handled explicitly."""
import random
import numpy as np
import torch


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if device.type == 'xla':
        import torch_xla
        import torch_xla.core.xla_model as xm
        if hasattr(torch_xla, 'manual_seed'):
            torch_xla.manual_seed(seed, device=device)
        else:
            xm.set_rng_state(seed, str(device))


def rng_state(device):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    if device.type == 'xla':
        import torch_xla.core.xla_model as xm
        state['xla'] = xm.get_rng_state(str(device))
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])
    if device.type == 'xla':
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(state['xla'], str(device))


def fit_scale(x):
    med = np.median(x, axis=0)
    qd = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    zero = qd == 0
    qd[zero] = .5 * (x.max(axis=0)[zero] - x.min(axis=0)[zero])
    fac = 1. / (qd + 1e-30)
    fac[qd == 0] = 0.
    return med, fac


def apply_scale(x, med, fac):
    s = fac[None, :] * (x - med[None, :])
    return (s / np.sqrt(1 + (s / 3) ** 2)).astype(np.float32)


def loss_parts(pred, clean, noisy, weight_target='clean', angular='pearson', lambda_cos=1.):
    p = pred.reshape(-1)
    y = noisy[:, None].expand_as(pred).reshape(-1)
    clean_flat = clean[:, None].expand_as(pred).reshape(-1)
    if weight_target == 'uniform':
        w = torch.ones_like(y)
    elif weight_target in ('clean', 'noisy'):
        source = clean_flat if weight_target == 'clean' else y
        w = torch.where(source.abs() > .001, .5, 1.)
    else:
        raise ValueError('Unknown weight_target')
    mse = (w * (p - y).square()).mean()
    if angular == 'pearson':
        p, y = p - p.mean(), y - y.mean()
    elif angular != 'cosine':
        raise ValueError('Unknown angular loss')
    angular_loss = 1 - (p * y).sum() / (p.norm() + 1e-8) / (y.norm() + 1e-8)
    return mse + lambda_cos * angular_loss, mse, angular_loss


def gradient_norm(term, parameters):
    grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    values = [g.detach().square().sum() for g in grads if g is not None]
    return torch.stack(values).sum().sqrt() if values else term.detach() * 0


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value
