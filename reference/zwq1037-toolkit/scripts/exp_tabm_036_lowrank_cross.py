"""Frozen TabM385-k32 with two rank16 explicit cross layers."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import gc
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import exp_tabm_034_market385_k32_ple as setup

recipe = setup.recipe
PROJECT = setup.PROJECT_DIR
RUN = PROJECT / 'data/interim/tree_experiments/EXP-TABM-036-LOWRANK-CROSS16'
FOLD = 'train059_valid6270_ex66'


class CrossTabM(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.tabm = recipe.TabM.make(n_num_features=dimension, cat_cardinalities=None,
            d_out=1, k=32, n_blocks=2, d_block=256, dropout=0.1, arch_type='tabm')
        # 保留主干和训练的随机序列，交叉分支从恒等映射开始。
        # Preserve backbone RNG and start the cross branch as identity.
        with torch.random.fork_rng(devices=[]):
            self.down = nn.ModuleList([nn.Linear(dimension, 16, bias=False) for _ in range(2)])
            self.up = nn.ModuleList([nn.Linear(16, dimension, bias=False) for _ in range(2)])
            for layer in self.down:
                nn.init.normal_(layer.weight, std=0.01)
            for layer in self.up:
                nn.init.zeros_(layer.weight)

    def forward(self, x):
        hidden = x
        for down, up in zip(self.down, self.up):
            hidden = hidden + x * up(down(hidden))
        return self.tabm(hidden)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    labels = pd.read_feather(PROJECT / 'data/raw/label.feather',
        columns=['sample_id', 'month', 'target']).sort_values('sample_id').reset_index(drop=True)
    ids = labels.sample_id.to_numpy()
    months = labels.month.to_numpy(dtype=np.int16)
    target = labels.target.to_numpy(dtype=np.float32)
    base, columns = setup.gru.load_relative319(PROJECT, labels)
    xs, _ = recipe.add_relative_features(base, months, columns)
    frames = []
    for filename, names in [('train_order_quote_position_features.feather', setup.ORDER_COLUMNS),
                            ('train_market_microstructure_features.feather', setup.FEATURE_COLUMNS)]:
        frame = pd.read_feather(PROJECT / 'data/processed' / filename,
            columns=['sample_id', *names]).sort_values('sample_id').reset_index(drop=True)
        assert np.array_equal(ids, frame.sample_id.to_numpy())
        frames.append(frame[names].to_numpy(dtype=np.float32))
    features = np.concatenate([base, xs, *frames], axis=1)
    assert features.shape[1] == 385
    del base, xs, frames, labels
    gc.collect()

    # 初始化等价性和分支梯度检查；不改变正式训练的随机状态。
    # Check initial equivalence and branch gradients before the formal run.
    with torch.random.fork_rng():
        smoke = CrossTabM(385).to(device).eval()
        x = torch.randn(16, 385, device=device)
        with torch.no_grad():
            assert torch.equal(smoke(x), smoke.tabm(x))
        smoke(x).square().mean().backward()
        assert all(layer.weight.grad.abs().sum() > 0 for layer in smoke.up)
        assert smoke(x).shape == (16, 32, 1)
        del smoke, x
    torch.cuda.empty_cache()
    print('cross_identity_shape_gradient_smoke_pass', flush=True)
    recipe.make_model = lambda dimension, model_device: CrossTabM(dimension).to(model_device)
    result = recipe.run_model(FOLD + '_cross16', features, target, months, 59, 62, 70, device)
    indices = result['validation_indices']
    folder = RUN / FOLD
    folder.mkdir(parents=True, exist_ok=True)
    old = pd.read_feather(PROJECT / 'data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32' / FOLD / 'validation_predictions.feather')
    assert np.array_equal(ids[indices], old.sample_id.to_numpy())
    pd.DataFrame({'sample_id': ids[indices], 'month': months[indices], 'target': target[indices],
        'baseline': old.candidate.to_numpy(), 'candidate': result['prediction']}).to_feather(folder / 'validation_predictions.feather')
    setup.save_model_result(result, folder / 'candidate_result.json')
    mask = months[indices] != 66
    baseline = recipe.cosine(target[indices][mask], old.candidate.to_numpy()[mask])
    candidate = result['metrics']['months_62_70_without_66']
    report = {'experiment': 'EXP-TABM-036-LOWRANK-CROSS16', 'baseline': float(baseline),
        'candidate': float(candidate), 'delta': float(candidate-baseline),
        'change': 'two rank16 identity-initialized explicit cross layers; recipe unchanged',
        'seconds': result['seconds'], 'single_gate': 0.001, 'passed_single': candidate-baseline >= 0.001}
    (RUN / 'score_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
