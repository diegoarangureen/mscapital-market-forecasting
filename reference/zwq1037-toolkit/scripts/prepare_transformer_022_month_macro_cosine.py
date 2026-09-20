"""Generate the month-macro cosine Transformer experiment from EXP-020."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/exp_transformer_020_factorized_ema_dev.py"
output_path = ROOT / "scripts/exp_transformer_022_month_macro_cosine.py"
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    'RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA"',
    'RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-022-MONTH-MACRO-COSINE"',
)

sampler = '''\n\nclass TwoMonthBalancedBatchSampler:\n    """Yield equal halves from two uniformly selected training months."""\n\n    def __init__(self, month_positions, batch_size, batch_count, seed):\n        if batch_size % 2:\n            raise ValueError("batch_size must be even")\n        self.month_positions = [np.asarray(values, dtype=np.int64) for values in month_positions]\n        if any(len(values) < batch_size // 2 for values in self.month_positions):\n            raise ValueError("Each month must contain at least half a batch")\n        self.half = batch_size // 2\n        self.batch_count = int(batch_count)\n        self.seed = int(seed)\n        self.epoch = 0\n\n    def set_epoch(self, epoch):\n        self.epoch = int(epoch)\n\n    def __len__(self):\n        return self.batch_count\n\n    def __iter__(self):\n        rng = np.random.default_rng(self.seed + self.epoch)\n        month_count = len(self.month_positions)\n        for _ in range(self.batch_count):\n            chosen = rng.choice(month_count, size=2, replace=False)\n            left = rng.choice(self.month_positions[int(chosen[0])], size=self.half, replace=False)\n            right = rng.choice(self.month_positions[int(chosen[1])], size=self.half, replace=False)\n            yield np.concatenate([left, right]).tolist()\n'''
source = source.replace("\ndef main():", sampler + "\n\ndef main():")

old_loader = '''    generator = torch.Generator().manual_seed(SEED + 17)\n    train_loader = DataLoader(\n        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,\n        num_workers=0, pin_memory=True, drop_last=True,\n    )'''
new_loader = '''    train_months = np.asarray(months[train_indices])\n    month_values = sorted(int(value) for value in np.unique(train_months))\n    month_positions = [np.flatnonzero(train_months == value) for value in month_values]\n    batch_sampler = TwoMonthBalancedBatchSampler(\n        month_positions, BATCH_SIZE, len(train_indices) // BATCH_SIZE, SEED + 17\n    )\n    train_loader = DataLoader(\n        train_dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,\n    )'''
if old_loader not in source:
    raise RuntimeError("Could not locate original train loader")
source = source.replace(old_loader, new_loader)
source = source.replace(
    '''    for epoch in range(1, EPOCHS + 1):\n        model.train()''',
    '''    for epoch in range(1, EPOCHS + 1):\n        batch_sampler.set_epoch(epoch)\n        model.train()''',
)
old_loss = '''            prediction = torch.nan_to_num(model(*inputs))\n            loss = 0.35 * F.smooth_l1_loss(prediction, batch_target) + 0.65 * namespace["cosine_loss"](prediction, batch_target)'''
new_loss = '''            prediction = torch.nan_to_num(model(*inputs))\n            half = len(batch_target) // 2\n            month_macro_cosine = 0.5 * (\n                namespace["cosine_loss"](prediction[:half].float(), batch_target[:half].float())\n                + namespace["cosine_loss"](prediction[half:].float(), batch_target[half:].float())\n            )\n            loss = 0.35 * F.smooth_l1_loss(prediction, batch_target) + 0.65 * month_macro_cosine'''
if old_loss not in source:
    raise RuntimeError("Could not locate original loss")
source = source.replace(old_loss, new_loss)
source = source.replace(
    '"generator_state": generator.get_state(),',
    '"sampler_epoch": epoch,',
)

old_summary = '''    deltas = {\n        key: ema_best["scores"][key] - raw_best["scores"][key]\n        for key in ["all_ex66", "62_65", "67_70"]\n    }\n    passed = bool(deltas["all_ex66"] >= 0.0001 and deltas["62_65"] >= 0 and deltas["67_70"] >= 0)'''
new_summary = '''    baseline_summary = json.loads((ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA/score_summary.json").read_text(encoding="utf-8"))\n    baseline_scores = baseline_summary["raw_best"]["scores"]\n    candidate_name = max(("raw", "ema"), key=lambda name: best[name]["scores"]["62_65"])\n    candidate = best[candidate_name]\n    deltas = {\n        key: candidate["scores"][key] - baseline_scores[key]\n        for key in ["all_ex66", "62_65", "67_70"]\n    }\n    monthly_deltas = {\n        month: candidate["scores"]["monthly"][month] - baseline_scores["monthly"][month]\n        for month in baseline_scores["monthly"]\n    }\n    passed = bool(\n        deltas["62_65"] >= 0.0005\n        and deltas["67_70"] >= 0.0005\n        and deltas["all_ex66"] >= 0.0007\n        and min(monthly_deltas.values()) >= -0.002\n    )'''
if old_summary not in source:
    raise RuntimeError("Could not locate original summary comparison")
source = source.replace(old_summary, new_summary)
source = source.replace(
    '"experiment": "EXP-TRANSFORMER-020-FACTORIZED-EMA",',
    '"experiment": "EXP-TRANSFORMER-022-MONTH-MACRO-COSINE",\n        "change": "two uniformly sampled months per batch; separate uncentered cosine averaged across month halves",\n        "batch_months": 2,\n        "samples_per_month_per_batch": BATCH_SIZE // 2,\n        "baseline": baseline_scores,\n        "selected_candidate": candidate_name,\n        "monthly_deltas": monthly_deltas,',
)
source = source.replace(
    '"pass_gate": "all_ex66>=+0.0001 and 62-65/67-70 deltas>=0",',
    '"pass_gate": "vs EXP020 raw: selection>=+0.0005, forward>=+0.0005, all>=+0.0007, worst month>=-0.002",',
)
source = source.replace(
    '"architecture": "factorized per-stream Transformer encoders + static379 fusion",',
    '"architecture": "factorized per-stream Transformer encoders + static379 fusion; month-macro training batches",',
)
compile(source, output_path.name, "exec")
output_path.write_text(source, encoding="utf-8")
print(output_path)
