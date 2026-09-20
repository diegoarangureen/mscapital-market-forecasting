"""Kaggle-only tail appended to the existing subsecond-event notebook source.

This file is concatenated into a self-contained notebook. It deliberately reuses
the already-tested cache builders and model components defined earlier.
"""


EVENT_RUN = Path('/kaggle/working/market_conditioned_event_residual')
EVENT_RUN.mkdir(parents=True, exist_ok=True)
EVENT_EPOCHS = 3
EVENT_BATCH_SIZE = 256
EVENT_LR = 2e-4
TX_CONTEXT_DIM = 20
ORDER_CONTEXT_DIM = 28


class StrictBaseModel(nn.Module):
    """Exact factorized four-source architecture used by strict dev V7."""

    def __init__(self, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.market = _FactorizedStreamEncoder(len(MARKET_FEATURES), MARKET_LEN, d_model, nhead, nlayers, dropout)
        self.transaction = _FactorizedStreamEncoder(len(TX_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout)
        self.order = _FactorizedStreamEncoder(len(ORDER_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout)
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192), nn.SiLU(), nn.LayerNorm(192),
            nn.Dropout(dropout), nn.Linear(192, d_model), nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.source_attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(self, market, transaction, order, static):
        summaries = torch.stack([
            self.market(market), self.transaction(transaction), self.order(order),
            self.static_adapter(static),
        ], dim=1)
        summaries = self.fusion(summaries + self.source_embedding)
        weights = torch.softmax(self.source_attention(summaries).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head((summaries * weights).sum(dim=1)).squeeze(-1)


def canonical_state(saved):
    result = {}
    for name, value in saved.items():
        while name.startswith(('module.', 'parallel.')):
            name = name.split('.', 1)[1]
        if name in result:
            raise RuntimeError('Duplicate canonical checkpoint key: ' + name)
        result[name] = value
    return result


class BasePredictionDataset(Dataset):
    def __init__(self, rows, arrays, static, static_norm):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.arrays = arrays
        self.static = static
        self.static_norm = static_norm

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def normalize_sequence(values, name):
        values = np.asarray(values, dtype=np.float32)
        padding = np.abs(values).sum(axis=-1) == 0
        mean = np.asarray(BUNDLED_STREAM_NORM[name]['mean'], dtype=np.float32)
        std = np.asarray(BUNDLED_STREAM_NORM[name]['std'], dtype=np.float32)
        values = np.clip(np.nan_to_num((values - mean) / std), -8, 8)
        values[padding] = 0
        return torch.from_numpy(values.astype(np.float32, copy=False))

    def __getitem__(self, position):
        row = int(self.rows[position])
        market = self.normalize_sequence(self.arrays['market'][row], 'market')
        transaction = self.normalize_sequence(self.arrays['tx'][row], 'tx')
        order = self.normalize_sequence(self.arrays['order'][row], 'order')
        static = np.asarray(self.static[row], dtype=np.float32)
        static = np.clip(np.nan_to_num((static - self.static_norm['mean']) / self.static_norm['std']), -8, 8)
        return market, transaction, order, torch.from_numpy(static.astype(np.float32, copy=False)), torch.tensor(row)


@torch.no_grad()
def compute_base_predictions(model, loader, device, output):
    model.eval()
    for batch in loader:
        inputs = [value.to(device, non_blocking=True).contiguous() for value in batch[:4]]
        rows = batch[4].numpy()
        prediction = model(*inputs)
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite strict-base prediction')
        output[rows] = prediction.float().cpu().numpy()


class ConditionedEventDataset(Dataset):
    """Attach the latest contemporaneous order-book state to each raw event."""

    CONTEXT_CHANNELS = np.asarray([0, 2, 4, 5, 6, 7, 8], dtype=np.int64)

    def __init__(self, rows, market, events, lengths, event_norm, base_prediction, target, target_scale):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.market = market
        self.events = events
        self.lengths = lengths
        self.event_norm = event_norm
        self.base_prediction = base_prediction
        self.target = target
        self.target_scale = float(target_scale)
        self.market_mean = np.asarray(BUNDLED_STREAM_NORM['market']['mean'], dtype=np.float32)
        self.market_std = np.asarray(BUNDLED_STREAM_NORM['market']['std'], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def condition(self, row, source, normalized_market, raw_market, market_present):
        length = int(self.lengths[source][row])
        limit = self.events[source].shape[1]
        categories = 2 if source == 'transaction' else 4
        output_dim = TX_CONTEXT_DIM if source == 'transaction' else ORDER_CONTEXT_DIM
        output = np.zeros((limit, output_dim + 1), dtype=np.float32)
        if length <= 0:
            return torch.from_numpy(output)

        raw = np.asarray(self.events[source][row, :length], dtype=np.float32)
        mean, std = self.event_norm[source]
        normalized_event = np.clip(np.nan_to_num((raw - mean) / std), -8, 8)

        event_seconds = np.clip(raw[:, -2] * 60.0, 0.0, 60.0)
        event_bins = np.clip(MARKET_LEN - 1 - np.floor(event_seconds / 3.0).astype(np.int64), 0, MARKET_LEN - 1)
        available_index = np.where(market_present, np.arange(MARKET_LEN), -1)
        latest_available = np.maximum.accumulate(available_index)
        context_index = latest_available[event_bins]
        context_valid = context_index >= 0
        safe_index = np.maximum(context_index, 0)
        context = normalized_market[safe_index][:, self.CONTEXT_CHANNELS].copy()
        context[~context_valid] = 0.0
        age = np.zeros(length, dtype=np.float32)
        age[context_valid] = np.log1p(event_bins[context_valid] - context_index[context_valid]) / np.log(float(MARKET_LEN))

        raw_mid = raw_market[safe_index, 0]
        raw_spread = np.maximum(np.abs(raw_market[safe_index, 2]), 1e-5)
        relative_prices = np.zeros((length, categories), dtype=np.float32)
        for category in range(categories):
            present_category = raw[:, category * 3] > 0
            relative = np.clip((raw[:, category * 3 + 2] - raw_mid) / raw_spread, -20, 20)
            relative_prices[:, category] = np.where(context_valid & present_category, relative, 0.0)

        conditioned = np.concatenate([
            normalized_event.astype(np.float32, copy=False),
            context.astype(np.float32, copy=False),
            age[:, None], context_valid.astype(np.float32)[:, None],
            relative_prices,
        ], axis=1)
        if conditioned.shape[1] != output_dim:
            raise AssertionError((source, conditioned.shape, output_dim))
        output[:length, :output_dim] = np.clip(np.nan_to_num(conditioned), -20, 20)
        output[:length, -1] = 1.0
        return torch.from_numpy(output)

    def __getitem__(self, position):
        row = int(self.rows[position])
        raw_market = np.asarray(self.market[row], dtype=np.float32)
        market_present = np.abs(raw_market).sum(axis=-1) > 0
        normalized_market = np.clip(np.nan_to_num((raw_market - self.market_mean) / self.market_std), -8, 8)
        normalized_market[~market_present] = 0.0
        transaction = self.condition(row, 'transaction', normalized_market, raw_market, market_present)
        order = self.condition(row, 'order', normalized_market, raw_market, market_present)
        base = np.float32(self.base_prediction[row])
        target = np.float32(self.target[row] / self.target_scale)
        return transaction, order, torch.tensor(base), torch.tensor(target), torch.tensor(row)


class EventResidualModel(nn.Module):
    def __init__(self, d_model=96, dropout=0.15):
        super().__init__()
        self.transaction = SubsecondEventEncoder(TX_CONTEXT_DIM, d_model, dropout)
        self.order = SubsecondEventEncoder(ORDER_CONTEXT_DIM, d_model, dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + 1), nn.Linear(d_model + 1, 48),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(48, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, transaction, order, base_prediction):
        event_summary = self.fusion(torch.cat([self.transaction(transaction), self.order(order)], dim=1))
        residual = self.head(torch.cat([event_summary, base_prediction[:, None]], dim=1)).squeeze(-1)
        return base_prediction + residual


@torch.no_grad()
def predict_residual(model, loader, device, target_scale, output):
    model.eval()
    for batch in loader:
        transaction, order, base_prediction = [value.to(device, non_blocking=True).contiguous() for value in batch[:3]]
        rows = batch[4].numpy()
        prediction = model(transaction, order, base_prediction)
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite event-residual prediction')
        output[rows] = prediction.float().cpu().numpy() * target_scale


def score_slices(prediction, target, months, rows):
    rows = np.asarray(rows, dtype=np.int64)
    result = {'all': cosine(prediction[rows], target[rows])}
    for name, month_mask in (
        ('selection', (months >= 62) & (months <= 65)),
        ('forward', (months >= 67) & (months <= 70)),
    ):
        chosen = rows[month_mask[rows]]
        result[name] = cosine(prediction[chosen], target[chosen])
    result['monthly'] = {str(int(month)): cosine(prediction[rows[months[rows] == month]], target[rows[months[rows] == month]]) for month in sorted(set(months[rows]))}
    return result


def main():
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select Kaggle T4 x2; exactly two CUDA devices are required')

    checkpoint_path = next(iter(Path('/kaggle/input').rglob('strict_dev_v7.pt')), None)
    static_norm_path = next(iter(Path('/kaggle/input').rglob('strict_dev_v7_static_norm.npz')), None)
    if checkpoint_path is None or static_norm_path is None:
        raise FileNotFoundError('Attach zwq1037/mscapital-strict-transformer-dev-v7')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    target_scale = float(checkpoint['target_scale'])

    ids, months, target = read_train_label()
    train_rows = np.flatnonzero(months <= 59)
    valid_rows = np.flatnonzero((months >= 62) & (months <= 70) & (months != 66))
    arrays = load_grid('train', len(ids))
    static = _STATIC_FEATURES
    static_file = np.load(static_norm_path)
    static_norm = {'mean': static_file['mean'].astype(np.float32), 'std': static_file['std'].astype(np.float32)}

    device = torch.device('cuda')
    base = StrictBaseModel()
    base.load_state_dict(canonical_state(checkpoint['model']), strict=True)
    base = nn.DataParallel(base.to(device), device_ids=[0, 1])
    base_prediction = np.full(len(ids), np.nan, dtype=np.float32)
    base_rows = np.concatenate([train_rows, valid_rows])
    base_loader = DataLoader(
        BasePredictionDataset(base_rows, arrays, static, static_norm),
        batch_size=EVENT_BATCH_SIZE, shuffle=False, num_workers=0,
    )
    compute_base_predictions(base, base_loader, device, base_prediction)
    base_score = score_slices(base_prediction * target_scale, target, months, valid_rows)
    if abs(base_score['all'] - float(checkpoint['score'])) > 2e-5:
        raise RuntimeError(f"Strict-base reproduction failed: {base_score['all']} vs {checkpoint['score']}")
    del base, base_loader
    torch.cuda.empty_cache()
    print(json.dumps({'strict_base_reproduced': base_score, 'checkpoint_score': float(checkpoint['score'])}), flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    for source in ('transaction', 'order'):
        build_source('train', source, ids, CACHE)
    events = {name: np.load(CACHE / f'{name}_events.npy', mmap_mode='r') for name in ('transaction', 'order')}
    lengths = {name: np.load(CACHE / f'{name}_lengths.npy') for name in events}
    selected_for_norm = np.sort(np.random.default_rng(SEED + 102).choice(train_rows, min(10000, len(train_rows)), replace=False))
    event_norm = {source: fit_stats(events[source], lengths[source], selected_for_norm) for source in events}

    train_dataset = ConditionedEventDataset(train_rows, arrays['market'], events, lengths, event_norm, base_prediction, target, target_scale)
    valid_dataset = ConditionedEventDataset(valid_rows, arrays['market'], events, lengths, event_norm, base_prediction, target, target_scale)
    valid_loader = DataLoader(valid_dataset, batch_size=EVENT_BATCH_SIZE, shuffle=False, num_workers=0)

    model = nn.DataParallel(EventResidualModel().to(device), device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=EVENT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EVENT_EPOCHS)
    smoke = next(iter(DataLoader(train_dataset, batch_size=EVENT_BATCH_SIZE, shuffle=False, num_workers=0)))
    model.train()
    smoke_prediction = model(*[value.to(device).contiguous() for value in smoke[:3]])
    smoke_prediction.square().mean().backward()
    allocated = [torch.cuda.max_memory_allocated(index) for index in range(2)]
    if min(allocated) < 1000000:
        raise RuntimeError('Both T4 GPUs must participate')
    model.zero_grad(set_to_none=True)
    print(json.dumps({'dual_gpu_smoke': 'passed', 'allocated_bytes': allocated}), flush=True)

    logs = []
    best_key = -1e9
    best_metrics = None
    prediction = np.full(len(ids), np.nan, dtype=np.float32)
    for epoch in range(1, EVENT_EPOCHS + 1):
        started = time.time()
        shuffled_rows = np.random.default_rng(SEED + epoch).permutation(train_rows)
        train_loader = DataLoader(
            ConditionedEventDataset(shuffled_rows, arrays['market'], events, lengths, event_norm, base_prediction, target, target_scale),
            batch_size=EVENT_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True,
        )
        model.train()
        total_loss = 0.0
        count = 0
        for batch_index, batch in enumerate(prefetched_batches(train_loader), start=1):
            transaction, order, base_values, y = [value.to(device, non_blocking=True).contiguous() for value in batch[:4]]
            optimizer.zero_grad(set_to_none=True)
            current = model(transaction, order, base_values).float()
            centered_prediction = current - current.mean()
            centered_target = y - y.mean()
            cosine_loss = 1 - (centered_prediction * centered_target).sum() / (centered_prediction.norm() * centered_target.norm()).clamp_min(1e-8)
            residual_penalty = (current - base_values).square().mean()
            loss = 0.30 * F.smooth_l1_loss(current, y) + 0.65 * cosine_loss + 0.05 * residual_penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite event-residual loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
            if batch_index % 300 == 0:
                print(f'epoch={epoch} batch={batch_index} loss={total_loss / count:.6f}', flush=True)
        scheduler.step()

        prediction[:] = np.nan
        predict_residual(model, valid_loader, device, target_scale, prediction)
        metrics = score_slices(prediction, target, months, valid_rows)
        delta = {name: metrics[name] - base_score[name] for name in ('all', 'selection', 'forward')}
        monthly_delta = {month: metrics['monthly'][month] - base_score['monthly'][month] for month in metrics['monthly']}
        key = min(delta['selection'], delta['forward']) + 0.25 * delta['all']
        record = {
            'epoch': epoch, 'loss': total_loss / max(count, 1), 'seconds': time.time() - started,
            'score': metrics, 'delta': delta, 'monthly_delta': monthly_delta, 'selection_key': key,
        }
        logs.append(record)
        print(json.dumps(record), flush=True)
        if key > best_key:
            best_key = key
            best_metrics = record
            torch.save(model.state_dict(), EVENT_RUN / 'best_event_residual.pt')
            pd.DataFrame({
                'sample_id': ids[valid_rows], 'month': months[valid_rows],
                'target': target[valid_rows], 'base_prediction': base_prediction[valid_rows] * target_scale,
                'prediction': prediction[valid_rows],
            }).to_feather(EVENT_RUN / 'validation_predictions.feather')

    forward_months = ['67', '68', '69', '70']
    improved_forward_months = sum(best_metrics['monthly_delta'][month] > 0 for month in forward_months)
    worst_forward_month = min(best_metrics['monthly_delta'][month] for month in forward_months)
    passed = bool(
        best_metrics['delta']['selection'] >= 0
        and best_metrics['delta']['forward'] >= 0.0005
        and improved_forward_months >= 2
        and worst_forward_month >= -0.001
    )
    result = {
        'experiment': 'EXP-SEQUENCE-027-MARKET-CONDITIONED-EVENT-RESIDUAL',
        'baseline': base_score, 'best': best_metrics, 'logs': logs,
        'improved_forward_months': improved_forward_months,
        'worst_forward_month_delta': worst_forward_month,
        'passed': passed,
        'gate': {'selection_delta_min': 0.0, 'forward_delta_min': 0.0005, 'forward_months_improved_min': 2, 'worst_forward_month_delta_min': -0.001},
        'formal_submission': False,
    }
    (EVENT_RUN / 'score_only.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        EVENT_RUN.mkdir(parents=True, exist_ok=True)
        (EVENT_RUN / 'failure.json').write_text(json.dumps({'exception': type(exc).__name__, 'message': str(exc)}), encoding='utf-8')
        raise
    finally:
        for name in ('transaction_events.npy', 'order_events.npy', 'transaction_lengths.npy', 'order_lengths.npy', 'sample_ids.npy'):
            path = CACHE / name
            if path.parent.resolve() != CACHE.resolve():
                raise RuntimeError('Cache cleanup path escaped its directory')
            path.unlink(missing_ok=True)
