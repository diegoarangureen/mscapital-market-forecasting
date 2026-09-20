from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_path = root / "scripts" / "exp_transformer_020_factorized_ema_dev.py"
target_path = root / "scripts" / "exp_transformer_023_patch_transformer_gru.py"
text = source_path.read_text(encoding="utf-8")
text = text.replace(
    '"""Paired raw-versus-EMA test for the owned factorized Transformer."""',
    '"""Controlled structured-patch Transformer-GRU test against EXP020 raw."""',
)
text = text.replace(
    'RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA"',
    'RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-023-PATCH-GRU"',
)
text = text.replace(
    'model = namespace["_JointMultiStreamStaticModel"]().to(device)',
    'model = PatchTransformerGRUModel().to(device)\n    print(f"parameter_count={sum(p.numel() for p in model.parameters()):,}", flush=True)',
)
marker = "\ndef open_arrays(rows: int):\n"
classes = r'''

class PatchTransformerGRUStream(nn.Module):
    """Encode short structured temporal patches, then restore recurrence with a GRU."""

    def __init__(
        self,
        input_dim: int,
        length: int,
        patch_size: int = 4,
        d_model: int = 96,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if length % patch_size != 0:
            raise ValueError(f"length={length} must be divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.patch_count = length // patch_size
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * patch_size),
            nn.Linear(input_dim * patch_size, d_model),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.recurrent_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, feature_count = values.shape
        patches = values.reshape(
            batch_size,
            self.patch_count,
            self.patch_size * feature_count,
        )
        patch_mask = patches.abs().sum(dim=-1) == 0
        tokens = self.projection(patches) + self.position
        tokens = self.encoder(tokens, src_key_padding_mask=patch_mask)
        recurrent, _ = self.gru(tokens.masked_fill(patch_mask.unsqueeze(-1), 0.0))
        gate = self.recurrent_gate(torch.cat([tokens, recurrent], dim=-1))
        tokens = gate * recurrent + (1.0 - gate) * tokens
        logits = self.attention(tokens).squeeze(-1).masked_fill(patch_mask, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)


class PatchTransformerGRUModel(nn.Module):
    """Apply the same patch-recurrent encoder independently to three streams."""

    def __init__(self, d_model: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.market = PatchTransformerGRUStream(
            len(MARKET_FEATURES), MARKET_LEN, patch_size=4, d_model=d_model, dropout=dropout
        )
        self.transaction = PatchTransformerGRUStream(
            len(TX_FEATURES), FLOW_LEN, patch_size=4, d_model=d_model, dropout=dropout
        )
        self.order = PatchTransformerGRUStream(
            len(ORDER_FEATURES), FLOW_LEN, patch_size=4, d_model=d_model, dropout=dropout
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, d_model),
            nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.source_attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(
        self,
        market: torch.Tensor,
        transaction: torch.Tensor,
        order: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        summaries = torch.stack(
            [
                self.market(market),
                self.transaction(transaction),
                self.order(order),
                self.static_adapter(static),
            ],
            dim=1,
        )
        summaries = self.fusion(summaries + self.source_embedding)
        weights = torch.softmax(
            self.source_attention(summaries).squeeze(-1), dim=1
        ).unsqueeze(-1)
        return self.head((summaries * weights).sum(dim=1)).squeeze(-1)

'''
if marker not in text:
    raise RuntimeError("open_arrays insertion marker not found")
text = text.replace(marker, classes + marker, 1)

start = text.index('    raw_best = best["raw"]')
end = text.index('\n\nif __name__ == "__main__":', start)
replacement = r'''    raw_best = best["raw"]
    ema_best = best["ema"]
    selected_name = (
        "raw"
        if raw_best["scores"]["62_65"] >= ema_best["scores"]["62_65"]
        else "ema"
    )
    selected = best[selected_name]
    baseline_path = (
        ROOT
        / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA"
        / "train059_valid6270_ex66/validation_predictions.feather"
    )
    baseline_frame = pd.read_feather(baseline_path)
    current_ids = np.asarray(ids[valid_indices])
    current_months = np.asarray(months[valid_indices])
    truth = np.asarray(target[valid_indices], np.float64)
    for name, expected, observed in (
        ("sample_id", current_ids, baseline_frame["sample_id"].to_numpy()),
        ("month", current_months, baseline_frame["month"].to_numpy()),
        ("target", truth, baseline_frame["target"].to_numpy()),
    ):
        if not np.allclose(expected, observed, rtol=0.0, atol=1e-8):
            raise AssertionError(f"EXP020 alignment mismatch: {name}")
    baseline_prediction = baseline_frame["raw"].to_numpy(np.float64)
    baseline_scores = scores(truth, current_months, baseline_prediction)
    deltas = {
        key: selected["scores"][key] - baseline_scores[key]
        for key in ["all_ex66", "62_65", "67_70"]
    }
    monthly_deltas = {
        month: selected["scores"]["monthly"][month] - baseline_scores["monthly"][month]
        for month in baseline_scores["monthly"]
    }
    passed = bool(
        deltas["62_65"] >= 0.0005
        and deltas["67_70"] >= 0.0005
        and deltas["all_ex66"] >= 0.0007
        and min(monthly_deltas.values()) >= -0.002
    )
    pd.DataFrame(
        {
            "sample_id": current_ids,
            "month": current_months,
            "target": truth,
            "baseline_exp020_raw": baseline_prediction,
            "raw": raw_best["prediction"],
            "ema": ema_best["prediction"],
            "candidate": selected["prediction"],
        }
    ).to_feather(FOLD / "validation_predictions.feather")
    torch.save(raw_best["state"], FOLD / "raw_best_state.pt")
    torch.save(ema_best["state"], FOLD / "ema_best_state.pt")
    summary = {
        "experiment": "EXP-TRANSFORMER-023-PATCH-TRANSFORMER-GRU",
        "change": "four-step structured patches, per-stream Transformer, gated GRU recurrence",
        "architecture": "patch projection -> Transformer -> gated GRU per stream + static379 fusion",
        "patch_sizes": {"market": 4, "transaction": 4, "order": 4},
        "ema_decay": EMA_DECAY,
        "baseline": baseline_scores,
        "selected_candidate": selected_name,
        "raw_best": {k: v for k, v in raw_best.items() if k not in {"prediction", "state"}},
        "ema_best": {k: v for k, v in ema_best.items() if k not in {"prediction", "state"}},
        "deltas": deltas,
        "monthly_deltas": monthly_deltas,
        "pass_gate": "vs EXP020 raw: selection>=+0.0005; forward>=+0.0005; all>=+0.0007; worst monthly delta>=-0.002",
        "passed": passed,
        "history": history,
    }
    (RUN / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
'''
text = text[:start] + replacement + text[end:]
target_path.write_text(text, encoding="utf-8")
print(target_path)
