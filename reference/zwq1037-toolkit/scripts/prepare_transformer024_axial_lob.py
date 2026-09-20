"""Build the controlled three-stream Axial-LOB experiment from EXP023 plumbing."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_path = root / "scripts" / "exp_transformer_023_patch_transformer_gru.py"
target_path = root / "scripts" / "exp_transformer_024_axial_lob.py"
text = source_path.read_text(encoding="utf-8")
text = text.replace(
    '"""Controlled structured-patch Transformer-GRU test against EXP020 raw."""',
    '"""Controlled three-stream Axial-LOB test against EXP020 raw."""',
)
text = text.replace(
    'RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-023-PATCH-GRU"',
    'RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-024-AXIAL-LOB"',
)
start = text.index("class PatchTransformerGRUStream")
end = text.index("\ndef open_arrays", start)
classes = r'''class AxialLOBStream(nn.Module):
    """Patch locally, then alternate temporal and feature-axis attention."""

    def __init__(
        self,
        input_dim: int,
        length: int,
        patch_size: int = 4,
        axis_dim: int = 64,
        output_dim: int = 96,
        nhead: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if length % patch_size:
            raise ValueError(
                f"length={length} must be divisible by patch_size={patch_size}"
            )
        self.input_dim = input_dim
        self.length = length
        self.patch_size = patch_size
        self.patch_count = length // patch_size
        self.patch_projection = nn.Linear(patch_size, axis_dim)
        self.time_position = nn.Parameter(
            torch.zeros(1, self.patch_count, 1, axis_dim)
        )
        self.feature_embedding = nn.Parameter(
            torch.zeros(1, 1, input_dim, axis_dim)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=axis_dim,
            nhead=nhead,
            dim_feedforward=axis_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        feature_layer = nn.TransformerEncoderLayer(
            d_model=axis_dim,
            nhead=nhead,
            dim_feedforward=axis_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=1
        )
        self.feature_encoder = nn.TransformerEncoder(
            feature_layer, num_layers=1
        )
        self.feature_attention = nn.Sequential(
            nn.LayerNorm(axis_dim), nn.Linear(axis_dim, 1)
        )
        self.to_output = nn.Sequential(
            nn.LayerNorm(axis_dim),
            nn.Linear(axis_dim, output_dim),
            nn.GELU(),
        )
        self.time_attention = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, 1)
        )
        nn.init.normal_(self.time_position, std=0.02)
        nn.init.normal_(self.feature_embedding, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, feature_count = values.shape
        if length != self.length or feature_count != self.input_dim:
            raise ValueError(
                f"Expected (*,{self.length},{self.input_dim}), got {values.shape}"
            )
        patches = values.reshape(
            batch_size,
            self.patch_count,
            self.patch_size,
            feature_count,
        ).permute(0, 1, 3, 2)
        padding_mask = patches.abs().sum(dim=(2, 3)).eq(0)
        safe_mask = padding_mask.clone()
        fully_padded = safe_mask.all(dim=1)
        if fully_padded.any():
            safe_mask[fully_padded, -1] = False

        tokens = (
            self.patch_projection(patches)
            + self.time_position
            + self.feature_embedding
        )
        time_steps = self.patch_count
        temporal = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * feature_count, time_steps, -1
        )
        temporal_mask = (
            safe_mask[:, None, :]
            .expand(batch_size, feature_count, time_steps)
            .reshape(batch_size * feature_count, time_steps)
        )
        temporal = self.temporal_encoder(
            temporal, src_key_padding_mask=temporal_mask
        )
        tokens = temporal.reshape(
            batch_size, feature_count, time_steps, -1
        ).permute(0, 2, 1, 3)

        feature_tokens = tokens.reshape(
            batch_size * time_steps, feature_count, -1
        )
        feature_tokens = self.feature_encoder(feature_tokens)
        tokens = feature_tokens.reshape(
            batch_size, time_steps, feature_count, -1
        )
        feature_weights = torch.softmax(
            self.feature_attention(tokens).squeeze(-1), dim=2
        ).unsqueeze(-1)
        time_tokens = self.to_output(
            (tokens * feature_weights).sum(dim=2)
        )
        time_logits = self.time_attention(time_tokens).squeeze(-1)
        time_logits = time_logits.masked_fill(safe_mask, -1e4)
        time_weights = torch.softmax(time_logits, dim=1).unsqueeze(-1)
        pooled = (time_tokens * time_weights).sum(dim=1)
        return pooled * (~fully_padded).to(pooled.dtype).unsqueeze(-1)


class AxialLOBModel(nn.Module):
    """Encode each stream on time and feature axes, then fuse four sources."""

    def __init__(self, d_model: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.market = AxialLOBStream(
            len(MARKET_FEATURES), MARKET_LEN, output_dim=d_model, dropout=dropout
        )
        self.transaction = AxialLOBStream(
            len(TX_FEATURES), FLOW_LEN, output_dim=d_model, dropout=dropout
        )
        self.order = AxialLOBStream(
            len(ORDER_FEATURES), FLOW_LEN, output_dim=d_model, dropout=dropout
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
        self.source_attention = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )
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
text = text[:start] + classes + text[end:]
text = text.replace(
    "model = PatchTransformerGRUModel().to(device)",
    "model = AxialLOBModel().to(device)",
)
text = text.replace(
    '"experiment": "EXP-TRANSFORMER-023-PATCH-TRANSFORMER-GRU",',
    '"experiment": "EXP-TRANSFORMER-024-AXIAL-LOB",',
)
text = text.replace(
    '"change": "four-step structured patches, per-stream Transformer, gated GRU recurrence",',
    '"change": "per-stream temporal and feature-axis attention",',
)
text = text.replace(
    '"architecture": "patch projection -> Transformer -> gated GRU per stream + static379 fusion",',
    '"architecture": "four-step local patches -> temporal axial attention -> feature axial attention -> static379 fusion",',
)
text = text.replace(
    '        "patch_sizes": {"market": 4, "transaction": 4, "order": 4},\n',
    "",
)
target_path.write_text(text, encoding="utf-8")
print(target_path)
