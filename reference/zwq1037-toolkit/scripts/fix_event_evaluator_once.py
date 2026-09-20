from pathlib import Path


path = Path(__file__).with_name("evaluate_event_residual_in_owned_softgate.py")
lines = path.read_text(encoding="utf-8").splitlines()
start = next(index for index, line in enumerate(lines) if line.strip() == "rv = pd.read_feather(")
end = next(index for index in range(start, len(lines)) if "frame = frame.merge(rv" in lines[index])
lines[start:end + 1] = [
    '    if "realized_volatility_60" not in frame.columns:',
    "        rv = pd.read_feather(",
    '            ROOT / "data/processed/train_market_microstructure_features.feather",',
    '            columns=["sample_id", "realized_volatility_60"],',
    "        )",
    '        frame = frame.merge(rv, on="sample_id", how="left", validate="one_to_one")',
]
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
