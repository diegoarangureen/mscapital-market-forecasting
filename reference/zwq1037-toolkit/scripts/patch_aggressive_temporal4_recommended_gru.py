from pathlib import Path

path = Path(r"F:\深度学习\projects\mscapital_market_forecasting\scripts\build_aggressive_temporal4_upgrade.py")
source = path.read_text(encoding="utf-8")
old = '''def main() -> None:
    missing = [str(path) for path in [*PATHS.values(), *OLD_PATHS.values()] if not path.exists()]
'''
new = '''def main() -> None:
    gru_metadata_path = (
        PROJECT
        / "outputs/submission_metadata/standalone_timeaware_three_stream_gru379_temporal3fold.json"
    )
    if not gru_metadata_path.exists():
        raise FileNotFoundError(f"Missing GRU selection metadata: {gru_metadata_path}")
    gru_metadata = json.loads(gru_metadata_path.read_text(encoding="utf-8"))
    PATHS["gru_temporal3"] = Path(gru_metadata["recommended_output_path"])
    missing = [str(path) for path in [*PATHS.values(), *OLD_PATHS.values()] if not path.exists()]
'''
if source.count(old) != 1:
    raise RuntimeError(f"Expected one main insertion point, found {source.count(old)}")
path.write_text(source.replace(old, new, 1), encoding="utf-8")
print(path)
