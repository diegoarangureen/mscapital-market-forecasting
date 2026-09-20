"""Prepare the full-data raw epoch-5 Transformer that won EXP-020 offline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_fulltrain"
SOURCE = KERNEL / "run_v4_transformer_ema_full.py"
OUTPUT = KERNEL / "run_v5_transformer_raw_full.py"
NOTEBOOK = OUTPUT.with_suffix(".ipynb")

text = SOURCE.read_text(encoding="utf-8")
swap = '''    with torch.no_grad():\n        for name, parameter in model.named_parameters():\n            parameter.copy_(ema_parameters[name])\n'''
if swap not in text:
    raise RuntimeError("EMA swap block not found")
text = text.replace(swap, "")
text = text.replace("transformer_ema_full", "transformer_raw_full")
text = text.replace("factorized_transformer379_ema0999_full_e5.pt", "factorized_transformer379_raw_full_e5.pt")
text = text.replace("standalone_factorized_transformer379_ema0999_full_e5.csv", "standalone_factorized_transformer379_raw_full_e5.csv")
text = text.replace('"experiment": "factorized_transformer379_ema0999_full"', '"experiment": "factorized_transformer379_raw_full"')
text = text.replace('"competition_submission_status": "prepared_not_submitted"', '"competition_submission_status": "prepared_not_submitted", "selected_from": "EXP-TRANSFORMER-020 raw epoch5"')
text = text.replace('"offline_all_ex66": 0.1538447396, "offline_delta": -0.0072022482', '"offline_all_ex66": 0.1610469878, "offline_delta_vs_v7": 0.0056624733')
compile(text, OUTPUT.name, "exec")
OUTPUT.write_text(text, encoding="utf-8")

template = json.loads((KERNEL / "run_v4_transformer_ema_full.ipynb").read_text(encoding="utf-8"))
template["cells"][0]["source"] = text.splitlines(keepends=True)
NOTEBOOK.write_text(json.dumps(template, ensure_ascii=False, indent=1), encoding="utf-8")

report = {
    "experiment": "factorized_transformer379_raw_full",
    "script": str(OUTPUT),
    "notebook": str(NOTEBOOK),
    "epochs": 5,
    "offline_all_ex66": 0.1610469878,
    "softgate_replacement_fraction": 0.5,
    "softgate_selection_delta": 0.0010478649,
    "softgate_forward_delta": 0.0019439085,
    "softgate_forward_months_improved": 3,
    "formal_submission": False,
}
(ROOT / "outputs/submission_metadata/transformer020_raw_full_prepared_20260919.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
