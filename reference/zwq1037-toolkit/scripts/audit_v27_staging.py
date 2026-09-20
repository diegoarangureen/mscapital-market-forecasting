import ast
import json
from pathlib import Path


ROOT = Path(r"F:\深度学习\projects\mscapital_market_forecasting")
SCRIPT = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v27_market_conditioned_event_residual_full.py"
NOTEBOOK = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v27_market_conditioned_event_residual_full.ipynb"
METADATA = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/kernel-metadata.json"

required_markers = [
    "factorized_transformer379_raw_full_e5.pt",
    "standalone_factorized_transformer379_raw_full_e5.csv",
    "torch.cuda.device_count() != 2",
    "full_base_reproduction_cosine",
    "standalone_market_conditioned_event_residual_full.csv",
    "_cleanup_event_cache",
]

source = SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source)
main_defs = [node.lineno for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
guards = []
for node in tree.body:
    if not isinstance(node, ast.If):
        continue
    try:
        test = ast.unparse(node.test)
    except Exception:
        test = ""
    if "__name__" in test and "__main__" in test:
        guards.append(node.lineno)

metadata = json.loads(METADATA.read_text(encoding="utf-8"))
notebook_json = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
notebook_source = "\n".join(
    "".join(cell.get("source", []))
    for cell in notebook_json.get("cells", [])
    if cell.get("cell_type") == "code"
)

report = {
    "script_exists": SCRIPT.exists(),
    "notebook_exists": NOTEBOOK.exists(),
    "main_def_lines": main_defs,
    "guard_lines": guards,
    "guard_after_final_main": bool(main_defs and guards and guards[-1] > main_defs[-1]),
    "exactly_one_guard": len(guards) == 1,
    "script_markers": {marker: marker in source for marker in required_markers},
    "notebook_markers": {marker: marker in notebook_source for marker in required_markers},
    "metadata_code_file": metadata.get("code_file"),
    "metadata_code_file_ok": metadata.get("code_file") == NOTEBOOK.name,
    "metadata_dataset_ok": "zwq1037/mscapital-transformer020-raw-full-v26" in metadata.get("dataset_sources", []),
}
report["passed"] = all(
    [
        report["guard_after_final_main"],
        report["exactly_one_guard"],
        all(report["script_markers"].values()),
        all(report["notebook_markers"].values()),
        report["metadata_code_file_ok"],
        report["metadata_dataset_ok"],
    ]
)
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["passed"] else 1)
