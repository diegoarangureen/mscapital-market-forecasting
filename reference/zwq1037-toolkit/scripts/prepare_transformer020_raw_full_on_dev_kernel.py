"""Stage EXP-020 raw full training on the existing Transformer dev kernel."""

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_fulltrain"
KERNEL_DIR = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev"
SOURCE_SCRIPT = SOURCE_DIR / "run_v5_transformer_raw_full.py"
SOURCE_NOTEBOOK = SOURCE_DIR / "run_v5_transformer_raw_full.ipynb"
TARGET_SCRIPT = KERNEL_DIR / "run_v25_transformer020_raw_full.py"
TARGET_NOTEBOOK = KERNEL_DIR / "run_v25_transformer020_raw_full.ipynb"

compile(SOURCE_SCRIPT.read_text(encoding="utf-8"), TARGET_SCRIPT.name, "exec")
shutil.copyfile(SOURCE_SCRIPT, TARGET_SCRIPT)
shutil.copyfile(SOURCE_NOTEBOOK, TARGET_NOTEBOOK)
metadata_path = KERNEL_DIR / "kernel-metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["code_file"] = TARGET_NOTEBOOK.name
metadata["enable_gpu"] = True
metadata["enable_tpu"] = False
metadata["machine_shape"] = "NvidiaTeslaT4"
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

report = {
    "status": "staged_not_uploaded",
    "kernel": metadata["id"],
    "expected_version": 25,
    "notebook": str(TARGET_NOTEBOOK),
    "code_bytes": TARGET_NOTEBOOK.stat().st_size,
    "dual_t4_required": True,
    "formal_submission": False,
}
(ROOT / "outputs/submission_metadata/transformer020_raw_full_dev_kernel_staged.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
