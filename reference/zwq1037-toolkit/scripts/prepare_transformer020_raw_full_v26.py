import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
full_dir = root / "data/interim/kaggle_kernels/multistream_factorized_transformer_fulltrain"
kernel_dir = root / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev"
source_script = full_dir / "run_v5_transformer_raw_full.py"
text = source_script.read_text(encoding="utf-8")

old_loader = '    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)'
new_loader = '    generator = torch.Generator().manual_seed(SEED + 17)\n    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)'
count = text.count(old_loader)
if count != 1:
    raise RuntimeError(f"Expected exactly one unfixed effective loader, found {count}")
text = text.replace(old_loader, new_loader, 1)
if text.count('generator = torch.Generator().manual_seed(SEED + 17)') != 2:
    raise RuntimeError("Both historical and effective entry points must define a generator")
compile(text, source_script.name, "exec")
source_script.write_text(text, encoding="utf-8")
source_notebook = source_script.with_suffix(".ipynb")
notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
notebook["cells"][0]["source"] = text.splitlines(keepends=True)
source_notebook.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

target_script = kernel_dir / "run_v26_transformer020_raw_full.py"
target_notebook = kernel_dir / "run_v26_transformer020_raw_full.ipynb"
target_script.write_text(text, encoding="utf-8")
target_nb = json.loads(source_notebook.read_text(encoding="utf-8"))
target_nb["cells"][0]["source"] = text.splitlines(keepends=True)
target_notebook.write_text(json.dumps(target_nb, ensure_ascii=False, indent=1), encoding="utf-8")

metadata_path = kernel_dir / "kernel-metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["code_file"] = target_notebook.name
metadata["enable_gpu"] = True
metadata["enable_tpu"] = False
metadata["machine_shape"] = "NvidiaTeslaT4"
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

wait25 = (root / "scripts/wait_transformer020_raw_full_v25.py").read_text(encoding="utf-8")
wait26 = wait25.replace("VERSION = 25", "VERSION = 26")
wait26 = wait26.replace("transformer020_raw_full_v25", "transformer020_raw_full_v26")
wait26 = wait26.replace("raw full v25 failed", "raw full v26 failed")
wait26 = wait26.replace(
    r'--file-pattern", r"score_only\.json$|result\.json$|standalone_factorized_transformer379_raw_full_e5\.csv$",',
    r'--file-pattern", r"score_only\.json$|result\.json$|standalone_factorized_transformer379_raw_full_e5\.csv$|factorized_transformer379_raw_full_e5\.pt$",',
)
old_check = '''    csv_files = list(OUT.rglob("standalone_factorized_transformer379_raw_full_e5.csv"))
    results = list(OUT.rglob("result.json"))
    if len(csv_files) != 1 or len(results) != 1:
        raise AssertionError(f"Expected one prediction and result file, got {len(csv_files)} and {len(results)}")'''
new_check = '''    csv_files = list(OUT.rglob("standalone_factorized_transformer379_raw_full_e5.csv"))
    results = list(OUT.rglob("result.json"))
    checkpoints = list(OUT.rglob("factorized_transformer379_raw_full_e5.pt"))
    if len(csv_files) != 1 or len(results) != 1 or len(checkpoints) != 1:
        raise AssertionError(
            f"Expected one prediction, result, and checkpoint file, got "
            f"{len(csv_files)}, {len(results)}, and {len(checkpoints)}"
        )'''
if old_check not in wait26:
    raise RuntimeError("Monitor verification marker missing")
wait26 = wait26.replace(old_check, new_check, 1)
wait26 = wait26.replace(
    'standalone_csv=str(csv_files[0]),\n        blend_candidates=built,',
    'standalone_csv=str(csv_files[0]),\n        full_checkpoint=str(checkpoints[0]),\n        blend_candidates=built,',
    1,
)
compile(wait26, "wait_transformer020_raw_full_v26.py", "exec")
(root / "scripts/wait_transformer020_raw_full_v26.py").write_text(wait26, encoding="utf-8")

launch = """$ErrorActionPreference = 'Stop'\n$env:OMP_NUM_THREADS = '2'\n$env:MKL_NUM_THREADS = '2'\n$env:OPENBLAS_NUM_THREADS = '2'\n& 'D:\\anaconda\\envs\\pytorch\\python.exe' -u \"$PSScriptRoot\\wait_transformer020_raw_full_v26.py\"\nexit $LASTEXITCODE\n"""
(root / "scripts/launch_wait_transformer020_raw_full_v26.ps1").write_text(launch, encoding="utf-8")

report = {
    "status": "staged_not_uploaded",
    "kernel": metadata["id"],
    "expected_version": 26,
    "notebook": str(target_notebook),
    "code_bytes": target_notebook.stat().st_size,
    "effective_loader_generators": text.count("generator = torch.Generator().manual_seed(SEED + 17)"),
    "downloads_full_checkpoint": True,
    "dual_t4_required": True,
    "formal_submission": False,
}
(root / "outputs/submission_metadata/transformer020_raw_full_v26_staged.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
