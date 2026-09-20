"""Build a staged full-data market-conditioned event residual Kaggle notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev"
BASE_SOURCE = KERNEL / "run_v26_transformer020_raw_full.py"
EVENT_SOURCE = KERNEL / "run_v23_subsecond_warmstart.py"
DEV_TAIL = ROOT / "scripts/market_conditioned_event_residual_kaggle_tail.py"
OUTPUT_SCRIPT = KERNEL / "run_v27_market_conditioned_event_residual_full.py"
OUTPUT_NOTEBOOK = KERNEL / "run_v27_market_conditioned_event_residual_full.ipynb"

base_text = BASE_SOURCE.read_text(encoding="utf-8")
base_tree = ast.parse(base_text)
base_tree.body = [
    node
    for node in base_tree.body
    if not (
        isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    )
]
base = ast.unparse(base_tree)

event_tree = ast.parse(EVENT_SOURCE.read_text(encoding="utf-8"))
definitions = {
    node.name: ast.unparse(node)
    for node in event_tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
}
needed = [
    "SubsecondEventEncoder",
    "encode_quantities",
    "write_complete_samples",
    "build_source",
    "fit_stats",
    "prefetched_batches",
]
event_helpers = "\n\n".join(definitions[name] for name in needed)

dev_tail = DEV_TAIL.read_text(encoding="utf-8")
prefix = dev_tail[: dev_tail.index("\ndef main():")]
prefix = prefix.replace(
    "EVENT_RUN = Path('/kaggle/working/market_conditioned_event_residual')",
    "EVENT_RUN = Path('/kaggle/working/market_conditioned_event_residual_full')",
)
prefix = prefix.replace(
    "EVENT_RUN.mkdir(parents=True, exist_ok=True)",
    "EVENT_RUN.mkdir(parents=True, exist_ok=True)\nCACHE = Path('/kaggle/working/market_conditioned_event_residual_full_cache')\nCACHE.mkdir(parents=True, exist_ok=True)",
)
prefix = prefix.replace(
    "target = np.float32(self.target[row] / self.target_scale)",
    "target = np.float32(0.0 if self.target is None else self.target[row] / self.target_scale)",
)

full_main = r'''
from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _event_cache(split, source, sample_ids, directory):
    directory.mkdir(parents=True, exist_ok=True)
    build_source(split, source, sample_ids, directory)
    values = np.load(directory / f"{source}_events.npy", mmap_mode="r")
    lengths = np.load(directory / f"{source}_lengths.npy")
    return values, lengths


def _cleanup_event_cache(directory):
    if directory.parent.resolve() != CACHE.resolve():
        raise RuntimeError("Event cache cleanup escaped CACHE")
    for path in directory.glob("*"):
        if path.is_file():
            path.unlink(missing_ok=True)
    directory.rmdir()


def main():
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select Kaggle T4 x2; exactly two CUDA devices are required")

    checkpoint_path = next(
        iter(Path("/kaggle/input").rglob("factorized_transformer379_raw_full_e5.pt")),
        None,
    )
    reference_csv_path = next(
        iter(Path("/kaggle/input").rglob("standalone_factorized_transformer379_raw_full_e5.csv")),
        None,
    )
    if checkpoint_path is None or reference_csv_path is None:
        raise FileNotFoundError(
            "Attach the v26 full Transformer checkpoint dataset with its standalone CSV"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_scale = float(checkpoint["target_scale"])

    train_ids, months, target = read_train_label()
    test_ids = read_submission_ids()
    train_rows = np.arange(train_ids.size, dtype=np.int64)
    test_rows = np.arange(test_ids.size, dtype=np.int64)
    train_arrays = load_grid("train", train_ids.size)
    test_arrays = load_grid("test", test_ids.size)
    train_static = _STATIC_FEATURES
    test_static = _load_test_static(test_ids)
    static_norm = _compute_static_norm(train_static, train_rows)

    device = torch.device("cuda")
    # Kaggle's current T4/PyTorch image can hit a cublasLt misaligned-address
    # failure when eval-mode Transformer inference runs inside DataParallel.
    # v26 used one GPU for prediction and both GPUs for training; mirror that
    # stable execution path here. The residual model below still has a strict
    # two-GPU smoke test before training starts.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    base = StrictBaseModel()
    base.load_state_dict(canonical_state(checkpoint["model"]), strict=True)
    base = base.to(device)
    base_train = np.full(train_ids.size, np.nan, dtype=np.float32)
    base_test = np.full(test_ids.size, np.nan, dtype=np.float32)
    compute_base_predictions(
        base,
        DataLoader(
            BasePredictionDataset(train_rows, train_arrays, train_static, static_norm),
            batch_size=EVENT_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        device,
        base_train,
    )
    compute_base_predictions(
        base,
        DataLoader(
            BasePredictionDataset(test_rows, test_arrays, test_static, static_norm),
            batch_size=EVENT_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        device,
        base_test,
    )
    if not np.isfinite(base_train).all() or not np.isfinite(base_test).all():
        raise RuntimeError("Nonfinite full-base prediction")

    reference = pd.read_csv(reference_csv_path).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(reference["sample_id"].to_numpy(np.int64), test_ids):
        raise AssertionError("Downloaded v26 CSV IDs do not align with test template")
    reference_prediction = reference["prediction"].to_numpy(np.float64)
    reproduced_prediction = base_test.astype(np.float64) * target_scale
    reproduction_cosine = float(
        reference_prediction @ reproduced_prediction
        / (np.linalg.norm(reference_prediction) * np.linalg.norm(reproduced_prediction) + 1e-30)
    )
    reproduction_max_abs = float(np.max(np.abs(reference_prediction - reproduced_prediction)))
    if reproduction_cosine < 0.9999999 or reproduction_max_abs > 2e-5:
        raise RuntimeError(
            f"Full-base reproduction failed: cosine={reproduction_cosine}, "
            f"max_abs={reproduction_max_abs}"
        )
    print(
        json.dumps(
            {
                "full_base_reproduction_cosine": reproduction_cosine,
                "full_base_reproduction_max_abs": reproduction_max_abs,
            }
        ),
        flush=True,
    )
    del base
    torch.cuda.empty_cache()

    train_cache = CACHE / "train"
    test_cache = CACHE / "test"
    train_events = {}
    train_lengths = {}
    test_events = {}
    test_lengths = {}
    for source in ("transaction", "order"):
        train_events[source], train_lengths[source] = _event_cache(
            "train", source, train_ids, train_cache
        )
        test_events[source], test_lengths[source] = _event_cache(
            "test", source, test_ids, test_cache
        )

    selected_for_norm = np.sort(
        np.random.default_rng(SEED + 102).choice(
            train_rows, min(10000, train_rows.size), replace=False
        )
    )
    event_norm = {
        source: fit_stats(
            train_events[source], train_lengths[source], selected_for_norm
        )
        for source in ("transaction", "order")
    }

    train_dataset = ConditionedEventDataset(
        train_rows,
        train_arrays["market"],
        train_events,
        train_lengths,
        event_norm,
        base_train,
        target,
        target_scale,
    )
    test_dataset = ConditionedEventDataset(
        test_rows,
        test_arrays["market"],
        test_events,
        test_lengths,
        event_norm,
        base_test,
        None,
        target_scale,
    )

    model = nn.DataParallel(EventResidualModel().to(device), device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=EVENT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EVENT_EPOCHS
    )
    checkpoint_out = EVENT_RUN / "event_residual_training_checkpoint.pt"
    logs = []
    start_epoch = 1
    if checkpoint_out.exists():
        recovery = torch.load(checkpoint_out, map_location=device, weights_only=False)
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        logs = recovery["logs"]
        start_epoch = int(recovery["next_epoch"])
        print(f"resume_event_epoch={start_epoch}", flush=True)

    smoke = next(
        iter(
            DataLoader(
                train_dataset,
                batch_size=EVENT_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    model.train()
    smoke_prediction = model(
        *[value.to(device).contiguous() for value in smoke[:3]]
    )
    smoke_prediction.square().mean().backward()
    allocated = [torch.cuda.max_memory_allocated(index) for index in range(2)]
    if min(allocated) < 1000000:
        raise RuntimeError("Both T4 GPUs must participate")
    model.zero_grad(set_to_none=True)
    print(json.dumps({"dual_gpu_smoke": "passed", "allocated_bytes": allocated}), flush=True)

    for epoch in range(start_epoch, EVENT_EPOCHS + 1):
        started = time.time()
        shuffled_rows = np.random.default_rng(SEED + epoch).permutation(train_rows)
        loader = DataLoader(
            ConditionedEventDataset(
                shuffled_rows,
                train_arrays["market"],
                train_events,
                train_lengths,
                event_norm,
                base_train,
                target,
                target_scale,
            ),
            batch_size=EVENT_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        model.train()
        total_loss = 0.0
        count = 0
        for batch_index, batch in enumerate(prefetched_batches(loader), start=1):
            transaction, order, base_values, y = [
                value.to(device, non_blocking=True).contiguous()
                for value in batch[:4]
            ]
            optimizer.zero_grad(set_to_none=True)
            current = model(transaction, order, base_values).float()
            centered_prediction = current - current.mean()
            centered_target = y - y.mean()
            cosine_term = 1 - (centered_prediction * centered_target).sum() / (
                centered_prediction.norm() * centered_target.norm()
            ).clamp_min(1e-8)
            residual_penalty = (current - base_values).square().mean()
            loss = (
                0.30 * F.smooth_l1_loss(current, y)
                + 0.65 * cosine_term
                + 0.05 * residual_penalty
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite event-residual full loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
            if batch_index % 500 == 0:
                print(
                    f"event_epoch={epoch} batch={batch_index} "
                    f"loss={total_loss / max(count, 1):.6f}",
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": total_loss / max(count, 1),
            "seconds": time.time() - started,
        }
        logs.append(row)
        temporary = checkpoint_out.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "logs": logs,
                "next_epoch": epoch + 1,
                "target_scale": target_scale,
            },
            temporary,
        )
        temporary.replace(checkpoint_out)
        print(json.dumps(row), flush=True)

    prediction = np.full(test_ids.size, np.nan, dtype=np.float32)
    predict_residual(
        model,
        DataLoader(
            test_dataset,
            batch_size=EVENT_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        device,
        target_scale,
        prediction,
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite event-residual full test prediction")

    submission_path = EVENT_RUN / "standalone_market_conditioned_event_residual_full.csv"
    pd.DataFrame(
        {"sample_id": test_ids, "prediction": prediction}
    ).to_csv(submission_path, index=False)
    model_path = EVENT_RUN / "market_conditioned_event_residual_full.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "target_scale": target_scale,
            "epochs": EVENT_EPOCHS,
            "seed": SEED,
            "event_norm": {
                source: {"mean": values[0], "std": values[1]}
                for source, values in event_norm.items()
            },
        },
        model_path,
    )
    result = {
        "experiment": "MARKET-CONDITIONED-EVENT-RESIDUAL-FULL",
        "status": "complete",
        "train_months": "all_available_0_70",
        "train_rows": int(train_rows.size),
        "test_rows": int(test_rows.size),
        "epochs": EVENT_EPOCHS,
        "seed": SEED,
        "visible_gpu_count": int(torch.cuda.device_count()),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "full_base_reproduction_cosine": reproduction_cosine,
        "full_base_reproduction_max_abs": reproduction_max_abs,
        "dev_softgate_selection_delta": 0.0012177148245949843,
        "dev_softgate_forward_delta": 0.0012401519911363623,
        "dev_forward_months_improved": 4,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_file": submission_path.name,
        "checkpoint_file": model_path.name,
        "training_log": logs,
        "competition_submission_status": "prepared_not_submitted",
    }
    (EVENT_RUN / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (EVENT_RUN / "score_only.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "model": "market_conditioned_event_residual_full",
                "dev_softgate_selection_delta": result["dev_softgate_selection_delta"],
                "dev_softgate_forward_delta": result["dev_softgate_forward_delta"],
                "dev_forward_months_improved": result["dev_forward_months_improved"],
                "prediction_file": submission_path.name,
                "visible_gpu_count": result["visible_gpu_count"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        EVENT_RUN.mkdir(parents=True, exist_ok=True)
        (EVENT_RUN / "failure.json").write_text(
            json.dumps(
                {"exception": type(exc).__name__, "message": str(exc)}
            ),
            encoding="utf-8",
        )
        raise
    finally:
        for directory in (CACHE / "train", CACHE / "test"):
            if directory.exists():
                _cleanup_event_cache(directory)
        if CACHE.exists():
            CACHE.rmdir()
'''

generated = (
    base
    + "\n\n"
    + "from concurrent.futures import ThreadPoolExecutor\n"
    + "from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence\n\n"
    + event_helpers
    + "\n\n"
    + prefix
    + "\n\n"
    + full_main
)
compile(generated, OUTPUT_SCRIPT.name, "exec")
OUTPUT_SCRIPT.write_text(generated, encoding="utf-8")

template = json.loads(
    (KERNEL / "run_v26_transformer020_raw_full.ipynb").read_text(encoding="utf-8")
)
template["cells"][0]["source"] = generated.splitlines(keepends=True)
OUTPUT_NOTEBOOK.write_text(
    json.dumps(template, ensure_ascii=False, indent=1), encoding="utf-8"
)

metadata_path = KERNEL / "kernel-metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["code_file"] = OUTPUT_NOTEBOOK.name
checkpoint_dataset = "zwq1037/mscapital-transformer020-raw-full-v26"
datasets = list(metadata.get("dataset_sources", []))
if checkpoint_dataset not in datasets:
    datasets.append(checkpoint_dataset)
metadata["dataset_sources"] = datasets
metadata["enable_gpu"] = True
metadata["enable_tpu"] = False
metadata["machine_shape"] = "NvidiaTeslaT4"
metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)

report = {
    "status": "staged_waiting_for_v26_checkpoint_dataset",
    "experiment": "MARKET-CONDITIONED-EVENT-RESIDUAL-FULL",
    "expected_kernel_version": 27,
    "kernel": metadata["id"],
    "notebook": str(OUTPUT_NOTEBOOK),
    "code_bytes": OUTPUT_NOTEBOOK.stat().st_size,
    "checkpoint_dataset": checkpoint_dataset,
    "dual_t4_required": True,
    "formal_submission": False,
}
(
    ROOT
    / "outputs/submission_metadata/market_conditioned_event_residual_full_v27_staged.json"
).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

