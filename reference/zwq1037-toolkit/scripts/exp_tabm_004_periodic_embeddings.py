"""Run official TabM with lightweight periodic numerical embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from rtdl_num_embeddings import PeriodicEmbeddings
from tabm import TabM

import exp_tabm_001_exp053r_features as experiment


EXPERIMENT_ID = "EXP-TABM-004-PERIODIC"
D_EMBEDDING = 8
N_FREQUENCIES = 16
FREQUENCY_INIT_SCALE = 0.01


def make_periodic_model(input_dimension: int, device: torch.device) -> TabM:
    embeddings = PeriodicEmbeddings(
        n_features=input_dimension,
        d_embedding=D_EMBEDDING,
        n_frequencies=N_FREQUENCIES,
        frequency_init_scale=FREQUENCY_INIT_SCALE,
        activation=True,
        lite=True,
    )
    model = TabM.make(
        n_num_features=input_dimension,
        cat_cardinalities=None,
        d_out=1,
        num_embeddings=embeddings,
        k=experiment.K,
        n_blocks=experiment.N_BLOCKS,
        d_block=experiment.D_BLOCK,
        dropout=experiment.DROPOUT,
        arch_type="tabm",
    )
    return model.to(device)


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.SEED = 42
    experiment.BATCH_SIZE = 1024
    experiment.EVAL_BATCH_SIZE = 2048
    experiment.make_model = make_periodic_model
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
        / "config.json"
    )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    metadata["parameters"].update(
        {
            "num_embeddings": "PeriodicEmbeddings",
            "d_embedding": D_EMBEDDING,
            "n_frequencies": N_FREQUENCIES,
            "frequency_init_scale": FREQUENCY_INIT_SCALE,
            "embedding_activation": True,
            "embedding_lite": True,
        }
    )
    metadata["main_change"] = (
        "add lightweight periodic numerical embeddings to all standardized inputs"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
