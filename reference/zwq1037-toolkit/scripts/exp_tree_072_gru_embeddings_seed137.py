"""Confirm GRU embedding features with an independently trained encoder seed."""

from __future__ import annotations

import exp_tree_071_gru_embeddings as experiment


# 保持树模型随机种子为 42，只替换 GRU 编码器，形成干净的配对确认。
# Keep the tree seed at 42 and change only the GRU encoder for a paired confirmation.
experiment.EXPERIMENT_ID = "EXP-TREE-072-GRU96-EMBEDDING-SEED137"
experiment.FOLDS = {
    "train049_valid5059": {
        "train_end_month": 49,
        "valid_start_month": 50,
        "valid_end_month": 59,
        "encoder_run": "EXP-GRU-005-JOINT-SEED137/train049_valid5059",
    },
    "train059_valid6070": {
        "train_end_month": 59,
        "valid_start_month": 60,
        "valid_end_month": 70,
        "encoder_run": "EXP-GRU-005-JOINT-SEED137/train059_valid6070",
    },
}


if __name__ == "__main__":
    experiment.main()
