"""Confirm a fixed kernel-5 Conv-GRU on the later forward window."""

from __future__ import annotations

import argparse

import exp_gru_006_conv_joint_dev as experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 137), required=True)
    args = parser.parse_args()

    # 保持 kernel-5 结构，只切换到更晚的训练与验证窗口。
    # Keep kernel-5 fixed and switch to the later forward-validation window.
    experiment.EXPERIMENT_ID = (
        f"EXP-GRU-009-CONV-JOINT-SEED{args.seed}-CONFIRM"
    )
    experiment.base.SEED = args.seed
    experiment.base.TRAIN_END_MONTH = 59
    experiment.base.VALID_START_MONTH = 60
    experiment.base.VALID_END_MONTH = 70
    experiment.base.FOLD_NAME = "train059_valid6070"
    experiment.main()


if __name__ == "__main__":
    main()
