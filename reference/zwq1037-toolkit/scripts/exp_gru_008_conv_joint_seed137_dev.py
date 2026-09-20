"""Confirm kernel-5 Conv-GRU on development months with paired seed 137."""

from __future__ import annotations

import exp_gru_006_conv_joint_dev as experiment


EXPERIMENT_ID = "EXP-GRU-008-CONV-JOINT-SEED137-DEV"


def main() -> None:
    # 只改变随机种子，保留 Conv-GRU 的结构与训练设置。
    # Change only the seed; keep the Conv-GRU architecture and training setup.
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.base.SEED = 137
    experiment.main()


if __name__ == "__main__":
    main()
