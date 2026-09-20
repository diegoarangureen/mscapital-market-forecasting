"""Test one TabM capacity increase on the promoted relative319 feature set."""

from __future__ import annotations

import exp_tabm_016_relative_scale_features as experiment


def main() -> None:
    experiment.EXPERIMENT_ID = "EXP-TABM-020-RELATIVE319-D384"
    experiment.tabm.D_BLOCK = 384
    experiment.main()


if __name__ == "__main__":
    main()
