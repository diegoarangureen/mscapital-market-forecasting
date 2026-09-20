"""Run the validated TabM configuration with an independent random seed."""

import exp_tabm_001_exp053r_features as experiment


if __name__ == "__main__":
    experiment.EXPERIMENT_ID = "EXP-TABM-002-SEED2026"
    experiment.SEED = 2026
    experiment.main()
