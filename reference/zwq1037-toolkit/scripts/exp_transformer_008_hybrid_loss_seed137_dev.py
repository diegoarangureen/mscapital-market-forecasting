"""Run the validated hybrid-loss Joint-Transformer with seed 137 on months 50-59."""

import exp_transformer_006_hybrid_loss_dev as experiment


def main() -> None:
    experiment.EXPERIMENT_ID = "EXP-TRANSFORMER-008-HYBRID-LOSS-SEED137-DEV"
    experiment.SEED = 137
    experiment.EPOCHS = 5
    experiment.EVALUATION_EPOCHS = {3, 4, 5}
    experiment.main()


if __name__ == "__main__":
    main()
