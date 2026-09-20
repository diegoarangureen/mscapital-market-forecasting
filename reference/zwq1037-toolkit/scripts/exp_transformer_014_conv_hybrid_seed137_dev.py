"""Test Conv Hybrid Joint-Transformer seed137 on months 50-59."""

import exp_transformer_010_conv_hybrid_dev as experiment


def main() -> None:
    experiment.EXPERIMENT_ID = "EXP-TRANSFORMER-014-CONV-HYBRID-SEED137-DEV"
    experiment.SEED = 137
    experiment.TRAIN_END_MONTH = 49
    experiment.VALID_START_MONTH = 50
    experiment.VALID_END_MONTH = 59
    experiment.FOLD_NAME = "train049_valid5059"
    experiment.EPOCHS = 5
    experiment.EVALUATION_EPOCHS = {3, 4, 5}
    experiment.gru.FOLD_NAME = experiment.FOLD_NAME
    experiment.main()


if __name__ == "__main__":
    main()
