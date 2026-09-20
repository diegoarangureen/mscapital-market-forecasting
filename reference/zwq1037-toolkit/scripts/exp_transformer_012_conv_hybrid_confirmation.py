"""Confirm Conv Hybrid Joint-Transformer on months 60-70."""

import exp_transformer_010_conv_hybrid_dev as experiment


def main() -> None:
    experiment.EXPERIMENT_ID = "EXP-TRANSFORMER-012-CONV-HYBRID-CONFIRM"
    experiment.TRAIN_END_MONTH = 59
    experiment.VALID_START_MONTH = 60
    experiment.VALID_END_MONTH = 70
    experiment.FOLD_NAME = "train059_valid6070"
    experiment.EPOCHS = 5
    experiment.EVALUATION_EPOCHS = {4, 5}
    experiment.gru.FOLD_NAME = experiment.FOLD_NAME
    experiment.main()


if __name__ == "__main__":
    main()
