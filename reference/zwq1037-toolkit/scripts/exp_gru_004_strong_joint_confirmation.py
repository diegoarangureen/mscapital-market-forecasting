"""Confirm frozen StrongGRU candidates on train 0-59, validation 60-70."""

import exp_gru_003_strong_joint as experiment


def main() -> None:
    experiment.EXPERIMENT_ID = "EXP-GRU-004-STRONG-JOINT-CONFIRM"
    experiment.TRAIN_END_MONTH = 59
    experiment.VALID_START_MONTH = 60
    experiment.VALID_END_MONTH = 70
    experiment.FOLD_NAME = "train059_valid6070"
    experiment.EPOCHS = 6
    experiment.EVALUATION_EPOCHS = {4, 6}
    experiment.main()


if __name__ == "__main__":
    main()
