"""Confirm Relative394 on train 0-59, validation 62-70 excluding month 66."""

import exp_transformer_017_relative394_xs_dev as experiment


def main() -> None:
    experiment.EXPERIMENT_ID = "EXP-TRANSFORMER-018-RELATIVE394-XS-CONFIRM-NO66"
    experiment.TRAIN_END_MONTH = 59
    experiment.VALID_START_MONTH = 62
    experiment.VALID_END_MONTH = 70
    experiment.EXCLUDED_VALID_MONTHS = (66,)
    experiment.FOLD_NAME = "train059_valid6070"
    experiment.EPOCHS = 5
    experiment.EVALUATION_EPOCHS = {4, 5}
    experiment.gru.FOLD_NAME = experiment.FOLD_NAME
    experiment.main()


if __name__ == "__main__":
    main()