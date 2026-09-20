"""Continue the existing Joint-Transformer development fold from epoch 6 to 12."""

import exp_transformer_001_joint_dev as experiment


def fixed_low_learning_rate(epoch: int) -> float:
    del epoch
    return 1.0e-4


def main() -> None:
    experiment.EPOCHS = 12
    experiment.EVALUATION_EPOCHS = {8, 10, 12}
    experiment.learning_rate = fixed_low_learning_rate
    experiment.main()


if __name__ == "__main__":
    main()
