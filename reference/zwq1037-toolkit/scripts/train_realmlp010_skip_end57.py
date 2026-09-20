"""Produce one end57 numerical-skip member after its strict validation gate."""
import argparse
import json
from pathlib import Path

import train_realmlp379_rq_temporal3fold as training
from exp_realmlp_010_numerical_skip import NumericalSkipRealMLPRQ

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", action="store_true", required=True)
    parser.parse_args()
    name = "realmlp379_numerical_skip_end57_e8"
    training.reference.RealMLPRQ = NumericalSkipRealMLPRQ
    training.RUN_NAME = name
    training.RUN_DIR = PROJECT / "data/interim/submissions" / name
    training.MODEL_DIR = PROJECT / "outputs/models" / name
    training.PRED_DIR = PROJECT / "outputs/predictions" / name
    training.META_PATH = PROJECT / "outputs/submission_metadata" / f"{name}.json"
    training.FOLD_ENDS = (57,)
    training.EPOCHS = 8
    training.main()


if __name__ == "__main__":
    main()
