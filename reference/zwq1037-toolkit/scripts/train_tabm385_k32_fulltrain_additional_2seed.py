from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
import train_tabm385_k32_temporal3fold_3seed as experiment

RUN_NAME = "tabm385_k32_fulltrain_additional_2seed"
experiment.RUN_NAME = RUN_NAME
experiment.RUN_DIR = PROJECT_DIR / "data" / "interim" / "submissions" / RUN_NAME
experiment.OUTPUT_DIR = PROJECT_DIR / "outputs" / "submissions"
experiment.PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions" / RUN_NAME
experiment.METADATA_PATH = PROJECT_DIR / "outputs" / "submission_metadata" / f"{RUN_NAME}.json"
experiment.FOLD_ENDS = (70,)
experiment.SEEDS = (137, 2026)

if __name__ == "__main__":
    experiment.main()
