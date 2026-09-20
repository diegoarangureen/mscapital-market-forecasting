"""Activate the prepared fulltrain version only after explicit validation gate."""
import argparse
import ast
import json

from prepare_transformer_multiwindow_v13 import KERNEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", action="store_true", required=True)
    parser.parse_args()
    source = (KERNEL / "run_v14_multiwindow10_fulltrain.py").read_text(encoding="utf-8")
    ast.parse(source)
    (KERNEL / "run.py").write_text(source, encoding="utf-8")
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (KERNEL / "run.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata_path = KERNEL / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # V13 updated this existing notebook's slug when its title changed.
    metadata["id"] = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Activated prepared fulltrain code in existing notebook; not pushed yet.")


if __name__ == "__main__":
    main()
