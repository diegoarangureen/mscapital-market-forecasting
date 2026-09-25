"""Audited entrypoint. Explicit --config required; see TRAINING_AGENT.md.
Historical implementation is preserved under legacy/ for reproduction only.
For a standalone Kaggle kernel use bundle_audited.py.
"""
from train_audited import parser, run

if __name__ == "__main__":
    run(parser().parse_args())
