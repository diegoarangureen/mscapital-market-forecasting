"""Run the notebook-inspired feature builder with its canonical output order."""

from __future__ import annotations

import build_notebook_inspired_features as implementation


implementation.MARKET_DECAY_FEATURE_COLUMNS = [
    name
    for signal in implementation.DECAY_SIGNALS
    for name in (
        f"{signal}_ewm_30",
        f"{signal}_ewm_120",
        f"{signal}_ewm_30_minus_120",
    )
]


if __name__ == "__main__":
    implementation.main()
