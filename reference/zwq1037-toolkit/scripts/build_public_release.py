"""Build a compact, redistributable Kaggle release of this project.

The release intentionally excludes competition data, downloaded third-party
notebooks, checkpoints, cached features, predictions, and submission files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/public_release/mscapital-financial-market-forecasting-toolkit"

TEXT_EXTENSIONS = {
    ".py",
    ".ps1",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
}


def copy_tree(source: Path, destination: Path) -> int:
    """Copy source files that are suitable for a public code archive."""
    copied = 0
    if not source.exists():
        return copied
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def main() -> None:
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)

    counts: dict[str, int] = {}
    for name in ("scripts", "src", "configs", "docs", "experiments", "tests"):
        counts[name] = copy_tree(ROOT / name, RELEASE / name)

    reports_source = ROOT / "outputs/submission_metadata"
    counts["reports"] = copy_tree(reports_source, RELEASE / "reports")

    shutil.copy2(ROOT / "requirements.txt", RELEASE / "requirements.txt")
    shutil.copy2(
        reports_source / "PROJECT_CLOSEOUT_20260920.md",
        RELEASE / "PROJECT_CLOSEOUT.md",
    )

    readme = """# MSCapital Financial Market Forecasting Toolkit

This is the public code and experiment archive behind a **0.152 public
leaderboard** solution for the Kaggle competition
[MSCapital – Real Financial Market Forecasting](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/overview).

## What is included

- Feature engineering for market, order-flow, transaction-flow, path,
  volatility, spectral, and cross-stream signals.
- Time-aware validation with train months 0–59, purge months 60–61, and
  validation months 62–70 while excluding anomalous month 66.
- TabM, RealMLP, GRU, factorized Transformer, tree-model, TSMixer, TCN, and
  hybrid experiments.
- EMA, temporal folds, loss experiments, model screening, and constrained
  blend searches.
- Experiment reports and the final project closeout record.

## What is intentionally excluded

Competition data, downloaded third-party notebooks, trained checkpoints,
cached features, prediction files, and submissions are not redistributed.
Obtain the competition data from Kaggle and follow its rules. Some experiment
scripts retain the original directory assumptions and are best treated as a
research archive; start with `docs/`, `configs/`, and the most recent scripts.

## Final validation protocol

1. Train on months 0–59.
2. Purge months 60–61.
3. Validate on months 62–70, excluding month 66.
4. Select on months 62–65 and confirm forward stability on months 67–70.
5. Require improvement in the selection, forward, and combined windows before
   full training or submission.

## Main finding

The strongest final blend combined public TabM/YangQ predictions with owned
TabM, RealMLP, GRU, and a Transformer slot. The Transformer slot mixed a raw
factorized Transformer (40%) with a market-conditioned event-residual
Transformer (60%). Further blend tuning did not improve the displayed 0.152
score; the limiting factor was new single-model signal rather than blend
weights.

See `PROJECT_CLOSEOUT.md` for exact weights, accepted evidence, rejected
directions, and reopening criteria.

## 中文说明

这是该比赛项目的精简开源归档，包含我们自己编写的特征、模型、验证、
EMA、时间折与融合搜索代码。比赛数据、他人 notebook、权重、缓存和预测
文件没有再分发。最终公开榜成绩为 **0.152**，详细实验结论见
`PROJECT_CLOSEOUT.md`。

## License

Code is released under the MIT License. Documentation and experiment reports
are released under CC BY 4.0. Third-party dependencies keep their own licenses.
"""
    (RELEASE / "README.md").write_text(readme, encoding="utf-8")

    license_text = """MIT License

Copyright (c) 2026 zwq1037

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    (RELEASE / "LICENSE").write_text(license_text, encoding="utf-8")

    metadata = {
        "title": "MSCapital Financial Market Forecasting Toolkit",
        "id": "zwq1037/mscapital-financial-market-forecasting-toolkit",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    (RELEASE / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    manifest = {
        "release": metadata["id"],
        "public_leaderboard_score": 0.152,
        "excluded": [
            "competition data",
            "third-party notebooks",
            "model checkpoints",
            "feature caches",
            "predictions and submissions",
        ],
        "file_counts": counts,
    }
    (RELEASE / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    size = sum(path.stat().st_size for path in RELEASE.rglob("*") if path.is_file())
    print(json.dumps({**manifest, "size_mb": round(size / 1024**2, 2)}, indent=2))


if __name__ == "__main__":
    main()
