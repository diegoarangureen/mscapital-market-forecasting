"""Prepare, but do not launch, full multiwindow training after the validation gate."""
import ast
import json
from pathlib import Path

from prepare_transformer_multiwindow_v13 import ARCHITECTURE, PROJECT, KERNEL


def main():
    source = (PROJECT / "data/interim/kaggle_kernels/multistream_factorized_transformer_fulltrain/run_v7_factorized_transformer_fulltrain.py").read_text(encoding="utf-8")
    start = source.rfind("class _FactorizedStreamEncoder(nn.Module):")
    end = source.index("def _build_xs40", start)
    source = source[:start] + ARCHITECTURE + "\n" + source[end:]
    source = source.replace('        if hasattr(model, "parallel") and isinstance(model.parallel, nn.DataParallel):\n            inference_model = model.parallel.module\n', '')
    source = source.replace("factorized_transformer_fulltrain", "factorized_transformer_multiwindow10_fulltrain")
    source = source.replace("factorized_transformer379_full_e4.csv", "factorized_transformer379_multiwindow10_full_e4.csv")
    source = source.replace("factorized_transformer_full_e4.pt", "factorized_transformer_multiwindow10_full_e4.pt")
    ast.parse(source)
    output = KERNEL / "run_v14_multiwindow10_fulltrain.py"
    output.write_text(source, encoding="utf-8")
    print(json.dumps({"prepared_only": str(output), "must_pass_validation_before_launch": True}))


if __name__ == "__main__":
    main()
