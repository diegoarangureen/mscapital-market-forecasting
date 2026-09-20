"""Avoid fused eval fastpaths and smoke-test real dual-GPU validation early."""
import ast
import json

from prepare_transformer_multiwindow_v13 import KERNEL

FASTPATH = '''
# 保持训练/验证走相同的常规Transformer路径，避开双卡融合推理内核。
# Use the regular Transformer path in eval as well as training.
torch.backends.mha.set_fastpath_enabled(False)
'''

SMOKE = '''    model.eval()
    smoke_batch = next(iter(valid_loader))
    smoke_inputs = [item.to(device).contiguous() for item in smoke_batch[:4]]
    with torch.no_grad():
        smoke_prediction = model(*smoke_inputs)
    torch.cuda.synchronize()
    if not torch.isfinite(smoke_prediction).all():
        raise RuntimeError("Nonfinite real-data dual-GPU startup prediction")
    print(f"startup_dual_gpu_eval_pass batch={smoke_prediction.numel()}", flush=True)
    del smoke_batch, smoke_inputs, smoke_prediction
    model.train()
'''


def fix(source):
    point = source.rfind("class _FactorizedStreamEncoder(nn.Module):")
    source = source[:point] + FASTPATH + "\n" + source[point:]
    source = source.replace("src_key_padding_mask=safe_padding)", "src_key_padding_mask=safe_padding.contiguous())")
    source = source.replace("torch.cat(summaries, dim=1)", "torch.cat(summaries, dim=1).contiguous()")
    source = source.replace("torch.cat(missing_masks, dim=1)", "torch.cat(missing_masks, dim=1).contiguous()")
    ast.parse(source)
    return source


def main():
    source = fix((KERNEL / "run_v13_multiwindow10.py").read_text(encoding="utf-8"))
    point = "    model = TransformerCnnModel().to(device)\n"
    if source.count(point) != 1:
        raise RuntimeError("Expected exactly one validation model init")
    source = source.replace(point, point + SMOKE, 1)
    ast.parse(source)
    for name in ("run_v14_multiwindow10_evalfix.py", "run.py"):
        (KERNEL / name).write_text(source, encoding="utf-8")
    notebook = {"cells": [{"id": "multiwindow-evalfix", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (KERNEL / "run.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata_path = KERNEL / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    full_path = KERNEL / "run_v14_multiwindow10_fulltrain.py"
    full = fix(full_path.read_text(encoding="utf-8"))
    full_path.write_text(full, encoding="utf-8")
    (KERNEL / "run_v15_multiwindow10_fulltrain.py").write_text(full, encoding="utf-8")
    print("Prepared fixed dev with startup dual-GPU eval and fixed fulltrain source")


if __name__ == "__main__":
    main()
