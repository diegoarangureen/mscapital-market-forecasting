"""Apply the proven regular-attention fix and early evaluation checks to V18."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
kernel = root / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'
source = (kernel / 'run_v18_predictive_pretrain_dev.py').read_text(encoding='utf-8')
position = source.rfind('class _FactorizedStreamEncoder(nn.Module):')
source = source[:position] + '''# 双卡训练与验证使用常规注意力内核。
# Use regular attention kernels in both training and evaluation.
torch.backends.mha.set_fastpath_enabled(False)

''' + source[position:]
anchor = '    for pretrain_epoch in range(1, PRETRAIN_EPOCHS + 1):'
smoke = '''    model.eval()
    smoke_batch = next(iter(pretrain_eval_loader))
    smoke_inputs = [item.to(device).contiguous() for item in smoke_batch[:3]]
    with torch.no_grad():
        smoke_prediction = model(*smoke_inputs, static=None, pretrain=True)
    torch.cuda.synchronize()
    if not torch.isfinite(smoke_prediction).all():
        raise RuntimeError("Nonfinite pretraining evaluation smoke output")
    main_smoke_batch = next(iter(valid_loader))
    main_smoke_inputs = [item.to(device).contiguous() for item in main_smoke_batch[:4]]
    with torch.no_grad():
        main_smoke_prediction = model(*main_smoke_inputs)
    torch.cuda.synchronize()
    if not torch.isfinite(main_smoke_prediction).all():
        raise RuntimeError("Nonfinite main-task evaluation smoke output")
    print("dual_gpu_pretrain_and_main_eval_smoke=passed", flush=True)
    del smoke_batch, smoke_inputs, smoke_prediction
    del main_smoke_batch, main_smoke_inputs, main_smoke_prediction

'''
assert source.count(anchor) == 1
source = source.replace(anchor, smoke + anchor)
ast.parse(source)
for name in ('run_v19_predictive_pretrain_evalfix.py', 'run.py'):
    (kernel / name).write_text(source, encoding='utf-8')
notebook = json.loads((kernel / 'run.ipynb').read_text(encoding='utf-8'))
notebook['cells'][0]['source'] = source.splitlines(keepends=True)
(kernel / 'run.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Prepared V19 regular-attention fix with pretraining/main dual-GPU eval smoke')
