"""Use benchmarked BF16 training while keeping validation in float32."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/exp_sequence_020_subsecond_gru_transformer_resumable.py').read_text(encoding='utf-8')
source = source.replace("os.environ.setdefault('MKL_NUM_THREADS', '2')", "os.environ.setdefault('MKL_NUM_THREADS', '2')\nos.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'")
source = source.replace("'smoke_batches': SMOKE_BATCHES,", "'smoke_batches': SMOKE_BATCHES, 'training_precision': 'bfloat16',")
old = '                p = model(*inputs)'
assert source.count(old) == 1
source = source.replace(old, "                with torch.autocast('cuda', dtype=torch.bfloat16):\n                    p = model(*inputs)\n                p = p.float()")
source = source.replace("    RUN.mkdir(parents=True, exist_ok=True)", "    if not torch.cuda.is_bf16_supported():\n        raise RuntimeError('Benchmark requires BF16 support')\n    RUN.mkdir(parents=True, exist_ok=True)")
destination = root / 'scripts/exp_sequence_020_subsecond_gru_transformer_bfloat16.py'
destination.write_text(source, encoding='utf-8')
print(destination)
