"""Keep the background input worker entirely on CPU."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/exp_sequence_020_subsecond_gru_transformer_prefetch.py').read_text(encoding='utf-8')
assert 'pin_memory=True' in source
source = source.replace('pin_memory=True', 'pin_memory=False')
destination = root / 'scripts/exp_sequence_020_subsecond_gru_transformer_cpu_prefetch.py'
destination.write_text(source, encoding='utf-8')
print(destination)
