"""Resume the saved experiment with float32 computation after GPU recovery."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'scripts/exp_sequence_020_subsecond_gru_transformer_cpu_prefetch.py'
text = source.read_text(encoding='utf-8')
text = text.replace("    if not torch.cuda.is_bf16_supported():\n        raise RuntimeError('Benchmark requires BF16 support')\n", '')
text = text.replace("'training_precision': 'bfloat16'", "'training_precision': 'float32'")
text = text.replace("with torch.autocast('cuda', dtype=torch.bfloat16):", "with torch.autocast('cuda', enabled=False):")
old = """        if recovery['config'] != config:
            raise RuntimeError('Checkpoint configuration mismatch')"""
new = """        previous_config = dict(recovery['config'])
        previous_precision = previous_config.pop('training_precision', 'float32')
        expected_config = dict(config)
        expected_config.pop('training_precision')
        if previous_config != expected_config or previous_precision not in ('bfloat16', 'float32'):
            raise RuntimeError('Checkpoint configuration mismatch')
        if previous_precision != 'float32':
            print('precision migration: saved BF16 training -> float32; model/optimizer retained', flush=True)
            (RUN / 'precision_migration.json').write_text(json.dumps({
                'from': previous_precision, 'to': 'float32',
                'epoch': recovery['epoch'], 'next_batch': recovery['next_batch'],
                'reason': 'GPU driver recovered after cuDNN internal error',
                'model_and_optimizer_preserved': True,
            }, indent=2), encoding='utf-8')"""
assert old in text
text = text.replace(old, new)
destination = ROOT / 'scripts/exp_sequence_020_subsecond_gru_transformer_fp32.py'
compile(text, str(destination), 'exec')
destination.write_text(text, encoding='utf-8')
print(destination)
