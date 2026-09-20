"""Prepare an initialization-only comparison; do not start training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'scripts/exp_sequence_020_subsecond_gru_transformer_fp32.py'
text = source.read_text(encoding='utf-8')
text = text.replace('from subsecond_event_transformer_model import make_model',
                    'from subsecond_event_warmstart_model import make_warmstart_model')
old_run = "RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER'"
new_run = "RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER-WARMSTART'"
assert old_run in text
text = text.replace(old_run, new_run)
checkpoint = ROOT / 'outputs/factorized_transformer_dev_v7_exact/best_transformer_cnn.pt'
checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
text = text.replace("'excluded_month': 66, 'cache': str(CACHE)",
                    "'excluded_month': 66, 'cache': str(CACHE), 'initialization': 'strict_old_transformer_market_static_fusion_head', 'teacher_sha256': '" + checksum + "'")
factory = """
def make_model():
    model, initialization = make_warmstart_model()
    (RUN / 'initialization.json').write_text(json.dumps(initialization, indent=2), encoding='utf-8')
    return model


"""
text = text.replace('def cosine(x, y):', factory + 'def cosine(x, y):', 1)
destination = ROOT / 'scripts/exp_sequence_020_subsecond_gru_transformer_warmstart.py'
compile(text, str(destination), 'exec')
destination.write_text(text, encoding='utf-8')
report = {'experiment': 'sequence020_warmstart', 'prepared': True, 'started': False,
          'script': str(destination), 'main_change': 'reuse trained market/static/fusion/head initialization',
          'unchanged': ['seed2026', 'batch256', 'four_epochs', 'learning_rate_2e-4',
                        'AdamW_weight_decay_1e-4', '0.35SmoothL1+0.65CenteredCosine',
                        'float32', 'train0-59_purge60-61_valid62-70_ex66', 'train_fitted_normalization'],
          'source_checkpoint_sha256': checksum,
          'launch_condition': 'cold-initialized four-epoch experiment terminal and below late gain gate',
          'no_concurrent_local_training': True, 'cold_start_precision_note': 'Cold run used BF16 for first1500 batches then float32 after driver recovery; warm run uses float32 throughout',
          'early_validation_requires_teacher_trained_only0_47': True}
path = ROOT / 'outputs/submission_metadata/subsecond_warmstart_experiment_prepared_20260917.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
