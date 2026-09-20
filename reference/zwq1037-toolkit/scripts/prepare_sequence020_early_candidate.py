"""Prepare the early-window event-model half of the fixed 25% paired check."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'scripts/exp_sequence_020_subsecond_gru_transformer_fp32.py'
text = source.read_text(encoding='utf-8')
old_run = "RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER'"
new_run = "RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-021-SUBSECOND-EARLY-CANDIDATE'"
assert old_run in text
text = text.replace(old_run, new_run)
text = text.replace('train_rows = np.flatnonzero(months <= 59)',
                    'train_rows = np.flatnonzero(months <= 47)')
text = text.replace("valid_rows = np.flatnonzero((months >= 62) & (months <= 70) & (months != 66))",
                    "valid_rows = np.flatnonzero((months >= 50) & (months <= 59))")
text = text.replace("'train_end': 59, 'valid_start': 62,\n              'excluded_month': 66",
                    "'train_end': 47, 'purged_months': [48, 49], 'valid_start': 50, 'valid_end': 59,\n              'excluded_month': None, 'paired_fixed_new_weight': 0.25")
text = text.replace("'best_cosine': best, 'baseline': OLD_BASELINE", "'best_cosine': best, 'baseline': None")
text = text.replace("'delta': best-OLD_BASELINE", "'delta': None")
text = text.replace("best >= OLD_BASELINE+.001", "False")
destination = ROOT / 'scripts/exp_sequence_021_subsecond_early_candidate.py'
compile(text, str(destination), 'exec')
destination.write_text(text, encoding='utf-8')
report = {'experiment': 'EXP-SEQUENCE-021-SUBSECOND-EARLY-CANDIDATE',
          'prepared': True, 'started': False, 'script': str(destination),
          'train_months': '0-47', 'purged_months': [48, 49], 'validation_months': '50-59',
          'epochs': 4, 'seed': 2026, 'precision': 'float32', 'batch_size': 256,
          'fixed_candidate_weight_after_pair': 0.25, 'weight_search': False,
          'requires_matched_old_transformer': True,
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
(ROOT / 'outputs/submission_metadata/sequence020_early_candidate_prepared_20260917.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))


if __name__ == '__main__':
    pass
