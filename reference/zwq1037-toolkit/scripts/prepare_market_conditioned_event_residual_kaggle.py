"""Build the self-contained Kaggle notebook for the market-conditioned event residual."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'
SOURCE = KERNEL / 'run_v23_subsecond_warmstart.py'
TAIL = ROOT / 'scripts/market_conditioned_event_residual_kaggle_tail.py'
OUTPUT_SCRIPT = KERNEL / 'run_v27_market_conditioned_event_residual.py'
OUTPUT_NOTEBOOK = KERNEL / 'run_v27_market_conditioned_event_residual.ipynb'


source = SOURCE.read_text(encoding='utf-8')
tree = ast.parse(source)
tree.body = [
    node for node in tree.body
    if not (isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'")
]
base = ast.unparse(tree)
tail = TAIL.read_text(encoding='utf-8')
generated = base + '\n\n' + tail
compile(generated, OUTPUT_SCRIPT.name, 'exec')
OUTPUT_SCRIPT.write_text(generated, encoding='utf-8')

template = json.loads((KERNEL / 'run_v23_subsecond_warmstart.ipynb').read_text(encoding='utf-8'))
template['cells'][0]['source'] = generated.splitlines(keepends=True)
OUTPUT_NOTEBOOK.write_text(json.dumps(template, ensure_ascii=False, indent=1), encoding='utf-8')

metadata_path = KERNEL / 'kernel-metadata.json'
metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
metadata['code_file'] = OUTPUT_NOTEBOOK.name
datasets = list(metadata.get('dataset_sources', []))
checkpoint_dataset = 'zwq1037/mscapital-strict-transformer-dev-v7'
if checkpoint_dataset not in datasets:
    datasets.append(checkpoint_dataset)
metadata['dataset_sources'] = datasets
metadata['enable_gpu'] = True
metadata['enable_tpu'] = False
metadata['machine_shape'] = 'NvidiaTeslaT4'
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

report = {
    'experiment': 'EXP-SEQUENCE-027-MARKET-CONDITIONED-EVENT-RESIDUAL',
    'script': str(OUTPUT_SCRIPT),
    'notebook': str(OUTPUT_NOTEBOOK),
    'code_bytes': OUTPUT_NOTEBOOK.stat().st_size,
    'destination': metadata['id'],
    'checkpoint_dataset': checkpoint_dataset,
    'dual_t4_required': True,
    'formal_submission': False,
}
(ROOT / 'outputs/submission_metadata/market_conditioned_event_residual_prepared_20260919.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8'
)
print(json.dumps(report, indent=2))
