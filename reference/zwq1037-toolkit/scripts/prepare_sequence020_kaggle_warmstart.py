"""Prepare a small private input bundle and a portable warm-start notebook."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'
ARTIFACTS = ROOT / 'outputs/factorized_transformer_dev_v7_exact'
BUNDLE = ROOT / 'data/interim/kaggle_datasets/strict_transformer_dev_v7'
BUNDLE.mkdir(parents=True, exist_ok=True)
for original, target in (('best_transformer_cnn.pt', 'strict_dev_v7.pt'),
                         ('norm_v2.json', 'strict_dev_v7_stream_norm.json'),
                         ('reconstructed_static_normalization.npz', 'strict_dev_v7_static_norm.npz')):
    shutil.copyfile(ARTIFACTS / original, BUNDLE / target)
source_result = json.loads((ARTIFACTS / 'result.json').read_text(encoding='utf-8'))
assert abs(source_result['best_cosine'] - 0.1553845145) < 1e-8
manifest = {'source': 'own private Kaggle notebook output, numeric version7',
            'kernel': 'zwq1037/multistream-factorized-transformer-multiwindow-dev',
            'version': 7, 'train': '0-59', 'purged': [60,61],
            'validation': '62-70 excluding66', 'source_cosine': source_result['best_cosine'],
            'checkpoint_sha256': hashlib.sha256((BUNDLE / 'strict_dev_v7.pt').read_bytes()).hexdigest(),
            'contains_raw_data': False, 'contains_feature_cache': False,
            'contains_credentials': False, 'destination_private': True}
(BUNDLE / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
dataset = {'id': 'zwq1037/mscapital-strict-transformer-dev-v7',
           'title': 'MSCapital Own Strict Transformer Dev V7',
           'licenses': [{'name': 'CC0-1.0'}]}
(BUNDLE / 'dataset-metadata.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')

source = (DIRECTORY / 'run_v21_subsecond_events_dev.py').read_text(encoding='utf-8')
tree = ast.parse(source)
nodes = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == 'make_model')
         and not (isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'")]
body = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
body = body.replace("Path('/kaggle/working/subsecond_event_experiment')",
                    "Path('/kaggle/working/subsecond_event_warmstart')")
body = body.replace("'excluded_month': 66, 'cache': str(CACHE)",
                    "'excluded_month': 66, 'cache': str(CACHE), 'initialization': 'strict_dev_v7', 'teacher_sha256': '" + manifest['checkpoint_sha256'] + "'")
warm_tree = ast.parse((ROOT / 'scripts/subsecond_event_warmstart_model.py').read_text(encoding='utf-8'))
factory = next(ast.unparse(node) for node in warm_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_warmstart_model')
factory = factory.replace('model = make_model()', 'model = _JointMultiStreamStaticModel()')
assert 'ROOT /' not in factory
locator = """DEFAULT_CHECKPOINT = next(iter(Path('/kaggle/input').rglob('strict_dev_v7.pt')), None)
if DEFAULT_CHECKPOINT is None:
    raise FileNotFoundError('Attach private MSCapital Own Strict Transformer Dev V7 input')
"""
footer = """def make_model():
    model, initialization = make_warmstart_model()
    (RUN / 'initialization.json').write_text(json.dumps(initialization, indent=2), encoding='utf-8')
    return model

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (RUN / 'failure.json').write_text(json.dumps({'exception': type(exc).__name__, 'message': str(exc)}), encoding='utf-8')
        raise
    finally:
        for name in ('transaction_events.npy', 'order_events.npy', 'transaction_lengths.npy', 'order_lengths.npy', 'sample_ids.npy'):
            path = CACHE / name
            if path.parent.resolve() != CACHE.resolve():
                raise RuntimeError('Cache cleanup path escaped its directory')
            path.unlink(missing_ok=True)
"""
generated = '\n\n'.join((body, locator, factory, footer))
compile(generated, 'run_v23_subsecond_warmstart.py', 'exec')
script = DIRECTORY / 'run_v23_subsecond_warmstart.py'
script.write_text(generated, encoding='utf-8')
notebook = json.loads((DIRECTORY / 'run_v21_subsecond_events_dev.ipynb').read_text(encoding='utf-8'))
notebook['cells'][0]['source'] = generated.splitlines(keepends=True)
path = DIRECTORY / 'run_v23_subsecond_warmstart.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
report = {'bundle': str(BUNDLE), 'destination_dataset': dataset['id'],
          'input_files': {file.name: file.stat().st_size for file in BUNDLE.iterdir() if file.is_file()},
          'private_dataset': True, 'notebook': str(path), 'code_bytes': path.stat().st_size,
          'notebook_destination': 'zwq1037/multistream-factorized-transformer-multiwindow-dev',
          'compiled': True, 'uploaded_dataset': False, 'uploaded_notebook': False,
          'training_window': '0-59 / purge60-61 / valid62-70 excluding66',
          'private_weights_previously_generated_on_kaggle': True}
(ROOT / 'outputs/submission_metadata/kaggle_subsecond_warmstart_prepared_20260917.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
