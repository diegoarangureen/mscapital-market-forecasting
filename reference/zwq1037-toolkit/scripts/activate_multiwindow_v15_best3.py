"""Use best validation epoch3 while preserving its four-epoch LR schedule."""
import ast
import json
from prepare_transformer_multiwindow_v13 import KERNEL

path = KERNEL / 'run_v14_multiwindow10_fulltrain.py'
source = path.read_text(encoding='utf-8')
old = 'EPOCHS = int(os.environ.get("EPOCHS", "4"))'
assert source.count(old) == 1
source = source.replace(old, 'EPOCHS = int(os.environ.get("EPOCHS", "3"))')
source = source.replace('T_max=max(EPOCHS, 1)', 'T_max=4')
source = source.replace('multiwindow10_full_e4', 'multiwindow10_full_e3')
source = source.replace('"epochs": EPOCHS, "seed": SEED,', '"epochs": EPOCHS, "schedule_epochs": 4, "seed": SEED,')
source = source.replace('"competition_submission_status": "not_submitted_by_user_request"',
                        '"competition_submission_status": "prediction_only_pending_selection"')
old_write = '''    for name in ("result.json", "score_only.json"):
        (WORK_DIR / name).write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")'''
new_write = '''    (WORK_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")
    score_only = {"status": "complete", "validation_single_cosine": 0.15375126559116598,
                  "validation_owned_proxy_half_replacement_delta": 0.0006437405555845266,
                  "trained_epochs": EPOCHS, "lr_schedule_epochs": 4}
    (WORK_DIR / "score_only.json").write_text(json.dumps(score_only, indent=2), encoding="utf-8")'''
assert old_write in source
source = source.replace(old_write, new_write)
ast.parse(source)
(KERNEL / 'run_v15_multiwindow10_fulltrain.py').write_text(source, encoding='utf-8')
(KERNEL / 'run.py').write_text(source, encoding='utf-8')
notebook = {'cells': [{'id': 'multiwindow-full-best3', 'cell_type': 'code', 'execution_count': None,
    'metadata': {}, 'outputs': [], 'source': source.splitlines(keepends=True)}],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 5}
(KERNEL / 'run.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
metadata_path = KERNEL / 'kernel-metadata.json'
metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
metadata['id'] = 'zwq1037/multistream-factorized-transformer-multiwindow-dev'
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print('Activated full training: 3epochs, same cosine T_max4, dual-GPU startup checks, input caches')
