"""Stage a seed extension using the exact source of a completed confirmation.

Does not launch a kernel. The archived trainer and its modules must match the
completed run's code hashes byte for byte; only the seed/arm selection changes.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def source_files(source, manifest):
    tree = ast.parse(Path(source).read_text(encoding='utf-8'))
    payloads = [n.args[0].value for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'b64decode' and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)]
    if len(payloads) != 1:
        raise ValueError('Expected exactly one archived source payload')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payloads[0], validate=True))) as z:
        if len(set(z.namelist())) != len(z.namelist()):
            raise ValueError('Duplicate archive names')
        files = {name: z.read(name) for name in z.namelist()}
    for name in files:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe archived path')
    for name, digest in manifest['code_hashes'].items():
        matches = [v for k, v in files.items() if Path(k).name == name]
        if len(matches) != 1 or hashlib.sha256(matches[0]).hexdigest() != digest:
            raise ValueError('Archived code mismatch: ' + name)
    return files


def stage(args):
    meta = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if meta['config']['mode'] != 'confirm' or meta['backend'] != 'xla':
        raise ValueError('Expected a completed TPU confirmation reference')
    if args.seed in meta['config']['seeds']:
        raise ValueError('Seed already present in reference; do not duplicate training')
    if not 0 < args.max_hours <= 3:
        raise ValueError('This extension is capped at three TPU hours')
    files = source_files(args.source, meta)
    config = dict(meta['config'], arms=['flow31'], seeds=[args.seed])
    files['config.json'] = json.dumps(config, indent=2).encode()
    expected = {k: meta[k] for k in ('dataset_fingerprint', 'torch_version', 'torch_xla_version')}
    files['extension_reference.json'] = json.dumps(expected).encode()
    files['run_main.py'] = f'''import json, os, pathlib, subprocess, sys
import torch, torch_xla
root = pathlib.Path(__file__).parent
expected = json.loads((root / 'extension_reference.json').read_text())
if str(torch.__version__) != expected['torch_version'] or str(torch_xla.__version__) != expected['torch_xla_version']:
    raise RuntimeError('Runtime changed relative to reference; stop before training')
hits = [p for p in pathlib.Path('/kaggle/input').rglob('manifest.json')
        if 'mscapital-prepared-v2' in str(p) and json.loads(p.read_text()).get('fingerprint') == expected['dataset_fingerprint']]
if len(hits) != 1:
    raise RuntimeError('Prepared dataset mount missing or ambiguous')
env = dict(os.environ, PYTHONPATH=str(root))
common = ['--data', str(hits[0].parent), '--out', '/kaggle/working/run']
subprocess.run([sys.executable, str(root / 'train_audited.py'), '--config', str(root / 'config.json'),
                *common, '--device', 'xla', '--max-hours', {str(args.max_hours)!r}], env=env, check=True)
subprocess.run([sys.executable, str(root / 'summarize_audited.py'), *common], env=env, check=True)
'''.encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in sorted(files.items()):
            z.writestr(name, value)
    payload = base64.b64encode(stream.getvalue()).decode()
    script = ('import base64, io, pathlib, runpy, sys, tempfile, zipfile\n'
              "if __name__ == '__main__':\n"
              "    with tempfile.TemporaryDirectory(prefix='mscapital-seed-') as d:\n"
              f'        zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(d)\n'
              '        sys.path.insert(0, d)\n'
              "        runpy.run_path(str(pathlib.Path(d) / 'run_main.py'), run_name='__main__')\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'main.py').write_text(script, encoding='utf-8')
    kernel = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    kernel.pop('id_no', None)
    kernel.update(id=args.kernel, title=args.kernel.split('/')[-1], code_file='main.py',
                  is_private=True, enable_gpu=False, enable_tpu=True,
                  enable_internet=False, kernel_sources=[], competition_sources=[], model_sources=[])
    (out / 'kernel-metadata.json').write_text(json.dumps(kernel, indent=2), encoding='utf-8')
    report = {'kernel': args.kernel, 'seed': args.seed, 'models': 10,
              'max_hours_soft': args.max_hours, 'max_hours_platform': 3,
              'reference_signature': meta['run_signature'], 'reference_code_hashes': meta['code_hashes'],
              'config': config, 'runtime': expected,
              'bundle_sha256': hashlib.sha256(script.encode()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
              'purpose': 'Complete third-seed confirmation of existing champion; no new holdout and no automatic submission.'}
    (out / 'stage_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('kernel', 'seed', 'models', 'max_hours_soft', 'bundle_sha256')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--metadata', required=True); p.add_argument('--out', required=True)
    p.add_argument('--kernel', required=True); p.add_argument('--seed', type=int, default=123)
    p.add_argument('--max-hours', type=float, default=2.75)
    stage(p.parse_args())
