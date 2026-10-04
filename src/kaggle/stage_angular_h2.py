"""Stage H2 using the exact archived H1 trainer and verified completed controls."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from stage_seed_confirmation import source_files


def candidate_config(reference, requested):
    control=reference['config']
    if (reference['backend']!='xla' or control['angular_target']!='noisy'
            or control['angular_aggregation']!='members' or control['mode']!='screen'
            or control['folds']!=[0,4] or control['seeds']!=[2026,42] or control['arms']!=['flow31']):
        raise ValueError('Unexpected completed control recipe')
    candidate=dict(control,angular_aggregation='mean')
    if any(candidate.get(k)!=v for k,v in requested.items()):
        raise ValueError('H2 request changes more than angular aggregation')
    return candidate


def stage(args):
    reference=json.loads(Path(args.reference).read_text())
    cfg=candidate_config(reference,json.loads(Path(args.config).read_text()))
    files=source_files(args.source,reference)
    files['h2.json']=json.dumps(cfg,indent=2).encode()
    expected={k:reference[k] for k in ('torch_version','torch_xla_version','dataset_fingerprint','run_signature')}
    files['h2_reference.json']=json.dumps(expected).encode()
    files['run_main.py']=b'''import json, os, pathlib, subprocess, sys
import torch, torch_xla
root=pathlib.Path(__file__).parent
expected=json.loads((root/'h2_reference.json').read_text())
if str(torch.__version__)!=expected['torch_version'] or str(torch_xla.__version__)!=expected['torch_xla_version']:
    raise RuntimeError('Runtime differs from matched control')
inputs=pathlib.Path('/kaggle/input')
data=[p.parent for p in inputs.rglob('manifest.json') if 'mscapital-prepared-v2' in str(p)
      and json.loads(p.read_text()).get('fingerprint')==expected['dataset_fingerprint']]
controls=[p.parent for p in inputs.rglob('run_worker_00.json')
          if 'mscapital-x34-h1-tpu' in str(p) and p.parent.name=='control'
          and json.loads(p.read_text()).get('run_signature')==expected['run_signature']]
if len(data)!=1 or len(controls)!=1:
    raise RuntimeError('Prepared data or exact completed control mount missing/ambiguous')
# Verify old control predictions BEFORE spending accelerator time.
from review_champion import labels, recovered_run
dm,arrays=labels(data[0])
control_meta,records=recovered_run(controls[0],dm,arrays)
if len(records)!=4:raise RuntimeError('Incomplete reference controls')
out=pathlib.Path('/kaggle/working')
env=dict(os.environ,PYTHONPATH=str(root))
subprocess.run([sys.executable,str(root/'train_audited.py'),'--config',str(root/'h2.json'),
                '--data',str(data[0]),'--out',str(out/'h2'),'--device','xla','--max-hours','1.75'],env=env,check=True)
done=list((out/'h2/jobs').glob('*/done.json'))
complete=len(done)==4
(out/'x34_h2_progress.json').write_text(json.dumps({'complete':complete,'completed':len(done),'expected':4,
    'reused_control_signature':expected['run_signature']},indent=2))
if complete:
    subprocess.run([sys.executable,str(root/'review_angular_screen.py'),'--labels',str(data[0]),
                    '--control',str(controls[0]),'--candidate',str(out/'h2'),'--difference','angular_aggregation',
                    '--out',str(out/'x34_h2_review.json')],env=env,check=True)
else:print('Incomplete H2 at budget stop: no comparison or automatic restart',flush=True)
'''
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
    payload=base64.b64encode(stream.getvalue()).decode()
    source=('import base64,io,pathlib,runpy,sys,tempfile,zipfile\n'
            "if __name__=='__main__':\n"
            "    with tempfile.TemporaryDirectory(prefix='mscapital-h2-') as d:\n"
            f'        zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(d)\n'
            '        sys.path.insert(0,d)\n'
            "        runpy.run_path(str(pathlib.Path(d)/'run_main.py'),run_name='__main__')\n")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'main.py').write_text(source,encoding='utf-8')
    meta=json.loads(Path(args.metadata).read_text());meta.pop('id_no',None)
    meta.update(id=args.kernel,title=args.kernel.split('/')[-1],code_file='main.py',is_private=True,
                enable_gpu=False,enable_tpu=True,enable_internet=False,competition_sources=[],model_sources=[],
                kernel_sources=['diegoaranguren/mscapital-x34-h1-tpu'])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report={'kernel':args.kernel,'new_models':4,'reused_models':4,'soft_hours':1.75,'platform_timeout_seconds':7200,
            'config':cfg,'reference_signature':reference['run_signature'],'reference_code_hashes':reference['code_hashes'],
            'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
            'bundle_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'note':'Exact archived training code; only angular_aggregation changes. H1 not combined. No automatic submission.'}
    (out/'stage_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('kernel','new_models','reused_models','soft_hours','bundle_sha256')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','reference','config','metadata','out','kernel'):p.add_argument('--'+key,required=True)
    stage(p.parse_args())
