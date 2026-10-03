"""Package bounded X34 H1 training and its comparison inside a private TPU kernel."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def stage(args):
    root=Path(__file__).parent
    configs={name:json.loads(Path(path).read_text()) for name,path in
             [('control',args.control),('h1',args.candidate)]}
    differences={k:[configs['control'].get(k),configs['h1'].get(k)]
                 for k in set(configs['control'])|set(configs['h1'])
                 if configs['control'].get(k)!=configs['h1'].get(k)}
    if differences!={'angular_target':['noisy','clean']}:
        raise ValueError('H1 must change only the angular target')
    for cfg in configs.values():
        if cfg['mode']!='screen' or cfg['seeds']!=[2026,42] or cfg['folds']!=[0,4] or cfg['arms']!=['flow31']:
            raise ValueError('Unexpected screen dimensions')
    reference=json.loads(Path(args.reference).read_text())
    runtime={k:reference[k] for k in ('torch_version','torch_xla_version','dataset_fingerprint')}
    files={f'audit/{p.name}':p.read_bytes() for p in (root/'audit').glob('*.py')}
    for name in ('train_audited.py','summarize_audited.py','review_champion.py','review_angular_screen.py'):
        files[name]=(root/name).read_bytes()
    for name,cfg in configs.items():files[name+'.json']=json.dumps(cfg,indent=2).encode()
    files['runtime.json']=json.dumps(runtime).encode()
    files['run_main.py']=b'''import json, os, pathlib, subprocess, sys, time
import torch, torch_xla
root=pathlib.Path(__file__).parent
expected=json.loads((root/'runtime.json').read_text())
if str(torch.__version__)!=expected['torch_version'] or str(torch_xla.__version__)!=expected['torch_xla_version']:
    raise RuntimeError('Pinned runtime changed; stop before training')
hits=[p for p in pathlib.Path('/kaggle/input').rglob('manifest.json')
      if 'mscapital-prepared-v2' in str(p) and json.loads(p.read_text()).get('fingerprint')==expected['dataset_fingerprint']]
if len(hits)!=1:raise RuntimeError('Prepared dataset missing or ambiguous')
data=hits[0].parent
env=dict(os.environ,PYTHONPATH=str(root))
out=pathlib.Path('/kaggle/working')
deadline=time.monotonic()+3.2*3600
planned=[(f,s,a) for f in (0,4) for s in (2026,42) for a in ('control','h1')]
completed=[]
for fold,seed,arm in planned:
    remaining=(deadline-time.monotonic())/3600
    if remaining<=0:break
    key=f'flow31_o70_f{fold}_s{seed}'
    folder=out/arm
    command=[sys.executable,str(root/'train_audited.py'),'--config',str(root/(arm+'.json')),
             '--data',str(data),'--out',str(folder),'--device','xla','--job-key',key,
             '--max-hours',str(remaining)]
    subprocess.run(command,env=env,check=True)
    done=folder/'jobs'/key/'done.json'
    if not done.exists():break
    completed.append({'arm':arm,'job':key})
    (out/'x34_progress.json').write_text(json.dumps({'complete':False,'models':completed,'expected':8},indent=2))
complete=len(completed)==8
(out/'x34_progress.json').write_text(json.dumps({'complete':complete,'models':completed,'expected':8},indent=2))
if complete:
    subprocess.run([sys.executable,str(root/'review_angular_screen.py'),'--labels',str(data),
                    '--control',str(out/'control'),'--candidate',str(out/'h1'),
                    '--difference','angular_target','--out',str(out/'x34_review.json')],env=env,check=True)
else:
    print('BUDGET STOP: incomplete screen; no promotion or automatic restart',flush=True)
'''
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(files.items()):z.writestr(name,value)
    payload=base64.b64encode(stream.getvalue()).decode()
    source=('import base64,io,pathlib,runpy,sys,tempfile,zipfile\n'
            "if __name__=='__main__':\n"
            "    with tempfile.TemporaryDirectory(prefix='mscapital-x34-') as d:\n"
            f'        zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(d)\n'
            '        sys.path.insert(0,d)\n'
            "        runpy.run_path(str(pathlib.Path(d)/'run_main.py'),run_name='__main__')\n")
    for name,value in files.items():
        if name.endswith('.py'):compile(value,name,'exec')
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=False)
    (folder/'main.py').write_text(source,encoding='utf-8')
    meta=json.loads(Path(args.metadata).read_text())
    meta.pop('id_no',None)
    meta.update(id=args.kernel,title=args.kernel.split('/')[-1],code_file='main.py',is_private=True,
                enable_gpu=False,enable_tpu=True,enable_internet=False,kernel_sources=[],competition_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report={'kernel':args.kernel,'models':8,'soft_hours':3.2,'platform_timeout_seconds':12600,
            'configs':configs,'runtime':runtime,'bundle_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'source_hashes':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
            'execution':'Interleaved fold/seed matched control and H1; no relaunch, H2 or submission.'}
    (folder/'stage_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('kernel','models','soft_hours','bundle_sha256')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control','candidate','reference','metadata','out','kernel'):p.add_argument('--'+key,required=True)
    stage(p.parse_args())
