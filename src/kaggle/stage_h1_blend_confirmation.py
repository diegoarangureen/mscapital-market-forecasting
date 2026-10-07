import json,pathlib,base64,io,zipfile,hashlib
repo=pathlib.Path('/tmp/repo');root=repo/'src/kaggle'
cfg=json.loads((repo/'configs/audited/x34_control.json').read_text())
cfg.update(mode='confirm',folds=[0,1,2,3,4],origins=[{'origin':59,'score_start':62,'score_end':66},{'origin':64,'score_start':67,'score_end':70}],frozen_from='2026-10-07 fixed 50:50 flow31+H1 blend confirmation; paired full temporal ensemble, no weight search')
files={f'audit/{p.name}':p.read_bytes() for p in (root/'audit').glob('*.py')}
for name in ['train_audited.py','summarize_audited.py','compare_audited.py']:files[name]=(root/name).read_bytes()
for arm in ['control','h1']:
 c=dict(cfg,angular_target='clean' if arm=='h1' else 'noisy')
 p=repo/f'configs/audited/confirm_h1_{arm}.json';p.write_text(json.dumps(c,indent=2)+'\n');files[arm+'.json']=p.read_bytes()
runtime=json.loads((repo/'research/x34_h1_stage.json').read_text())['runtime'];files['runtime.json']=json.dumps(runtime).encode()
files['run_main.py']=b'''import json,os,pathlib,subprocess,sys,time
import numpy as np,torch,torch_xla
from audit.metrics import paired_panel
from audit.protocol import jobs
root=pathlib.Path(__file__).parent
expected=json.loads((root/'runtime.json').read_text())
if str(torch.__version__)!=expected['torch_version'] or str(torch_xla.__version__)!=expected['torch_xla_version']:raise RuntimeError('Pinned runtime changed')
hits=[p for p in pathlib.Path('/kaggle/input').rglob('manifest.json') if 'mscapital-prepared-v2' in str(p) and json.loads(p.read_text()).get('fingerprint')==expected['dataset_fingerprint']]
if len(hits)!=1:raise RuntimeError('Prepared dataset missing/ambiguous')
data=hits[0].parent;out=pathlib.Path('/kaggle/working');env=dict(os.environ,PYTHONPATH=str(root));deadline=time.monotonic()+11.5*3600
planned=jobs(json.loads((root/'control.json').read_text()));completed=[]
for job in planned:
 for arm in ('control','h1'):
  remaining=(deadline-time.monotonic())/3600
  if remaining<=0:break
  subprocess.run([sys.executable,str(root/'train_audited.py'),'--config',str(root/(arm+'.json')),'--data',str(data),'--out',str(out/arm),'--device','xla','--job-key',job.key,'--max-hours',str(remaining)],env=env,check=True)
  if not (out/arm/'jobs'/job.key/'done.json').exists():break
  completed.append({'arm':arm,'job':job.key})
  (out/'confirmation_progress.json').write_text(json.dumps({'complete':len(completed)==40,'models':completed,'expected':40},indent=2))
 if len(completed)%2 or time.monotonic()>=deadline:break
if len(completed)==40:
 for arm in ('control','h1'):
  subprocess.run([sys.executable,str(root/'summarize_audited.py'),'--data',str(data),'--out',str(out/arm)],env=env,check=True)
 subprocess.run([sys.executable,str(root/'compare_audited.py'),'--data',str(data),'--base',str(out/'control'),'--candidate',str(out/'h1'),'--base-arm','flow31','--candidate-arm','flow31','--out',str(out/'h1_comparison.json')],env=env,check=True)
 with np.load(out/'control/aggregate_flow31.npz') as b,np.load(out/'h1/aggregate_flow31.npz') as c:
  if not np.array_equal(b['rows'],c['rows']):raise RuntimeError('Rows differ')
  rows=b['rows'];bp=b['processed'];cp=c['processed'];blend=(bp+cp)/2
  y=np.load(data/'y.npy',mmap_mode='r')[rows];m=np.load(data/'month.npy',mmap_mode='r')[rows]
  report=paired_panel(bp,blend,y,m,'cosine');report['fixed_blend_weights']=[0.5,0.5];report['outer_protocol']=True;report['historical_windows_reused']=True
  report['by_origin']={str(o):paired_panel(bp[mask],blend[mask],y[mask],m[mask],'cosine') for o,mask in [(59,(m>=62)&(m<=66)),(64,(m>=67)&(m<=70))]}
  (out/'blend_confirmation.json').write_text(json.dumps(report,indent=2))
  np.savez_compressed(out/'blend_external.npz',rows=rows,control=bp,h1=cp,blend=blend,y=y,month=m)
else:print('INCOMPLETE: no comparison',flush=True)
'''
for name,value in files.items():
 if name.endswith('.py'):compile(value,name,'exec')
s=io.BytesIO()
with zipfile.ZipFile(s,'w',zipfile.ZIP_DEFLATED) as z:
 for name,value in sorted(files.items()):z.writestr(name,value)
payload=base64.b64encode(s.getvalue()).decode()
source=f"import base64,io,pathlib,runpy,sys,tempfile,zipfile\nif __name__=='__main__':\n with tempfile.TemporaryDirectory(prefix='mscapital-h1-confirm-') as d:\n  zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(d)\n  sys.path.insert(0,d)\n  runpy.run_path(str(pathlib.Path(d)/'run_main.py'),run_name='__main__')\n"
(repo/'dist/h1_confirmation.py').write_text(source)
report={'models':40,'soft_tpu_hours':11.5,'hard_tpu_hours':12,'fixed_blend_weights':[0.5,0.5],'runtime':runtime,'source_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},'bundle_sha256':hashlib.sha256(source.encode()).hexdigest()}
(repo/'research/h1_confirmation_stage.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['models','soft_tpu_hours','hard_tpu_hours','bundle_sha256']}))
