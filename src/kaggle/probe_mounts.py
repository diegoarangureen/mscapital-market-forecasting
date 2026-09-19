import subprocess
r = subprocess.run(['find','/kaggle/input','-maxdepth','4'], capture_output=True, text=True, timeout=120)
print(r.stdout)
print('ERR', r.stderr[:500])
