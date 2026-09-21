import os
out = []
try:
    import jax
    out.append('JAX DEVICES: %s' % jax.devices())
except Exception as e:
    out.append('jax err: %s' % e)
out.append('ACCELERATOR env: %s' % os.environ.get('KAGGLE_ACCELERATOR','none'))
out.append('TPU_NAME env: %s' % os.environ.get('TPU_NAME','none'))
with open('/kaggle/working/probe_result.txt','w') as f:
    f.write('\n'.join(out))
print('\n'.join(out))
