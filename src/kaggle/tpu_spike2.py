import os, json, time
out = {'env_tpu': {k: v for k, v in os.environ.items() if 'TPU' in k.upper()}}
try:
    import jax; out['jax'] = jax.__version__
    out['jax_devices'] = [str(d) for d in jax.devices()]
except Exception as e:
    out['jax_err'] = str(e)[:300]
try:
    import tensorflow as tf; out['tf'] = tf.__version__
    out['tf_tpu'] = [d.name for d in tf.config.list_logical_devices('TPU')]
except Exception as e:
    out['tf_err'] = str(e)[:300]
try:
    import flax; out['flax'] = flax.__version__
except Exception as e:
    out['flax_err'] = str(e)[:100]
print('SPIKE2 ' + json.dumps(out), flush=True)
json.dump(out, open('/kaggle/working/spike2.json', 'w'))
