# TPU spike: verify torch_xla works on Kaggle TPU and time a small RealMLP-ish step.
import time, os, json
out = {}
try:
    import torch
    out['torch'] = torch.__version__
    import torch_xla
    out['torch_xla'] = torch_xla.__version__
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    out['device'] = str(dev)
    import torch.nn as nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(455, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 16)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(256, 455, device=dev); y = torch.randn(256, device=dev)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for i in range(20):
        opt.zero_grad()
        pred = m(x).mean(dim=1)
        loss = loss_fn(pred, y)
        loss.backward(); opt.step()
    xm.mark_step(); xm.wait_device_ops()
    out['loss'] = float(loss.cpu()); out['train_20steps_s'] = time.time() - t0
    out['status'] = 'ok'
except Exception as e:
    import traceback; out['status'] = 'error'; out['error'] = traceback.format_exc()[-2000:]
print('SPIKE_RESULT ' + json.dumps(out), flush=True)
json.dump(out, open('/kaggle/working/spike.json', 'w'))
