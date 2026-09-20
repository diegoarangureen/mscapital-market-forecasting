"""Check production DP eval shapes before spending full-training compute."""
import ast
from prepare_transformer_multiwindow_v13 import KERNEL

SMOKE = '''    model.eval()
    smoke_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)
    smoke_batch = next(iter(smoke_loader))
    for smoke_size in (512, 6, 7):
        smoke_inputs = [item[:smoke_size].to(device).contiguous() for item in smoke_batch[:4]]
        with torch.no_grad():
            smoke_prediction = model(*smoke_inputs)
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(gpu_id)
        if not torch.isfinite(smoke_prediction).all():
            raise RuntimeError("Nonfinite fulltrain startup prediction")
        print(f"startup_dual_gpu_eval_pass batch={smoke_size}", flush=True)
    del smoke_loader, smoke_batch, smoke_inputs, smoke_prediction
    model.train()
'''

path = KERNEL / 'run_v14_multiwindow10_fulltrain.py'
source = path.read_text(encoding='utf-8')
marker = '    model = TransformerCnnModel().to(device)\n'
position = source.rfind(marker)
assert position >= source.rfind('def main():')
if 'Nonfinite fulltrain startup prediction' not in source:
    position += len(marker)
    source = source[:position] + SMOKE + source[position:]
ast.parse(source)
path.write_text(source, encoding='utf-8')
(KERNEL / 'run_v15_multiwindow10_fulltrain.py').write_text(source, encoding='utf-8')
print('Fulltrain startup eval checks prepared: batches512,6,7; synchronize both GPUs')
