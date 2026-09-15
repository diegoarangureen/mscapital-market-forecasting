"""Regenerate gbm_xcos_ab.py and mlp246_v2.py (lost in 2026-09-15 13:23 sandbox wipe).
Replays the exact recorded transformations over repo files. Run from /tmp/work."""
import os
os.chdir('/tmp/work')
R = '/tmp/repo/src/recovered'

# --- gbm_xcos_ab.py from gbm_xrank_ab.py ---
src = open(f'{R}/gbm_xrank_ab.py').read()
src = src.replace("""y_raw = lab.target.values.astype(np.float64); month = lab.month.values
import pandas as pd
yr = pd.Series(y_raw).groupby(month).rank(pct=True).values.astype(np.float64)
yr = (yr - 0.5) * 2  # per-month uniform rank -> [-1,1]
y = yr""", "y_raw = lab.target.values.astype(np.float64); month = lab.month.values\ny = y_raw")
src = src.replace("""        ds=lgb.Dataset(Xtr,y[trm]); dsv=lgb.Dataset(Xv,yv,reference=ds)
        m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])""",
"""        def cos_ev(preds, dset):
            yy = dset.get_label()
            return ('cos', float(preds@yy/(np.linalg.norm(preds)*np.linalg.norm(yy)+1e-30)), True)
        ds=lgb.Dataset(Xtr,y[trm]); dsv=lgb.Dataset(Xv,yv,reference=ds)
        m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],feval=cos_ev,callbacks=[lgb.early_stopping(100,first_metric_only=False,verbose=False)])""")
src = src.replace('xallrank','xallcos').replace('XRANK_DONE','XCOS_DONE')
src = src.replace("'246f rawY 0.132116'","'246f L2-ES 0.132116'").replace("'246f rawY shift 0.128442'","'246f L2-ES shift 0.128442'")
open('gbm_xcos_ab.py','w').write(src)
import ast; ast.parse(src); print('gbm_xcos_ab.py OK')

# --- mlp246_v2.py from mlp_246.py (clean version) ---
src = open(f'{R}/mlp_246.py').read()
old_loss = "            loss = ((net(xb).squeeze(-1)-ytt[perm[i:i+bs]])**2).mean()"
new_loss = """            pred = net(xb).squeeze(-1); yb = ytt[perm[i:i+bs]]
            mse = ((pred-yb)**2).mean()
            pc = pred - pred.mean(); yc = yb - yb.mean()
            cosl = 1 - (pc*yc).sum()/(pc.norm()*yc.norm()+1e-8)
            loss = mse + 0.1*cosl"""
assert old_loss in src
src = src.replace(old_loss, new_loss)
src = src.replace("    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)",
"""    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    import copy
    ema = copy.deepcopy(net)
    for p_ in ema.parameters(): p_.requires_grad_(False)
    def ema_update(d=0.999):
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(d).add_(pn, alpha=1-d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)""")
src = src.replace("            opt.zero_grad(); loss.backward(); opt.step()",
                  "            opt.zero_grad(); loss.backward(); opt.step(); ema_update()")
src = src.replace("        net.eval()\n        with torch.no_grad():",
                  "        ema.eval()\n        with torch.no_grad():")
src = src.replace("pv[i:i+65536] = net(xb).squeeze(-1).numpy()",
                  "pv[i:i+65536] = ema(xb).squeeze(-1).numpy()")
src = src.replace("np.save(f'{W}/mlp246_val.npy', pv)", "np.save(f'{W}/mlp246v2_val.npy', pv)")
src = src.replace("print('MLP246_DONE', flush=True)", "print('MLP246V2_DONE', flush=True)")
src = src.replace("M = np.memmap(f'{W}/mlp246.f32', dtype=np.float32, mode='w+', shape=(n, D))",
                  "M = np.memmap(f'{W}/mlp246v2.f32', dtype=np.float32, mode='w+', shape=(n, D))")
open('mlp246_v2.py','w').write(src)
ast.parse(src); print('mlp246_v2.py OK')
