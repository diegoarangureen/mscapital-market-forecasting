# Multi-metric evaluation panel (ZWQ1037-style). Paste into kernel scripts (they run standalone).
# Usage: report_panel(pred_val, y_val, months_val, tag)
#   pred_val, y_val: 1-D float arrays aligned by sample; months_val: int array of month ids.
# Prints: overall cosine, 62-70-no66 slice, 67-70 slice, per-month cosine, monthly std,
# worst month, q25, LOMO minimum (leave-one-month-out cosine over the primary slice months).
import numpy as np

def _cos(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0

def report_panel(pred, y, months, tag=''):
    months = np.asarray(months)
    out = {}
    out['overall'] = _cos(pred, y)
    prim = (months >= 62) & (months != 66)
    out['62-70-no66'] = _cos(pred[prim], y[prim]) if prim.sum() else float('nan')
    rec = months >= 67
    out['67-70'] = _cos(pred[rec], y[rec]) if rec.sum() else float('nan')
    pm = {}
    for m in sorted(set(months[prim].tolist())):
        sel = months == m
        pm[m] = _cos(pred[sel], y[sel])
    vals = np.array(list(pm.values()))
    out['monthly_std'] = float(vals.std()); out['worst_month'] = float(vals.min())
    out['q25'] = float(np.quantile(vals, 0.25))
    lomo = []
    for m in pm:
        sel = prim & (months != m)
        lomo.append(_cos(pred[sel], y[sel]))
    out['lomo_min'] = float(min(lomo)) if lomo else float('nan')
    print(f'PANEL {tag}: overall={out["overall"]:.6f} no66={out["62-70-no66"]:.6f} '
          f'recent={out["67-70"]:.6f} std={out["monthly_std"]:.6f} worst={out["worst_month"]:.6f} '
          f'q25={out["q25"]:.6f} lomo_min={out["lomo_min"]:.6f}', flush=True)
    for m, v in pm.items():
        print(f'  month {m}: {v:.6f}', flush=True)
    return out
