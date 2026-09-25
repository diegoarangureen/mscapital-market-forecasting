"""Score complete vectors; never average batch/fold cosines to obtain a score."""
import numpy as np


def cosine(pred, target):
    p, y = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("Expected nonempty aligned vectors")
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite score inputs")
    den = np.linalg.norm(p) * np.linalg.norm(y)
    return float(p @ y / den) if den > 0 else 0.0


def pearson(pred, target):
    p, y = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    return cosine(p - p.mean(), y - y.mean())


SCORERS = {"cosine": cosine, "pearson": pearson}


class PredictionMean:
    """Missing predictions do not count as zero. Reject duplicate model contributions."""
    def __init__(self, size):
        self.total = np.zeros(size, dtype=np.float64)
        self.count = np.zeros(size, dtype=np.int32)
        self.models = set()

    def add(self, key, rows, pred):
        rows = np.asarray(rows, dtype=np.int64)
        pred = np.asarray(pred, dtype=np.float64)
        if key in self.models:
            raise ValueError(f"Duplicate model: {key}")
        if rows.ndim != 1 or rows.shape != pred.shape or len(np.unique(rows)) != len(rows):
            raise ValueError("Bad prediction row mapping")
        if (rows < 0).any() or (rows >= len(self.total)).any() or not np.isfinite(pred).all():
            raise ValueError("Invalid prediction contribution")
        self.total[rows] += pred
        self.count[rows] += 1
        self.models.add(key)

    def mean(self):
        return np.divide(self.total, self.count, out=np.full(len(self.total), np.nan),
                         where=self.count > 0)


def postprocess(pred, bounds=None, nodata=None):
    result = np.asarray(pred, dtype=np.float64).copy()
    if bounds is not None:
        lo, hi = bounds
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("Invalid frozen clipping bounds")
        result = np.clip(result, lo, hi)
    # Zero AFTER clipping, even when the fitted clipping interval excludes zero.
    if nodata is not None:
        result[np.asarray(nodata, dtype=bool)] = 0.0
    return result


def panel(pred, target, months, metric="cosine"):
    p, y, m = np.asarray(pred), np.asarray(target), np.asarray(months)
    if p.shape != y.shape or p.shape != m.shape:
        raise ValueError("Panel inputs are not aligned")
    if not len(p):
        raise ValueError("Empty evaluation panel")
    score = SCORERS[metric]
    monthly = {str(int(k)): score(p[m == k], y[m == k]) for k in np.unique(m)}
    lomo = {str(int(k)): score(p[m != k], y[m != k]) for k in np.unique(m)
            if np.any(m != k)}
    return {"metric": metric, "n": len(p), "overall": score(p, y),
            "cosine": cosine(p, y), "pearson": pearson(p, y),
            "prediction_mean": float(p.mean()), "prediction_std": float(p.std()),
            "monthly": monthly, "lomo": lomo, "worst_month": min(monthly.values()),
            "without_66_sensitivity": score(p[m != 66], y[m != 66]) if np.any(m != 66) else None}


def paired_panel(base, candidate, target, months, metric="cosine", bootstrap=1000, seed=17):
    b = panel(base, target, months, metric)
    c = panel(candidate, target, months, metric)
    m = np.asarray(months)
    groups = [np.flatnonzero(m == k) for k in np.unique(m)]
    rng = np.random.default_rng(seed)
    deltas = []
    # Exact sufficient statistics avoid copying millions of rows for every draw.
    def moments(pred):
        pp,yy=np.asarray(pred,np.float64),np.asarray(target,np.float64)
        return np.asarray([[len(ix),pp[ix].sum(),yy[ix].sum(),pp[ix]@pp[ix],
                            yy[ix]@yy[ix],pp[ix]@yy[ix]] for ix in groups])
    bm,cm=moments(base),moments(candidate)
    def from_moments(v):
        n,sp,sy,pp,yy,py=v
        if metric=='pearson':
            pp,yy,py=pp-sp*sp/n,yy-sy*sy/n,py-sp*sy/n
        den=np.sqrt(max(pp,0)*max(yy,0))
        return float(py/den) if den>0 else 0.
    for _ in range(bootstrap):
        draw=rng.integers(len(groups),size=len(groups))
        deltas.append(from_moments(cm[draw].sum(axis=0))-from_moments(bm[draw].sum(axis=0)))
    return {"base": b, "candidate": c, "delta": c["overall"] - b["overall"],
            "monthly_delta": {k: c["monthly"][k] - v for k, v in b["monthly"].items()},
            "lomo_delta": {k: c["lomo"][k] - v for k, v in b["lomo"].items()},
            "month_bootstrap_interval_95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
            "uncertainty_note": "Paired resampling of months; few/dependent months and prior selection limit inference."}
