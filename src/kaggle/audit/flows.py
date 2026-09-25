"""Sample-complete streaming and chronological event math for X30v2/X31v2."""
import os
import numpy as np


def complete_samples(chunks):
    """Keep the trailing sample until complete. Input must be grouped in increasing ID.

    No event is duplicated. Stable tie order is preserved, including chunk boundaries.
    A globally ungrouped input fails rather than silently creating incorrect features.
    """
    carry = None
    last_id = -1
    for chunk in chunks:
        chunk = tuple(np.asarray(c) for c in chunk)
        if not len(chunk[0]):
            continue
        sid = chunk[0]
        if np.any(sid < 0) or np.any(sid != sid.astype(np.int64)):
            raise ValueError("sample_id must be nonnegative integers")
        if np.any(sid[1:] < sid[:-1]) or sid[0] < last_id:
            raise ValueError("Input must be globally grouped by increasing sample_id")
        last_id = sid[-1]
        if carry is not None:
            chunk = tuple(np.concatenate([a, b]) for a, b in zip(carry, chunk))
        sid = chunk[0]
        cut = int(np.searchsorted(sid, sid[-1], side="left"))
        if cut:
            yield tuple(c[:cut] for c in chunk)
        carry = tuple(c[cut:].copy() for c in chunk)
    if carry is not None:
        yield carry


def ziter(path, cols, elems=None, allow_nonfinite=()):
    import pyarrow as pa
    elems = int(elems or os.environ.get("CHUNK_ROWS", 1 << 19))
    if elems < 1 or cols[0] != "sample_id":
        raise ValueError("Positive chunk size and sample_id as first column required")
    # Arrow still decompresses the giant record batch; this is NOT constant-memory IO.
    # Complete-sample reblocking fixes event continuity independently of IPC batch size.
    # allow_nonfinite: columns whose NaN is a DOCUMENTED nodata marker (probe 2026-09-25:
    # train/market.feather transaction_avgprice has 68,629,744 NaN = intervals with no
    # trades; every other column/stream is fully finite). They are zeroed and counted
    # loudly; nonfinite anywhere else still raises.
    allow = set(allow_nonfinite)
    nodata_counts = {}
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        def raw():
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                refs = [batch.column(c) for c in cols]
                for start in range(0, batch.num_rows, elems):
                    yield tuple(c.slice(start, elems).to_numpy(zero_copy_only=False) for c in refs)
        for chunk in complete_samples(raw()):
            fixed = []
            for name, c in zip(cols, chunk):
                if np.issubdtype(c.dtype, np.floating):
                    bad = ~np.isfinite(c)
                    if bad.any():
                        if name not in allow:
                            raise ValueError(f"Nonfinite raw data: {path} column {name}")
                        nodata_counts[name] = nodata_counts.get(name, 0) + int(bad.sum())
                        c = np.where(bad, 0.0, c)
                fixed.append(c)
            chunk = tuple(fixed)
            if "seconds_before_predict" in cols:
                times = chunk[cols.index("seconds_before_predict")]
                limit = 600 if str(path).endswith('market.feather') else 60
                if (times < 0).any() or (times > limit).any():
                    raise ValueError("seconds_before_predict outside its stream window")
            for key in ('side','order_action'):
                if key in cols and not np.isin(chunk[cols.index(key)],[0,1]).all():
                    raise ValueError(f'Unexpected encoding: {key}')
            yield chunk
        for name, n in nodata_counts.items():
            print(f'ziter nodata: {n} nonfinite zeroed in {name} ({path})', flush=True)


def chronological_order(sid, seconds):
    # Original row index makes equal-timestamp handling explicit and deterministic.
    return np.lexsort((np.arange(len(sid)), -np.asarray(seconds, dtype=np.float64), sid))


def ofi_level(pb, pv, pa, pav, same):
    db, da = np.diff(pb), np.diff(pa)
    bid = np.where(db > 0, pv[1:], np.where(db < 0, -pv[:-1], np.diff(pv)))
    ask = np.where(da < 0, pav[1:], np.where(da > 0, -pav[:-1], np.diff(pav)))
    valid = same & (pb[1:] > 0) & (pb[:-1] > 0) & (pa[1:] > 0) & (pa[:-1] > 0)
    return np.where(valid, bid - ask, 0.0)


def kyle_terms(sid, price, signed_volume):
    same = sid[1:] == sid[:-1]
    valid = same & (price[1:] > 0) & (price[:-1] > 0)
    ret = np.zeros(len(same), dtype=np.float64)
    ret[valid] = np.log(price[1:][valid] / price[:-1][valid])
    sv = signed_volume[1:]
    return ret * sv, np.where(valid, sv * sv, 0.0)


def new_order_gaps(sid, seconds, is_new):
    selected = np.flatnonzero(is_new)
    s, t = sid[selected], seconds[selected]
    same = s[1:] == s[:-1]
    return s[1:][same], np.abs(np.diff(t))[same]


def local_mid_reference(mid_by_bucket, valid_bucket):
    count = valid_bucket.sum(axis=1)
    mean = np.divide(np.where(valid_bucket, mid_by_bucket, 0).sum(axis=1), count,
                     out=np.ones(len(count), dtype=np.float64), where=count > 0)
    return np.maximum(mean, 1e-12)
