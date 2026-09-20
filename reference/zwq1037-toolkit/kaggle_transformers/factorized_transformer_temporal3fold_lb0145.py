import gc
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from scipy.stats import rankdata
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
except ModuleNotFoundError:
    pa = None
    ds = None
    feather = None
    ipc = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None


DATA = Path(os.environ.get(
    "DATA_DIR",
    "/kaggle/input/competitions/ms-capital-real-financial-market-forecasting",
))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working/factorized_transformer"))
GRID_INPUT = None
for _candidate in (
    Path("/kaggle/input/mscapital-multistream-grid-cache-v2"),
    Path("/kaggle/input/zwq1037/mscapital-multistream-grid-cache-v2"),
):
    if (_candidate / "train_v2_market_200x11.mmap").exists():
        GRID_INPUT = _candidate
        break
if GRID_INPUT is None:
    for _manifest in Path("/kaggle/input").rglob("cache_manifest.json"):
        if (_manifest.parent / "train_v2_market_200x11.mmap").exists():
            GRID_INPUT = _manifest.parent
            break
if GRID_INPUT is None:
    raise FileNotFoundError("MSCapital multistream grid cache v2 is not attached")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(GRID_INPUT)))
OUT_CSV = os.environ.get("OUT_CSV", "submission.csv")

BATCH_SIZE_ARROW = int(os.environ.get("BATCH_SIZE_ARROW", "65536"))
SEED = int(os.environ.get("SEED", "2026"))
TRAIN_END_MONTH = int(os.environ.get("TRAIN_END_MONTH", "59"))
VALID_START_MONTH = int(os.environ.get("VALID_START_MONTH", "62"))
VALID_END_MONTH = int(os.environ.get("VALID_END_MONTH", "70"))
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
LR = float(os.environ.get("LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
REBUILD_GRID = os.environ.get("REBUILD_GRID", "0") == "1"
USE_AMP = os.environ.get("USE_AMP", "0") == "1"
GRID_VERSION = os.environ.get("GRID_VERSION", "v2")
BUNDLED_STREAM_NORM = {'market': {'mean': [-0.0015414220979437232, 9.696150300442241e-06, -0.003384847892448306, -0.0013178293593227863, 0.0304400734603405, 0.05962716415524483, -0.0031259567476809025, 0.0020670865196734667, 9.735998153686523, 4.324946403503418, 0.9573338031768799], 'std': [0.02812854014337063, 0.0024828508030623198, 0.11209283024072647, 0.11349061876535416, 0.554838240146637, 0.4417761564254761, 0.055987726897001266, 0.017819251865148544, 4.224153995513916, 3.7372190952301025, 1.0261322259902954]}, 'tx': {'mean': [-2.145182236290566e-07, -5.897786650166381e-06, 0.002079953905194998, 1.5597456693649292, 0.4176950752735138, -0.002701796591281891, 0.10776454955339432], 'std': [3.240046862629242e-05, 0.0007616449729539454, 0.006281367968767881, 2.294206142425537, 0.7207639813423157, 0.5228376984596252, 0.23715408146381378]}, 'order': {'mean': [0.0005054076900705695, 0.005767660681158304, 2.821978807449341, 0.7524027228355408, 0.02068391628563404, 0.3065175414085388, 0.009760000742971897, 0.1391841471195221, 0.01644830033183098, 0.008429652079939842], 'std': [0.009913619607686996, 0.010851267725229263, 2.4329922199249268, 0.7964388132095337, 0.6251509189605713, 0.5694777369499207, 0.60309237241745, 0.27976927161216736, 0.6067987084388733, 0.4691857099533081]}}

MARKET_LEN = int(os.environ.get("MARKET_LEN", "200"))
FLOW_LEN = int(os.environ.get("FLOW_LEN", "60"))
MARKET_SECONDS = 600.0
FLOW_SECONDS = 60.0

MARKET_FEATURES = [
    "mid_rel",
    "txpx_rel",
    "rel_spread1",
    "rel_spread2",
    "imb1",
    "imb2",
    "micro_rel",
    "slope",
    "log_depth",
    "log_txv",
    "log_txc",
]
TX_FEATURES = [
    "vwap_rel",
    "price_mean_rel",
    "price_std",
    "log_vol",
    "log_count",
    "signed_ratio",
    "buy_trade_ratio",
]
ORDER_FEATURES = [
    "price_mean_rel",
    "price_std",
    "log_vol",
    "log_count",
    "signed_ratio",
    "action_ratio",
    "pressure_ratio",
    "cancel_ratio",
    "new_imb",
    "cancel_imb",
]

EPS = 1e-6


def require_runtime():
    if pa is None or feather is None or ipc is None:
        raise ModuleNotFoundError("pyarrow is required on Kaggle.")
    if torch is None:
        raise ModuleNotFoundError("torch is required on Kaggle.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_train_label():
    tab = feather.read_table(
        str(DATA / "train" / "label.feather"),
        columns=["sample_id", "month", "target"],
        memory_map=False,
    )
    sample_id = tab["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    month = tab["month"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    target = tab["target"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    order = np.argsort(sample_id)
    return sample_id[order], month[order], target[order]


def read_submission_ids():
    ids = []
    with open(DATA / "submission.csv", "r") as f:
        next(f)
        for line in f:
            if line:
                ids.append(int(line.split(",", 1)[0]))
    return np.asarray(ids, dtype=np.int64)


def build_indexer(sample_ids):
    max_id = int(sample_ids.max())
    if max_id <= 25_000_000:
        indexer = np.full(max_id + 1, -1, dtype=np.int32)
        indexer[sample_ids] = np.arange(sample_ids.size, dtype=np.int32)
        return "direct", indexer
    return "dict", {int(s): i for i, s in enumerate(sample_ids)}


def map_ids(batch_ids, indexer):
    kind, payload = indexer
    if kind == "direct":
        out = np.full(batch_ids.size, -1, dtype=np.int32)
        ok = (batch_ids >= 0) & (batch_ids < payload.size)
        out[ok] = payload[batch_ids[ok]]
        return out
    return np.fromiter((payload.get(int(x), -1) for x in batch_ids), dtype=np.int32, count=batch_ids.size)


def iter_batches(path, columns):
    if ds is not None:
        try:
            dataset = ds.dataset(str(path), format="ipc")
            scanner = dataset.scanner(columns=columns, batch_size=BATCH_SIZE_ARROW, use_threads=False)
            for i, batch in enumerate(scanner.to_batches(), start=1):
                yield batch, i, None
            return
        except Exception as exc:
            print(f"  dataset scanner fallback for {path.name}: {type(exc).__name__}: {exc}", flush=True)

    source = pa.memory_map(str(path), "r")
    try:
        try:
            reader = ipc.RecordBatchFileReader(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                try:
                    batch = batch.select(columns)
                except Exception:
                    pass
                yield batch, i + 1, reader.num_record_batches
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_stream(source)
            for i, batch in enumerate(reader, start=1):
                try:
                    batch = batch.select(columns)
                except Exception:
                    pass
                yield batch, i, None
    finally:
        source.close()


def col_np(batch, name, dtype=None):
    arr = batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def open_memmap(path, shape, dtype=np.float16, mode="w+"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def temp_memmap(split, name, shape, dtype=np.float32):
    path = CACHE_DIR / f"{split}_tmp_{name}.mmap"
    arr = open_memmap(path, shape, dtype=dtype, mode="w+")
    arr[:] = 0
    return arr


def cleanup_temp(split, prefix):
    for path in CACHE_DIR.glob(f"{split}_tmp_{prefix}*.mmap"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def feature_paths(split):
    return {
        "market": CACHE_DIR / f"{split}_{GRID_VERSION}_market_{MARKET_LEN}x{len(MARKET_FEATURES)}.mmap",
        "market_count": CACHE_DIR / f"{split}_{GRID_VERSION}_market_count_{MARKET_LEN}.mmap",
        "tx": CACHE_DIR / f"{split}_{GRID_VERSION}_tx_{FLOW_LEN}x{len(TX_FEATURES)}.mmap",
        "tx_count": CACHE_DIR / f"{split}_{GRID_VERSION}_tx_count_{FLOW_LEN}.mmap",
        "order": CACHE_DIR / f"{split}_{GRID_VERSION}_order_{FLOW_LEN}x{len(ORDER_FEATURES)}.mmap",
        "order_count": CACHE_DIR / f"{split}_{GRID_VERSION}_order_count_{FLOW_LEN}.mmap",
    }


def market_bin(seconds):
    bins = MARKET_LEN - 1 - np.floor(seconds / (MARKET_SECONDS / MARKET_LEN)).astype(np.int32)
    return np.clip(bins, 0, MARKET_LEN - 1)


def flow_bin(seconds):
    bins = FLOW_LEN - 1 - np.floor(seconds / (FLOW_SECONDS / FLOW_LEN)).astype(np.int32)
    return np.clip(bins, 0, FLOW_LEN - 1)


def add_at_2d(arr, row, col, values):
    values = np.nan_to_num(values, nan=0.0, posinf=50.0, neginf=-50.0)
    values = np.clip(values, -50.0, 50.0)
    np.add.at(arr, (row, col), values.astype(arr.dtype, copy=False))


def add_at_3d(arr, row, col, feat, values):
    values = np.nan_to_num(values, nan=0.0, posinf=50.0, neginf=-50.0)
    values = np.clip(values, -50.0, 50.0)
    np.add.at(arr, (row, col, feat), values.astype(arr.dtype, copy=False))


def clean_feature_block(x, clip=50.0):
    x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(x, -clip, clip)


def build_market_grid(split, sample_ids):
    n = sample_ids.size
    paths = feature_paths(split)
    feat = open_memmap(paths["market"], (n, MARKET_LEN, len(MARKET_FEATURES)), np.float16)
    cnt = open_memmap(paths["market_count"], (n, MARKET_LEN), np.float16)
    feat[:] = 0.0
    cnt[:] = 0.0
    indexer = build_indexer(sample_ids)

    columns = [
        "sample_id", "seconds_before_predict", "transaction_avgprice", "transaction_volume",
        "transaction_count", "ask_price_1", "ask_price_2", "bid_price_1", "bid_price_2",
        "ask_volume_1", "ask_volume_2", "bid_volume_1", "bid_volume_2",
    ]
    t0 = time.time()
    rows_seen = 0
    for batch, batch_no, total_batches in iter_batches(DATA / split / "market.feather", columns):
        sid = col_np(batch, "sample_id", np.int64)
        row = map_ids(sid, indexer)
        ok = row >= 0
        if not np.all(ok):
            row = row[ok]
        sec = col_np(batch, "seconds_before_predict", np.float32)
        ask1 = col_np(batch, "ask_price_1", np.float32)
        ask2 = col_np(batch, "ask_price_2", np.float32)
        bid1 = col_np(batch, "bid_price_1", np.float32)
        bid2 = col_np(batch, "bid_price_2", np.float32)
        askv1 = col_np(batch, "ask_volume_1", np.float32)
        askv2 = col_np(batch, "ask_volume_2", np.float32)
        bidv1 = col_np(batch, "bid_volume_1", np.float32)
        bidv2 = col_np(batch, "bid_volume_2", np.float32)
        txpx = col_np(batch, "transaction_avgprice", np.float32)
        txv = col_np(batch, "transaction_volume", np.float32)
        txc = col_np(batch, "transaction_count", np.float32)
        if not np.all(ok):
            sec = sec[ok]; ask1 = ask1[ok]; ask2 = ask2[ok]; bid1 = bid1[ok]; bid2 = bid2[ok]
            askv1 = askv1[ok]; askv2 = askv2[ok]; bidv1 = bidv1[ok]; bidv2 = bidv2[ok]
            txpx = txpx[ok]; txv = txv[ok]; txc = txc[ok]

        col = market_bin(sec)
        mid = (ask1 + bid1) * 0.5
        depth1 = askv1 + bidv1
        depth2 = depth1 + askv2 + bidv2
        imb1 = (bidv1 - askv1) / (depth1 + EPS)
        imb2 = (bidv1 + bidv2 - askv1 - askv2) / (depth2 + EPS)
        micro = (ask1 * bidv1 + bid1 * askv1) / (depth1 + EPS)
        values = [
            mid - 1.0,
            txpx - 1.0,
            (ask1 - bid1) / (mid + EPS),
            (ask2 - bid2) / (mid + EPS),
            imb1,
            imb2,
            micro / (mid + EPS) - 1.0,
            ((ask2 - ask1) + (bid1 - bid2)) / (mid + EPS),
            np.log1p(depth2),
            np.log1p(txv),
            np.log1p(txc),
        ]
        np.add.at(cnt, (row, col), 1.0)
        for j, val in enumerate(values):
            add_at_3d(feat, row, col, j, val)

        rows_seen += batch.num_rows
        if batch_no % 200 == 0:
            print(f"  {split}/market batch={batch_no}, rows={rows_seen:,}, elapsed={time.time() - t0:.1f}s", flush=True)
        del batch, sid, row, ok, sec, ask1, ask2, bid1, bid2, askv1, askv2, bidv1, bidv2, txpx, txv, txc

    mask = cnt > 0
    for j in range(len(MARKET_FEATURES)):
        channel = feat[:, :, j]
        channel[mask] = channel[mask] / cnt[mask]
        channel[~mask] = 0.0
        feat[:, :, j] = clean_feature_block(channel)
    feat.flush()
    cnt.flush()
    print(f"  {split}/market grid done, rows={rows_seen:,}, time={time.time() - t0:.1f}s", flush=True)
    return paths


def build_transaction_grid(split, sample_ids):
    n = sample_ids.size
    paths = feature_paths(split)
    feat = open_memmap(paths["tx"], (n, FLOW_LEN, len(TX_FEATURES)), np.float16)
    cnt = open_memmap(paths["tx_count"], (n, FLOW_LEN), np.float16)
    feat[:] = 0.0
    cnt[:] = 0.0
    indexer = build_indexer(sample_ids)
    columns = ["sample_id", "seconds_before_predict", "price", "volume", "side"]

    vol_sum = temp_memmap(split, "tx_vol_sum", (n, FLOW_LEN))
    amount_sum = temp_memmap(split, "tx_amount_sum", (n, FLOW_LEN))
    price_sum = temp_memmap(split, "tx_price_sum", (n, FLOW_LEN))
    price_sq_sum = temp_memmap(split, "tx_price_sq_sum", (n, FLOW_LEN))
    signed_vol = temp_memmap(split, "tx_signed_vol", (n, FLOW_LEN))
    buy_count = temp_memmap(split, "tx_buy_count", (n, FLOW_LEN))

    t0 = time.time()
    rows_seen = 0
    for batch, batch_no, total_batches in iter_batches(DATA / split / "transaction.feather", columns):
        sid = col_np(batch, "sample_id", np.int64)
        row = map_ids(sid, indexer)
        ok = row >= 0
        if not np.all(ok):
            row = row[ok]
        sec = col_np(batch, "seconds_before_predict", np.float32)
        price = col_np(batch, "price", np.float32)
        volume = col_np(batch, "volume", np.float32)
        side = col_np(batch, "side", np.int8)
        if not np.all(ok):
            sec = sec[ok]; price = price[ok]; volume = volume[ok]; side = side[ok]
        col = flow_bin(sec)
        sgn = np.where(side == 0, 1.0, -1.0).astype(np.float32)
        add_at_2d(cnt, row, col, np.ones_like(price, dtype=np.float32))
        add_at_2d(vol_sum, row, col, volume)
        add_at_2d(amount_sum, row, col, price * volume)
        add_at_2d(price_sum, row, col, price)
        add_at_2d(price_sq_sum, row, col, price * price)
        add_at_2d(signed_vol, row, col, sgn * volume)
        add_at_2d(buy_count, row, col, (side == 0).astype(np.float32))
        rows_seen += batch.num_rows
        if batch_no % 200 == 0:
            print(f"  {split}/transaction batch={batch_no}, rows={rows_seen:,}, elapsed={time.time() - t0:.1f}s", flush=True)
        del batch, sid, row, ok, sec, price, volume, side, col, sgn

    mask = cnt > 0
    price_mean = np.zeros_like(cnt)
    price_mean[mask] = price_sum[mask] / cnt[mask]
    price_std = np.zeros_like(cnt)
    price_std[mask] = np.sqrt(np.maximum(price_sq_sum[mask] / cnt[mask] - price_mean[mask] ** 2, 0.0))
    vwap = np.zeros_like(cnt)
    nz = vol_sum > 0
    vwap[nz] = amount_sum[nz] / (vol_sum[nz] + EPS)
    feat[:, :, 0] = clean_feature_block(vwap - 1.0)
    feat[:, :, 1] = clean_feature_block(price_mean - 1.0)
    feat[:, :, 2] = clean_feature_block(price_std)
    feat[:, :, 3] = clean_feature_block(np.log1p(vol_sum))
    feat[:, :, 4] = clean_feature_block(np.log1p(cnt))
    feat[:, :, 5] = clean_feature_block(signed_vol / (vol_sum + 1.0))
    feat[:, :, 6] = clean_feature_block(buy_count / (cnt + 1.0))
    feat[~mask] = 0.0
    feat.flush()
    cnt.flush()
    print(f"  {split}/transaction grid done, rows={rows_seen:,}, time={time.time() - t0:.1f}s", flush=True)
    del vol_sum, amount_sum, price_sum, price_sq_sum, signed_vol, buy_count
    gc.collect()
    cleanup_temp(split, "tx_")
    return paths


def build_order_grid(split, sample_ids):
    n = sample_ids.size
    paths = feature_paths(split)
    feat = open_memmap(paths["order"], (n, FLOW_LEN, len(ORDER_FEATURES)), np.float16)
    cnt = open_memmap(paths["order_count"], (n, FLOW_LEN), np.float16)
    feat[:] = 0.0
    cnt[:] = 0.0
    indexer = build_indexer(sample_ids)
    columns = ["sample_id", "seconds_before_predict", "price", "volume", "side", "order_action"]

    vol_sum = temp_memmap(split, "order_vol_sum", (n, FLOW_LEN))
    price_sum = temp_memmap(split, "order_price_sum", (n, FLOW_LEN))
    price_sq_sum = temp_memmap(split, "order_price_sq_sum", (n, FLOW_LEN))
    signed_vol = temp_memmap(split, "order_signed_vol", (n, FLOW_LEN))
    action_vol = temp_memmap(split, "order_action_vol", (n, FLOW_LEN))
    pressure = temp_memmap(split, "order_pressure", (n, FLOW_LEN))
    cancel_vol = temp_memmap(split, "order_cancel_vol", (n, FLOW_LEN))
    new_vol = temp_memmap(split, "order_new_vol", (n, FLOW_LEN))
    buy_new = temp_memmap(split, "order_buy_new", (n, FLOW_LEN))
    sell_new = temp_memmap(split, "order_sell_new", (n, FLOW_LEN))
    buy_cancel = temp_memmap(split, "order_buy_cancel", (n, FLOW_LEN))
    sell_cancel = temp_memmap(split, "order_sell_cancel", (n, FLOW_LEN))

    t0 = time.time()
    rows_seen = 0
    for batch, batch_no, total_batches in iter_batches(DATA / split / "order.feather", columns):
        sid = col_np(batch, "sample_id", np.int64)
        row = map_ids(sid, indexer)
        ok = row >= 0
        if not np.all(ok):
            row = row[ok]
        sec = col_np(batch, "seconds_before_predict", np.float32)
        price = col_np(batch, "price", np.float32)
        volume = col_np(batch, "volume", np.float32)
        side = col_np(batch, "side", np.int8)
        action = col_np(batch, "order_action", np.int8)
        if not np.all(ok):
            sec = sec[ok]; price = price[ok]; volume = volume[ok]; side = side[ok]; action = action[ok]
        col = flow_bin(sec)
        sgn = np.where(side == 0, 1.0, -1.0).astype(np.float32)
        act = np.where(action == 0, 1.0, -1.0).astype(np.float32)
        is_new = action == 0
        is_cancel = action == 1
        add_at_2d(cnt, row, col, np.ones_like(price, dtype=np.float32))
        add_at_2d(vol_sum, row, col, volume)
        add_at_2d(price_sum, row, col, price)
        add_at_2d(price_sq_sum, row, col, price * price)
        add_at_2d(signed_vol, row, col, sgn * volume)
        add_at_2d(action_vol, row, col, act * volume)
        add_at_2d(pressure, row, col, sgn * act * volume)
        add_at_2d(cancel_vol, row, col, np.where(is_cancel, volume, 0.0))
        add_at_2d(new_vol, row, col, np.where(is_new, volume, 0.0))
        add_at_2d(buy_new, row, col, np.where((side == 0) & is_new, volume, 0.0))
        add_at_2d(sell_new, row, col, np.where((side == 1) & is_new, volume, 0.0))
        add_at_2d(buy_cancel, row, col, np.where((side == 0) & is_cancel, volume, 0.0))
        add_at_2d(sell_cancel, row, col, np.where((side == 1) & is_cancel, volume, 0.0))
        rows_seen += batch.num_rows
        if batch_no % 200 == 0:
            print(f"  {split}/order batch={batch_no}, rows={rows_seen:,}, elapsed={time.time() - t0:.1f}s", flush=True)
        del batch, sid, row, ok, sec, price, volume, side, action, col, sgn, act, is_new, is_cancel

    mask = cnt > 0
    price_mean = np.zeros_like(cnt)
    price_mean[mask] = price_sum[mask] / cnt[mask]
    price_std = np.zeros_like(cnt)
    price_std[mask] = np.sqrt(np.maximum(price_sq_sum[mask] / cnt[mask] - price_mean[mask] ** 2, 0.0))
    feat[:, :, 0] = clean_feature_block(price_mean - 1.0)
    feat[:, :, 1] = clean_feature_block(price_std)
    feat[:, :, 2] = clean_feature_block(np.log1p(vol_sum))
    feat[:, :, 3] = clean_feature_block(np.log1p(cnt))
    feat[:, :, 4] = clean_feature_block(signed_vol / (vol_sum + 1.0))
    feat[:, :, 5] = clean_feature_block(action_vol / (vol_sum + 1.0))
    feat[:, :, 6] = clean_feature_block(pressure / (vol_sum + 1.0))
    feat[:, :, 7] = clean_feature_block(cancel_vol / (vol_sum + 1.0))
    feat[:, :, 8] = clean_feature_block((buy_new - sell_new) / (new_vol + 1.0))
    feat[:, :, 9] = clean_feature_block((buy_cancel - sell_cancel) / (cancel_vol + 1.0))
    feat[~mask] = 0.0
    feat.flush()
    cnt.flush()
    print(f"  {split}/order grid done, rows={rows_seen:,}, time={time.time() - t0:.1f}s", flush=True)
    del vol_sum, price_sum, price_sq_sum, signed_vol, action_vol, pressure, cancel_vol, new_vol
    del buy_new, sell_new, buy_cancel, sell_cancel
    gc.collect()
    cleanup_temp(split, "order_")
    return paths


def grid_exists(split, n):
    paths = feature_paths(split)
    checks = [
        (paths["market"], (n, MARKET_LEN, len(MARKET_FEATURES))),
        (paths["tx"], (n, FLOW_LEN, len(TX_FEATURES))),
        (paths["order"], (n, FLOW_LEN, len(ORDER_FEATURES))),
    ]
    return all(path.exists() and path.stat().st_size > 0 for path, _ in checks)


def build_all_grids(split, sample_ids):
    if grid_exists(split, sample_ids.size) and not REBUILD_GRID:
        print(f"{split} grids already exist; set REBUILD_GRID=1 to rebuild", flush=True)
        return feature_paths(split)
    print(f"\n=== build grids: {split} ===", flush=True)
    build_market_grid(split, sample_ids)
    build_transaction_grid(split, sample_ids)
    build_order_grid(split, sample_ids)
    return feature_paths(split)


def load_grid(split, n):
    paths = feature_paths(split)
    return {
        "market": np.memmap(paths["market"], dtype=np.float16, mode="r", shape=(n, MARKET_LEN, len(MARKET_FEATURES))),
        "tx": np.memmap(paths["tx"], dtype=np.float16, mode="r", shape=(n, FLOW_LEN, len(TX_FEATURES))),
        "order": np.memmap(paths["order"], dtype=np.float16, mode="r", shape=(n, FLOW_LEN, len(ORDER_FEATURES))),
    }


def compute_norm(arrays, train_indices, max_rows=50000, chunk_rows=2048):
    rng = np.random.default_rng(SEED)
    idx = np.asarray(train_indices)
    if idx.size > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    idx = np.sort(idx)
    norm = {}
    for name, arr in arrays.items():
        n_feat = arr.shape[-1]
        total = np.zeros(n_feat, dtype=np.float64)
        total_sq = np.zeros(n_feat, dtype=np.float64)
        total_n = np.zeros(n_feat, dtype=np.float64)
        for start in range(0, idx.size, chunk_rows):
            part_idx = idx[start:start + chunk_rows]
            x = np.asarray(arr[part_idx], dtype=np.float32)
            flat = x.reshape(-1, n_feat)
            finite = np.isfinite(flat)
            total += np.where(finite, flat, 0.0).sum(axis=0)
            flat_safe = np.where(finite, flat, 0.0)
            total_sq += (flat_safe * flat_safe).sum(axis=0)
            total_n += finite.sum(axis=0)
            del x, flat, finite
        denom = np.maximum(total_n, 1.0)
        mean = (total / denom).astype(np.float32)
        var = np.maximum(total_sq / denom - mean.astype(np.float64) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        norm[name] = {"mean": mean, "std": std}
    return norm


def save_norm(norm):
    out = {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()} for k, v in norm.items()}
    with open(WORK_DIR / f"norm_{GRID_VERSION}.json", "w") as f:
        json.dump(out, f)


def load_norm():
    with open(WORK_DIR / f"norm_{GRID_VERSION}.json") as f:
        raw = json.load(f)
    norm = {}
    for k, v in raw.items():
        mean = np.nan_to_num(np.array(v["mean"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(np.array(v["std"], dtype=np.float32), nan=1.0, posinf=1.0, neginf=1.0)
        std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
        norm[k] = {"mean": mean, "std": std}
    return norm


class GridDataset(Dataset):
    def __init__(self, arrays, indices, norm, target=None, target_scale=1.0):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm = norm
        self.target = target
        self.target_scale = target_scale

    def __len__(self):
        return self.indices.size

    def _norm(self, name, x):
        pad = np.abs(x).sum(axis=-1) == 0
        mean = self.norm[name]["mean"]
        std = self.norm[name]["std"]
        x = (x - mean) / std
        x = np.clip(x, -8.0, 8.0)
        x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)
        x[pad] = 0.0
        return x.astype(np.float32, copy=False)

    def __getitem__(self, i):
        idx = self.indices[i]
        market = self._norm("market", np.asarray(self.arrays["market"][idx], dtype=np.float32))
        tx = self._norm("tx", np.asarray(self.arrays["tx"][idx], dtype=np.float32))
        order = self._norm("order", np.asarray(self.arrays["order"][idx], dtype=np.float32))
        if self.target is None:
            return torch.from_numpy(market), torch.from_numpy(tx), torch.from_numpy(order)
        y = np.float32(np.nan_to_num(self.target[idx] / self.target_scale, nan=0.0, posinf=0.0, neginf=0.0))
        return torch.from_numpy(market), torch.from_numpy(tx), torch.from_numpy(order), torch.tensor(y)


class ConvBlock(nn.Module):
    """Local temporal mixing without cuDNN Conv1d kernels."""
    def __init__(self, d_model, kernel=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        left = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        right = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        local_average = (left + 2.0 * x + right) * 0.25
        return x + self.net(self.norm(local_average))


class TransformerCnnModel(nn.Module):
    def __init__(self, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.market_proj = nn.Linear(len(MARKET_FEATURES), d_model)
        self.tx_proj = nn.Linear(len(TX_FEATURES), d_model)
        self.order_proj = nn.Linear(len(ORDER_FEATURES), d_model)
        self.market_pos = nn.Parameter(torch.zeros(1, MARKET_LEN, d_model))
        self.tx_pos = nn.Parameter(torch.zeros(1, FLOW_LEN, d_model))
        self.order_pos = nn.Parameter(torch.zeros(1, FLOW_LEN, d_model))
        self.table_embed = nn.Parameter(torch.zeros(3, d_model))
        self.conv1 = ConvBlock(d_model, kernel=5, dropout=dropout)
        self.conv2 = ConvBlock(d_model, kernel=3, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.attn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.market_pos, std=0.02)
        nn.init.normal_(self.tx_pos, std=0.02)
        nn.init.normal_(self.order_pos, std=0.02)
        nn.init.normal_(self.table_embed, std=0.02)

    def forward(self, market, tx, order):
        m = self.market_proj(market) + self.market_pos + self.table_embed[0]
        t = self.tx_proj(tx) + self.tx_pos + self.table_embed[1]
        o = self.order_proj(order) + self.order_pos + self.table_embed[2]
        x = torch.cat([m, t, o], dim=1)
        pad_mask = torch.cat([
            market.abs().sum(-1) == 0,
            tx.abs().sum(-1) == 0,
            order.abs().sum(-1) == 0,
        ], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        score = self.attn(x).squeeze(-1)
        score = score.masked_fill(pad_mask, -1e4)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (x * weight).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def cosine_np(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float((pred * target).sum() / (np.linalg.norm(pred) * np.linalg.norm(target) + 1e-12))


def cosine_loss(pred, target):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    pred = pred - pred.mean()
    target = target - target.mean()
    return 1.0 - F.cosine_similarity(pred, target, dim=0, eps=1e-6)


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        market, tx, order, y = batch
        market = market.to(device, non_blocking=True).contiguous()
        tx = tx.to(device, non_blocking=True)
        order = order.to(device, non_blocking=True).contiguous()
        y = y.to(device, non_blocking=True)
        market = torch.nan_to_num(market, nan=0.0, posinf=8.0, neginf=-8.0)
        tx = torch.nan_to_num(tx, nan=0.0, posinf=8.0, neginf=-8.0)
        order = torch.nan_to_num(order, nan=0.0, posinf=8.0, neginf=-8.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            pred = model(market, tx, order)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            loss = 0.35 * F.smooth_l1_loss(pred, y) + 0.65 * cosine_loss(pred, y)
        if not torch.isfinite(loss):
            print("  skip non-finite loss batch", flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * y.numel()
        n += y.numel()
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        if len(batch) == 4:
            market, tx, order, _ = batch
        else:
            market, tx, order = batch
        market = market.to(device, non_blocking=True).contiguous()
        tx = tx.to(device, non_blocking=True)
        order = order.to(device, non_blocking=True).contiguous()
        market = torch.nan_to_num(market, nan=0.0, posinf=8.0, neginf=-8.0)
        tx = torch.nan_to_num(tx, nan=0.0, posinf=8.0, neginf=-8.0)
        order = torch.nan_to_num(order, nan=0.0, posinf=8.0, neginf=-8.0)
        pred = model(market, tx, order)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).detach().float().cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds)


def make_train_indices(months):
    if TRAIN_END_MONTH >= VALID_START_MONTH:
        raise ValueError("TRAIN_END_MONTH must be earlier than VALID_START_MONTH.")
    train_idx = np.where(months <= TRAIN_END_MONTH)[0]
    valid_idx = np.where(
        (months >= VALID_START_MONTH)
        & (months <= VALID_END_MONTH)
        & (months != 66)
    )[0]
    if MAX_TRAIN_SAMPLES > 0 and train_idx.size > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(SEED)
        train_idx = rng.choice(train_idx, MAX_TRAIN_SAMPLES, replace=False)
    return np.sort(train_idx), valid_idx


def write_submission(ids, pred):
    with open(OUT_CSV, "w") as f:
        f.write("sample_id,prediction\n")
        for sid, p in zip(ids, pred):
            f.write(f"{int(sid)},{float(p)}\n")
    print(f"saved {OUT_CSV}, rows={len(ids):,}", flush=True)


def _unused_dev_main():
    require_runtime()
    seed_everything(SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"DATA={DATA}\nWORK_DIR={WORK_DIR}\nCACHE_DIR={CACHE_DIR}\n"
        f"train_end_month={TRAIN_END_MONTH}, valid_months={VALID_START_MONTH}-{VALID_END_MONTH}, "
        f"market_len={MARKET_LEN}, flow_len={FLOW_LEN}, batch_size={BATCH_SIZE}, "
        f"max_train_samples={MAX_TRAIN_SAMPLES}, epochs={EPOCHS}",
        flush=True,
    )

    train_ids, months, target = read_train_label()
    build_all_grids("train", train_ids)

    train_arrays = load_grid("train", train_ids.size)
    train_idx, valid_idx = make_train_indices(months)
    print(f"train_idx={train_idx.size:,}, valid_idx={valid_idx.size:,}", flush=True)

    norm = {
        name: {
            "mean": np.asarray(values["mean"], dtype=np.float32),
            "std": np.asarray(values["std"], dtype=np.float32),
        }
        for name, values in BUNDLED_STREAM_NORM.items()
    }
    save_norm(norm)
    print("using bundled train-fold stream normalization", flush=True)

    target_scale = float(np.std(target[train_idx]))
    print(f"target_scale={target_scale:.8f}", flush=True)
    train_ds = GridDataset(train_arrays, train_idx, norm, target=target, target_scale=target_scale)
    valid_ds = GridDataset(train_arrays, valid_idx, norm, target=target, target_scale=target_scale)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCnnModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    best_cos = -1e9
    best_epoch = 0
    best_path = WORK_DIR / "best_transformer_cnn.pt"
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        pred_valid_scaled = predict(model, valid_loader, device)
        pred_valid = pred_valid_scaled * target_scale
        score = cosine_np(pred_valid, target[valid_idx])
        print(f"epoch={epoch} loss={loss:.6f} valid_cos={score:.6f} time={time.time() - t0:.1f}s", flush=True)
        if score > best_cos:
            best_cos = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "score": best_cos, "target_scale": target_scale, "epoch": best_epoch}, best_path)
            print(f"  saved best: {best_cos:.6f}", flush=True)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    target_scale = float(ckpt["target_scale"])
    best_epoch = int(ckpt["epoch"])
    pred_valid = predict(model, valid_loader, device) * target_scale
    final_score = cosine_np(pred_valid, target[valid_idx])
    pd.DataFrame({
        "sample_id": train_ids[valid_idx],
        "month": months[valid_idx],
        "target": target[valid_idx],
        "prediction": pred_valid,
    }).to_csv(WORK_DIR / "validation_predictions.csv", index=False)
    result = {
        "experiment": "multistream_factorized_transformer_dev",
        "train_months": f"0-{TRAIN_END_MONTH}",
        "purged_months": f"{TRAIN_END_MONTH + 1}-{VALID_START_MONTH - 1}",
        "validation_months": f"{VALID_START_MONTH}-{VALID_END_MONTH}",
        "excluded_validation_months": [66],
        "train_rows": int(train_idx.size),
        "validation_rows": int(valid_idx.size),
        "epochs": EPOCHS,
        "status": "complete",
        "visible_gpu_count": int(_VISIBLE_GPU_COUNT),
        "gpu_names": _VISIBLE_GPU_NAMES,
        "best_epoch": best_epoch,
        "best_cosine": float(final_score),
        "candidate_prediction_file": "validation_predictions.csv",
        "checkpoint_file": "best_transformer_cnn.pt",
        "loss": "0.35 SmoothL1 + 0.65 centered cosine",
    }
    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(WORK_DIR / "score_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    score_only = {
        "best_cosine": float(final_score),
        "best_epoch": best_epoch,
    }
    with open(WORK_DIR / "score_only.json", "w") as f:
        json.dump(score_only, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)



# Joint static branch and strict two-GPU execution.
STATIC_DATA = None
for _candidate in (
    Path(os.environ.get("STATIC_DATA_DIR", "/kaggle/input/mscapital-relative319-dev")),
    Path("/kaggle/input/datasets/zwq1037/mscapital-relative319-dev"),
    Path("/kaggle/input/zwq1037/mscapital-relative319-dev"),
):
    if (
        (_candidate / "features.npy").exists()
        and (_candidate / "sample_ids.npy").exists()
        and (_candidate / "feature_columns.json").exists()
    ):
        STATIC_DATA = _candidate
        break
if STATIC_DATA is None:
    for _features_path in Path("/kaggle/input").rglob("features.npy"):
        _candidate = _features_path.parent
        if (
            (_candidate / "sample_ids.npy").exists()
            and (_candidate / "feature_columns.json").exists()
        ):
            STATIC_DATA = _candidate
            break
if STATIC_DATA is None:
    raise FileNotFoundError("Relative319 static feature dataset is not attached")
print(f"STATIC_DATA={STATIC_DATA}", flush=True)
STATIC_FEATURE_COUNT = 379
REQUIRE_TWO_GPUS = os.environ.get("REQUIRE_TWO_GPUS", "1") == "1"
_VISIBLE_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
_VISIBLE_GPU_NAMES = [
    torch.cuda.get_device_name(index)
    for index in range(_VISIBLE_GPU_COUNT)
]
print(
    f"startup_visible_gpu_count={_VISIBLE_GPU_COUNT}, "
    f"gpu_names={_VISIBLE_GPU_NAMES}",
    flush=True,
)
if REQUIRE_TWO_GPUS and _VISIBLE_GPU_COUNT < 2:
    raise RuntimeError(
        f"Two GPUs required, but only {_VISIBLE_GPU_COUNT} visible. "
        "Select Kaggle GPU T4 x2."
    )
_BASE_STATIC = np.load(STATIC_DATA / "features.npy", mmap_mode="r")
_STATIC_IDS = np.load(STATIC_DATA / "sample_ids.npy", mmap_mode="r")
_EXPECTED_IDS, _STATIC_MONTHS, _ = read_train_label()
if _BASE_STATIC.shape != (_EXPECTED_IDS.size, 319):
    raise AssertionError(f"Unexpected base static shape: {_BASE_STATIC.shape}")
if not np.array_equal(np.asarray(_STATIC_IDS), _EXPECTED_IDS):
    raise AssertionError("Relative319 sample_id order differs from labels.")

_ORDER_PATH = next(iter(Path("/kaggle/input").rglob("train_order_quote_position_features.feather")), None)
if _ORDER_PATH is None:
    raise FileNotFoundError("Order quote position20 input is not attached")
_order_table = feather.read_table(str(_ORDER_PATH), memory_map=True)
_ORDER_IDS = _order_table["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
_ORDER_COLUMNS = [name for name in _order_table.column_names if name != "sample_id"]
_ORDER20 = np.column_stack([
    _order_table[name].to_numpy(zero_copy_only=False) for name in _ORDER_COLUMNS
]).astype(np.float32, copy=False)
if len(_ORDER_COLUMNS) != 20 or not np.array_equal(_ORDER_IDS, _EXPECTED_IDS):
    raise AssertionError("Order quote position20 input does not align")

_TOP_FEATURES = [
    "trade_volume_imbalance_20", "last60_ask_price_1_end", "ofi_1_20_mean",
    "trade_count_imbalance_60", "new_order_volume_imbalance_10",
    "net_order_pressure_30", "ofi_1_last", "microprice_displacement_last",
    "bid_price_1_last", "ask_price_1_last", "trade_volume_imbalance_60",
    "net_order_pressure_10", "last_trade_seconds_before_predict",
    "microprice_displacement_20_mean", "spread_1_mean",
    "cancel_order_pressure_10", "last60_spread_2_mean",
    "last60_bid_price_2_end", "bid_price_2_last", "bid_price_2_std",
]
_BASE_COLUMNS = json.loads((STATIC_DATA / "feature_columns.json").read_text())
_top_indices = [_BASE_COLUMNS.index(name) for name in _TOP_FEATURES]
_XS40 = np.zeros((len(_BASE_STATIC), 40), dtype=np.float32)
for _month in np.unique(_STATIC_MONTHS):
    _rows = np.flatnonzero(_STATIC_MONTHS == _month)
    _values = np.asarray(_BASE_STATIC[_rows][:, _top_indices], dtype=np.float32)
    for _j in range(20):
        _vector = _values[:, _j]
        _finite = np.isfinite(_vector)
        if _finite.any():
            _fv = _vector[_finite]
            _ranks = rankdata(_fv, method="average").astype(np.float32)
            _XS40[_rows[_finite], _j] = 2.0 * _ranks / (len(_fv) + 1.0) - 1.0
            _std = max(float(_fv.std(dtype=np.float64)), 1e-8)
            _XS40[_rows[_finite], 20 + _j] = np.clip(
                (_fv - float(_fv.mean(dtype=np.float64))) / _std, -10.0, 10.0
            )

class _CombinedStaticFeatures:
    def __init__(self, base, xs40, order20):
        self.base = base
        self.xs40 = xs40
        self.order20 = order20
        self.shape = (base.shape[0], 379)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return np.concatenate([
            np.asarray(self.base[index], dtype=np.float32),
            np.asarray(self.xs40[index], dtype=np.float32),
            np.asarray(self.order20[index], dtype=np.float32),
        ], axis=-1)

_STATIC_FEATURES = _CombinedStaticFeatures(_BASE_STATIC, _XS40, _ORDER20)
print(f"static_features={_STATIC_FEATURES.shape} (Relative319 + XS40 + order20)", flush=True)
_STATIC_NORM = None


def _compute_static_norm(features, indices, max_rows=100000, chunk_rows=4096):
    rng = np.random.default_rng(SEED + 101)
    selected = np.asarray(indices)
    if selected.size > max_rows:
        selected = np.sort(rng.choice(selected, max_rows, replace=False))
    total = np.zeros(features.shape[1], dtype=np.float64)
    total_sq = np.zeros(features.shape[1], dtype=np.float64)
    count = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, selected.size, chunk_rows):
        values = np.asarray(
            features[selected[start:start + chunk_rows]],
            dtype=np.float32,
        )
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        total += safe.sum(axis=0)
        total_sq += (safe * safe).sum(axis=0)
        count += finite.sum(axis=0)
    denominator = np.maximum(count, 1.0)
    mean = total / denominator
    variance = np.maximum(total_sq / denominator - mean * mean, 0.0)
    std = np.sqrt(variance)
    mean = np.nan_to_num(
        mean, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std)
    std = np.nan_to_num(
        std, nan=1.0, posinf=1.0, neginf=1.0
    ).astype(np.float32)
    return {"mean": mean, "std": std}


class GridDataset(Dataset):
    def __init__(
        self,
        arrays,
        indices,
        norm,
        target=None,
        target_scale=1.0,
    ):
        global _STATIC_NORM
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm = norm
        self.target = target
        self.target_scale = target_scale
        if _STATIC_NORM is None:
            _STATIC_NORM = _compute_static_norm(
                _STATIC_FEATURES,
                self.indices,
            )

    def __len__(self):
        return self.indices.size

    def _norm_sequence(self, name, values):
        pad = np.abs(values).sum(axis=-1) == 0
        values = (
            values - self.norm[name]["mean"]
        ) / self.norm[name]["std"]
        values = np.clip(values, -8.0, 8.0)
        values = np.nan_to_num(
            values, nan=0.0, posinf=8.0, neginf=-8.0
        )
        values[pad] = 0.0
        return values.astype(np.float32, copy=False)

    def __getitem__(self, item):
        index = self.indices[item]
        market = self._norm_sequence(
            "market",
            np.asarray(self.arrays["market"][index], dtype=np.float32),
        )
        transaction = self._norm_sequence(
            "tx",
            np.asarray(self.arrays["tx"][index], dtype=np.float32),
        )
        order = self._norm_sequence(
            "order",
            np.asarray(self.arrays["order"][index], dtype=np.float32),
        )
        static = np.asarray(_STATIC_FEATURES[index], dtype=np.float32)
        static = (
            static - _STATIC_NORM["mean"]
        ) / _STATIC_NORM["std"]
        static = np.clip(
            np.nan_to_num(
                static, nan=0.0, posinf=8.0, neginf=-8.0
            ),
            -8.0,
            8.0,
        ).astype(np.float32, copy=False)
        tensors = (
            torch.from_numpy(market),
            torch.from_numpy(transaction),
            torch.from_numpy(order),
            torch.from_numpy(static),
        )
        if self.target is None:
            return tensors
        scaled_target = np.float32(np.nan_to_num(
            self.target[index] / self.target_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ))
        return tensors + (torch.tensor(scaled_target),)


class _JointMultiStreamStaticModel(nn.Module):
    def __init__(
        self,
        d_model=96,
        nhead=4,
        nlayers=2,
        dropout=0.15,
    ):
        super().__init__()
        self.market_proj = nn.Linear(
            len(MARKET_FEATURES), d_model
        )
        self.tx_proj = nn.Linear(len(TX_FEATURES), d_model)
        self.order_proj = nn.Linear(
            len(ORDER_FEATURES), d_model
        )
        self.market_pos = nn.Parameter(
            torch.zeros(1, MARKET_LEN, d_model)
        )
        self.tx_pos = nn.Parameter(
            torch.zeros(1, FLOW_LEN, d_model)
        )
        self.order_pos = nn.Parameter(
            torch.zeros(1, FLOW_LEN, d_model)
        )
        self.table_embed = nn.Parameter(
            torch.zeros(3, d_model)
        )
        self.conv1 = ConvBlock(
            d_model, kernel=5, dropout=dropout
        )
        self.conv2 = ConvBlock(
            d_model, kernel=3, dropout=dropout
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=nlayers
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, d_model),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.market_pos, std=0.02)
        nn.init.normal_(self.tx_pos, std=0.02)
        nn.init.normal_(self.order_pos, std=0.02)
        nn.init.normal_(self.table_embed, std=0.02)

    def forward(self, market, transaction, order, static):
        market_tokens = (
            self.market_proj(market)
            + self.market_pos
            + self.table_embed[0]
        )
        transaction_tokens = (
            self.tx_proj(transaction)
            + self.tx_pos
            + self.table_embed[1]
        )
        order_tokens = (
            self.order_proj(order)
            + self.order_pos
            + self.table_embed[2]
        )
        tokens = torch.cat(
            [market_tokens, transaction_tokens, order_tokens],
            dim=1,
        )
        padding_mask = torch.cat([
            market.abs().sum(-1) == 0,
            transaction.abs().sum(-1) == 0,
            order.abs().sum(-1) == 0,
        ], dim=1)
        tokens = self.conv1(tokens)
        tokens = self.conv2(tokens)
        tokens = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )
        attention_logits = self.attention(tokens).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            padding_mask, -1e4
        )
        attention_weights = torch.softmax(
            attention_logits, dim=1
        ).unsqueeze(-1)
        sequence_pooled = (
            tokens * attention_weights
        ).sum(dim=1)
        static_pooled = self.static_adapter(static)
        return self.head(torch.cat(
            [sequence_pooled, static_pooled],
            dim=1,
        )).squeeze(-1)


class TransformerCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        gpu_count = (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        )
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(gpu_count)
        ]
        print(
            f"visible_gpu_count={gpu_count}, "
            f"gpu_names={gpu_names}",
            flush=True,
        )
        if REQUIRE_TWO_GPUS and gpu_count < 2:
            raise RuntimeError(
                f"Two GPUs required, but only {gpu_count} visible. "
                "Select Kaggle GPU T4 x2."
            )
        core = _JointMultiStreamStaticModel()
        if gpu_count > 1:
            self.parallel = nn.DataParallel(
                core,
                device_ids=list(range(gpu_count)),
            )
        else:
            self.parallel = core
        print(
            f"data_parallel={gpu_count > 1}, "
            f"device_ids={list(range(gpu_count))}",
            flush=True,
        )

    def forward(self, market, transaction, order, static):
        return self.parallel(
            market,
            transaction,
            order,
            static,
        )


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    row_count = 0
    for batch in loader:
        market, transaction, order, static, target = batch
        market = market.to(device, non_blocking=True).contiguous()
        transaction = transaction.to(device, non_blocking=True).contiguous()
        order = order.to(device, non_blocking=True).contiguous()
        static = static.to(device, non_blocking=True).contiguous()
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=USE_AMP and device.type == "cuda"
        ):
            prediction = model(
                market, transaction, order, static
            )
            prediction = torch.nan_to_num(
                prediction,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            loss = (
                0.35 * F.smooth_l1_loss(
                    prediction, target
                )
                + 0.65 * cosine_loss(
                    prediction, target
                )
            )
        if not torch.isfinite(loss):
            print("skip non-finite loss batch", flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = target.numel()
        total += float(loss.detach().cpu()) * count
        row_count += count
    return total / max(row_count, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    output = []
    for batch in loader:
        if len(batch) == 5:
            market, transaction, order, static, _ = batch
        else:
            market, transaction, order, static = batch
        market = market.to(device, non_blocking=True).contiguous()
        transaction = transaction.to(device, non_blocking=True).contiguous()
        order = order.to(device, non_blocking=True).contiguous()
        static = static.to(device, non_blocking=True).contiguous()
        inference_model = model
        if hasattr(model, "parallel") and isinstance(model.parallel, nn.DataParallel):
            inference_model = model.parallel.module
        prediction = inference_model(
            market, transaction, order, static
        )
        prediction = torch.nan_to_num(
            prediction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        output.append(
            prediction.detach().float().cpu().numpy()
        )
    return np.concatenate(output)



class _FactorizedStreamEncoder(nn.Module):
    def __init__(self, input_dim, length, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, length, d_model))
        self.conv5 = ConvBlock(d_model, kernel=5, dropout=dropout)
        self.conv3 = ConvBlock(d_model, kernel=3, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values):
        values = values.contiguous()
        padding_mask = values.abs().sum(-1) == 0
        tokens = self.projection(values) + self.position
        tokens = self.conv5(tokens)
        tokens = self.conv3(tokens)
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        logits = self.attention(tokens).squeeze(-1).masked_fill(padding_mask, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)


class _JointMultiStreamStaticModel(nn.Module):
    """Encode each table on its own time axis, then mix four summary tokens."""
    def __init__(self, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.market = _FactorizedStreamEncoder(
            len(MARKET_FEATURES), MARKET_LEN, d_model, nhead, nlayers, dropout
        )
        self.transaction = _FactorizedStreamEncoder(
            len(TX_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout
        )
        self.order = _FactorizedStreamEncoder(
            len(ORDER_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192), nn.SiLU(), nn.LayerNorm(192),
            nn.Dropout(dropout), nn.Linear(192, d_model), nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.source_attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(self, market, transaction, order, static):
        summaries = torch.stack([
            self.market(market),
            self.transaction(transaction),
            self.order(order),
            self.static_adapter(static),
        ], dim=1)
        summaries = self.fusion(summaries + self.source_embedding)
        weights = torch.softmax(self.source_attention(summaries).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head((summaries * weights).sum(dim=1)).squeeze(-1)

def _build_xs40(base_features):
    """Build the same 40 cross-sectional features; test is one anonymous group."""
    output = np.zeros((len(base_features), 40), dtype=np.float32)
    values = np.asarray(base_features[:, _top_indices], dtype=np.float32)
    for column_index in range(20):
        vector = values[:, column_index]
        finite = np.isfinite(vector)
        if finite.any():
            finite_values = vector[finite]
            ranks = rankdata(finite_values, method="average").astype(np.float32)
            output[finite, column_index] = 2.0 * ranks / (len(finite_values) + 1.0) - 1.0
            std = max(float(finite_values.std(dtype=np.float64)), 1e-8)
            output[finite, 20 + column_index] = np.clip(
                (finite_values - float(finite_values.mean(dtype=np.float64))) / std,
                -10.0, 10.0,
            )
    return output


def _load_test_static(test_ids):
    test_base_path = STATIC_DATA / "test_features.npy"
    test_ids_path = STATIC_DATA / "test_sample_ids.npy"
    if not test_base_path.exists() or not test_ids_path.exists():
        test_base_path = next(iter(Path("/kaggle/input").rglob("test_features.npy")), None)
        if test_base_path is None:
            raise FileNotFoundError("Relative319 test_features.npy is not attached")
        test_ids_path = test_base_path.parent / "test_sample_ids.npy"
        if not test_ids_path.exists():
            raise FileNotFoundError("Relative319 test_sample_ids.npy is not attached")
    test_base = np.load(test_base_path, mmap_mode="r")
    static_test_ids = np.load(test_ids_path, mmap_mode="r")
    if test_base.shape != (test_ids.size, 319):
        raise AssertionError(f"Unexpected test Relative319 shape: {test_base.shape}")
    if not np.array_equal(np.asarray(static_test_ids), test_ids):
        raise AssertionError("Test Relative319 IDs differ from submission template")
    order_path = next(iter(Path("/kaggle/input").rglob("test_order_quote_position_features.feather")), None)
    if order_path is None:
        raise FileNotFoundError("Test Order20 input is not attached")
    order_table = feather.read_table(str(order_path), memory_map=True)
    order_ids = order_table["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    test_order20 = np.column_stack([
        order_table[name].to_numpy(zero_copy_only=False) for name in _ORDER_COLUMNS
    ]).astype(np.float32, copy=False)
    if not np.array_equal(order_ids, test_ids):
        raise AssertionError("Test Order20 IDs differ from submission template")
    return _CombinedStaticFeatures(test_base, _build_xs40(test_base), test_order20)


def _unit_prediction(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)


def main():
    global _STATIC_FEATURES, _STATIC_NORM
    require_runtime()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    fold_ends = (52, 57, 62)
    print(
        f"temporal3fold={fold_ends}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"visible_gpu_count={_VISIBLE_GPU_COUNT}, gpu_names={_VISIBLE_GPU_NAMES}",
        flush=True,
    )
    train_ids, months, target = read_train_label()
    train_arrays = load_grid("train", train_ids.size)
    train_static = _STATIC_FEATURES
    norm = {
        name: {
            "mean": np.asarray(values["mean"], dtype=np.float32),
            "std": np.asarray(values["std"], dtype=np.float32),
        }
        for name, values in BUNDLED_STREAM_NORM.items()
    }
    save_norm(norm)

    test_ids = read_submission_ids()
    test_arrays = load_grid("test", test_ids.size)
    test_static = _load_test_static(test_ids)
    test_idx = np.arange(test_ids.size, dtype=np.int64)
    device = torch.device("cuda")
    fold_predictions = []
    fold_results = []

    for fold_end in fold_ends:
        seed_everything(SEED)
        _STATIC_FEATURES = train_static
        train_idx = np.flatnonzero(months <= fold_end)
        target_scale = float(np.std(target[train_idx]))
        _STATIC_NORM = _compute_static_norm(_STATIC_FEATURES, train_idx)
        train_ds = GridDataset(
            train_arrays, train_idx, norm, target=target,
            target_scale=target_scale,
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        )
        model = TransformerCnnModel().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(EPOCHS, 1)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        logs = []
        for epoch in range(1, EPOCHS + 1):
            started = time.time()
            loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "loss": float(loss),
                "seconds": time.time() - started,
            }
            logs.append(row)
            print(
                f"fold_end={fold_end} epoch={epoch}/{EPOCHS} "
                f"loss={loss:.6f} time={row['seconds']:.1f}s",
                flush=True,
            )
        model_path = WORK_DIR / f"factorized_transformer_fold_end{fold_end}_e4.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "target_scale": target_scale,
                "epochs": EPOCHS,
                "seed": SEED,
                "fold_end": fold_end,
            },
            model_path,
        )
        del train_loader, train_ds
        gc.collect()
        torch.cuda.empty_cache()

        _STATIC_FEATURES = test_static
        test_ds = GridDataset(
            test_arrays, test_idx, norm, target=None,
            target_scale=target_scale,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        prediction = predict(model, test_loader, device) * target_scale
        if prediction.shape != (test_ids.size,) or not np.isfinite(prediction).all():
            raise AssertionError(f"Invalid fold_end={fold_end} test predictions")
        fold_path = WORK_DIR / f"factorized_transformer379_fold_end{fold_end}_e4.csv"
        pd.DataFrame(
            {"sample_id": test_ids, "prediction": prediction}
        ).to_csv(fold_path, index=False)
        fold_predictions.append(_unit_prediction(prediction))
        fold_results.append(
            {
                "fold_end": fold_end,
                "train_rows": int(train_idx.size),
                "target_scale": target_scale,
                "prediction_mean": float(prediction.mean()),
                "prediction_std": float(prediction.std()),
                "prediction_file": fold_path.name,
                "checkpoint_file": model_path.name,
                "training_log": logs,
            }
        )
        del model, optimizer, scheduler, scaler, test_loader, test_ds
        gc.collect()
        torch.cuda.empty_cache()

    ensemble = np.mean(fold_predictions, axis=0)
    submission_path = WORK_DIR / "factorized_transformer379_temporal3fold_e4.csv"
    pd.DataFrame(
        {"sample_id": test_ids, "prediction": ensemble}
    ).to_csv(submission_path, index=False)
    result = {
        "experiment": "multistream_factorized_transformer_temporal3fold",
        "status": "complete",
        "fold_ends": list(fold_ends),
        "seeds": [SEED],
        "model_count": len(fold_ends),
        "test_rows": int(test_ids.size),
        "epochs_per_model": EPOCHS,
        "visible_gpu_count": int(_VISIBLE_GPU_COUNT),
        "gpu_names": _VISIBLE_GPU_NAMES,
        "loss": "0.35 SmoothL1 + 0.65 centered cosine",
        "ensemble_method": "equal average after per-fold z-standardization",
        "prediction_mean": float(ensemble.mean()),
        "prediction_std": float(ensemble.std()),
        "prediction_file": submission_path.name,
        "folds": fold_results,
        "competition_submission_status": "prepared_not_submitted",
    }
    for name in ("result.json", "score_only.json"):
        (WORK_DIR / name).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

