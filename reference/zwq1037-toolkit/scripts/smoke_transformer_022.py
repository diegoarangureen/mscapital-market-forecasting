"""One-batch structural smoke test for EXP-TRANSFORMER-022."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import exp_transformer_022_month_macro_cosine as experiment


torch.set_num_threads(2)
ids = np.load(experiment.CACHE_META / "sample_ids.npy", mmap_mode="r")
months = np.load(experiment.CACHE_META / "months.npy", mmap_mode="r")
target = np.load(experiment.CACHE_META / "targets.npy", mmap_mode="r")
static = np.load(experiment.STATIC_PATH, mmap_mode="r")
arrays = experiment.open_arrays(len(ids))
train_indices = np.flatnonzero(months <= 59)
target_scale = float(np.std(target[train_indices]))
namespace, norm = experiment.extract_runtime(static)
namespace["_STATIC_NORM"] = namespace["_compute_static_norm"](static, train_indices)
dataset = namespace["GridDataset"](arrays, train_indices, norm, target=target, target_scale=target_scale)
train_months = np.asarray(months[train_indices])
positions = [np.flatnonzero(train_months == month) for month in range(60)]
sampler = experiment.TwoMonthBalancedBatchSampler(positions, 256, 1, experiment.SEED)
sampler.set_epoch(1)
batch = next(iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0)))
batch_months = train_months[np.asarray(next(iter(sampler)))]
if len(np.unique(batch_months[:128])) != 1 or len(np.unique(batch_months[128:])) != 1:
    raise AssertionError("Batch halves are not month-pure")
device = torch.device("cuda")
model = namespace["_JointMultiStreamStaticModel"]().to(device)
inputs = [value.to(device).contiguous() for value in batch[:4]]
y = batch[4].to(device)
prediction = model(*inputs)
loss = 0.35 * F.smooth_l1_loss(prediction, y) + 0.325 * (
    namespace["cosine_loss"](prediction[:128].float(), y[:128].float())
    + namespace["cosine_loss"](prediction[128:].float(), y[128:].float())
)
loss.backward()
if not torch.isfinite(loss) or not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
    raise RuntimeError("Nonfinite smoke result")
print({"months": [int(batch_months[0]), int(batch_months[128])], "loss": float(loss.detach()), "prediction_shape": list(prediction.shape)})
