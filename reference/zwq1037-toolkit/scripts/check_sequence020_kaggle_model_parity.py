"""Check the generated portable model against the local model on CPU."""
import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from subsecond_event_transformer_model import make_model

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v21_subsecond_events_dev.py'
tree = ast.parse(path.read_text(encoding='utf-8'))
classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
namespace = {'torch': torch, 'nn': nn, 'pack_padded_sequence': pack_padded_sequence,
             'pad_packed_sequence': pad_packed_sequence,
             'MARKET_FEATURES': list(range(11)), 'MARKET_LEN': 200,
             'TX_FEATURES': list(range(7)), 'ORDER_FEATURES': list(range(10)),
             'FLOW_LEN': 60, 'STATIC_FEATURE_COUNT': 379}
for name in ('ConvBlock', 'SubsecondEventEncoder', '_FactorizedStreamEncoder', '_JointMultiStreamStaticModel'):
    exec(compile(ast.Module(body=[classes[name]], type_ignores=[]), str(path), 'exec'), namespace)
torch.set_num_threads(1)
torch.manual_seed(2026)
local = make_model().eval()
torch.manual_seed(2026)
portable = namespace['_JointMultiStreamStaticModel']().eval()
assert local.state_dict().keys() == portable.state_dict().keys()
assert all(torch.equal(value, portable.state_dict()[key]) for key, value in local.state_dict().items())
market = torch.randn(3, 200, 11)
market[0] = 0
transaction = torch.randn(3, 129, 10)
order = torch.randn(3, 257, 16)
transaction[:, :, -1] = 0
order[:, :, -1] = 0
transaction[1, :12, -1] = 1
transaction[2, :, -1] = 1
order[1, :39, -1] = 1
order[2, :, -1] = 1
static = torch.randn(3, 379)
inputs = (market, transaction, order, static)
expected, actual = local(*inputs), portable(*inputs)
assert torch.isfinite(actual).all()
assert torch.equal(expected, actual)
expected.square().mean().backward()
actual.square().mean().backward()
for name, parameter in local.named_parameters():
    other = dict(portable.named_parameters())[name]
    if parameter.grad is not None:
        assert torch.isfinite(other.grad).all() and torch.equal(parameter.grad, other.grad)
print('portable_vs_local: parameters, outputs, gradients and empty-stream handling identical')
