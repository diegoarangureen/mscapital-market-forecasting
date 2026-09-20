import ast
from pathlib import Path
import torch
from torch import nn
path=Path(__file__).resolve().parents[1]/"data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v26_linear_sequence.py"
tree=ast.parse(path.read_text(encoding="utf-8"))
nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {"_LinearStreamReadout","_JointMultiStreamStaticModel"}]
ns={"torch":torch,"nn":nn,"MARKET_LEN":200,"FLOW_LEN":60,"MARKET_FEATURES":list(range(11)),"TX_FEATURES":list(range(7)),"ORDER_FEATURES":list(range(10)),"STATIC_FEATURE_COUNT":379}
exec(compile(ast.Module(nodes,[]),str(path),"exec"),ns)
model=ns["_JointMultiStreamStaticModel"]()
inputs=[torch.randn(8,200,11),torch.randn(8,60,7),torch.randn(8,60,10),torch.randn(8,379)]
y=model(*inputs);loss=y.square().mean();loss.backward()
assert y.shape==(8,) and torch.isfinite(y).all() and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print({"shape":list(y.shape),"params":sum(p.numel() for p in model.parameters()),"finite":True})

