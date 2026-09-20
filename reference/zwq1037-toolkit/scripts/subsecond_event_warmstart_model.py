"""Reuse strictly trained market/static/fusion weights; initialize event GRUs fresh."""
import json
from pathlib import Path
import torch
from subsecond_event_transformer_model import make_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'outputs/factorized_transformer_dev_v7_exact/best_transformer_cnn.pt'


def make_warmstart_model(checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = checkpoint['model']
    canonical = {}
    for name, value in saved.items():
        while name.startswith(('module.', 'parallel.')):
            name = name.split('.', 1)[1]
        if name in canonical:
            raise RuntimeError('Duplicate canonical checkpoint key')
        canonical[name] = value
    model = make_model()
    current = model.state_dict()
    copied = {}
    allowed = ('market.', 'static_adapter.', 'source_embedding', 'fusion.', 'source_attention.', 'head.')
    for name, value in current.items():
        if name.startswith(allowed):
            if name not in canonical or canonical[name].shape != value.shape:
                raise RuntimeError('Missing or mismatched transferable parameter: ' + name)
            copied[name] = canonical[name]
    incompatible = model.load_state_dict(copied, strict=False)
    assert not incompatible.unexpected_keys
    assert all(name.startswith(('transaction.', 'order.')) for name in incompatible.missing_keys)
    metadata = {'checkpoint': str(checkpoint_path), 'source_score': float(checkpoint['score']),
                'source_epoch': int(checkpoint['epoch']), 'source_target_scale': float(checkpoint['target_scale']),
                'copied_tensor_count': len(copied), 'copied_parameter_elements': sum(value.numel() for value in copied.values()),
                'total_parameter_elements': sum(value.numel() for value in current.values()),
                'source_training_window': '0-59', 'allowed_validation_window': '62-70 excluding 66',
                'early_50_59_validation_allowed': False,
                'fresh_branches': ['transaction_event_gru', 'order_event_gru'], 'experiment_started': False}
    return model, metadata


if __name__ == '__main__':
    torch.set_num_threads(1)
    model, metadata = make_warmstart_model()
    path = ROOT / 'outputs/submission_metadata/subsecond_warmstart_prepared_20260917.json'
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
