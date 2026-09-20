"""Try the official numeric-version download API and verify before loading weights."""
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / 'outputs/factorized_transformer_dev_v7_exact'
api = KaggleApi()
api.authenticate()

def fetch(name):
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = 'zwq1037'
    request.kernel_slug = 'multistream-factorized-transformer-multiwindow-dev'
    request.version_number = 7
    request.file_path = 'factorized_transformer/' + name
    with api.build_kaggle_client() as client:
        redirect = client.kernels.kernels_api_client.download_kernel_output(request)
    response = requests.get(redirect.url, timeout=60)
    response.raise_for_status()
    if len(response.content) > 30 * 1024 * 1024:
        raise RuntimeError('Unexpectedly large artifact')
    return response.content

result_content = fetch('result.json')
result = json.loads(result_content)
assert result['experiment'] == 'multistream_factorized_transformer_dev'
assert result['train_months'] == '0-59' and result['purged_months'] == '60-61'
assert abs(result['best_cosine'] - 0.1553845145) < 1e-8
destination.mkdir(parents=True, exist_ok=True)
(destination / 'result.json').write_bytes(result_content)
for name in ('best_transformer_cnn.pt', 'norm_v2.json'):
    content = fetch(name)
    (destination / name).write_bytes(content)
    print(json.dumps({'downloaded': name, 'bytes': len(content)}))
print('verified numeric version-7 dev model recovered')
