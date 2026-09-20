"""Download only verified version-7 outputs through the official Kaggle SDK."""
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'outputs/factorized_transformer_dev_v7_exact'
api = KaggleApi()
api.authenticate()
request = ApiListKernelSessionOutputRequest()
request.user_name = 'zwq1037'
request.kernel_slug = 'multistream-factorized-transformer-multiwindow-dev'
request.version_label = '7'
request.page_size = 200
with api.build_kaggle_client() as client:
    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
files = {item.file_name: item for item in response.files or []}
print(json.dumps({'requested_version': 7, 'filenames': list(files)}, indent=2))
expected = 'factorized_transformer/result.json'
if expected not in files:
    raise RuntimeError('Requested version did not return expected dev result; no weights downloaded')
result_response = requests.get(files[expected].url, timeout=60)
result_response.raise_for_status()
result = result_response.json()
assert result['experiment'] == 'multistream_factorized_transformer_dev'
assert result['train_months'] == '0-59' and result['purged_months'] == '60-61'
assert abs(result['best_cosine'] - 0.1553845145) < 1e-8
DESTINATION.mkdir(parents=True, exist_ok=True)
(DESTINATION / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
for name in ('best_transformer_cnn.pt', 'norm_v2.json'):
    key = 'factorized_transformer/' + name
    if key not in files:
        raise RuntimeError('Missing version-7 artifact: ' + name)
    with requests.get(files[key].url, stream=True, timeout=60) as download:
        download.raise_for_status()
        temporary = DESTINATION / (name + '.tmp')
        size = 0
        with temporary.open('wb') as handle:
            for block in download.iter_content(chunk_size=1024 * 1024):
                size += len(block)
                if size > 30 * 1024 * 1024:
                    raise RuntimeError('Unexpectedly large dev model artifact')
                handle.write(block)
        temporary.replace(DESTINATION / name)
    print(json.dumps({'downloaded': name, 'bytes': size}))
print('verified version-7 dev model recovered')
