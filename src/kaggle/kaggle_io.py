"""Minimal Kaggle dataset/kernel IO over kagglesdk (KGAT bearer token)."""
import os, time, requests
os.environ.setdefault('KAGGLE_API_TOKEN', open(os.path.expanduser('~/.kaggle/access_token')).read().strip())
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest, ApiBlobType
from kagglesdk.datasets.types.dataset_api_service import (ApiCreateDatasetRequest, ApiDatasetNewFile,
    ApiCreateDatasetVersionRequest, ApiCreateDatasetVersionRequestBody)

def client():
    return KaggleClient()

def upload_file(c, path):
    r = ApiStartBlobUploadRequest()
    r.type = ApiBlobType.DATASET
    r.name = os.path.basename(path)
    r.content_length = os.path.getsize(path)
    r.last_modified_epoch_seconds = int(os.path.getmtime(path))
    resp = c.blobs.blob_api_client.start_blob_upload(r)
    with open(path, 'rb') as fp:
        put = requests.put(resp.create_url, data=fp, timeout=3600)
    if put.status_code not in (200, 201):
        raise RuntimeError(f'upload {path} failed: {put.status_code} {put.text[:200]}')
    return resp.token

def create_dataset(c, owner, slug, title, file_tokens, private=True):
    r = ApiCreateDatasetRequest()
    r.owner_slug = owner; r.slug = slug; r.title = title
    r.is_private = private; r.license_name = 'Other'
    r.files = []
    for name, tok in file_tokens:
        f = ApiDatasetNewFile(); f.token = str(tok); r.files.append(f)
    return c.datasets.dataset_api_client.create_dataset(r)

def new_version(c, owner, slug, file_tokens, notes):
    body = ApiCreateDatasetVersionRequestBody()
    body.version_notes = notes; body.delete_old_versions = False
    body.files = []
    for name, tok in file_tokens:
        f = ApiDatasetNewFile(); f.token = str(tok); body.files.append(f)
    r = ApiCreateDatasetVersionRequest()
    r.owner_slug = owner; r.dataset_slug = slug; r.body = body
    return c.datasets.dataset_api_client.create_dataset_version(r)

# ---- kernel helpers ----
from kagglesdk.kernels.types.kernels_api_service import (ApiSaveKernelRequest,
    ApiGetKernelSessionStatusRequest, ApiDownloadKernelOutputRequest)

def push_kernel(c, owner, slug, title, script_path, datasets=(), kernels=(), gpu=True, timeout_s=36000, comps=()):
    r = ApiSaveKernelRequest()
    # LESSON (Sep 20): if slug-derived name != title-derived name the kernel wedges server-side
    # (403 on ALL session endpoints, undeletable zombie). Force title == slug.
    r.slug = f'{owner}/{slug}'; r.new_title = slug
    r.text = open(script_path).read()
    r.language = 'python'; r.kernel_type = 'script'
    r.is_private = True; r.enable_gpu = gpu; r.enable_internet = False
    r.session_timeout_seconds = timeout_s
    if datasets: r.dataset_data_sources = list(datasets)
    if kernels: r.kernel_data_sources = list(kernels)
    if comps: r.competition_data_sources = list(comps)
    return c.kernels.kernels_api_client.save_kernel(r)

def kernel_status(c, owner, slug):
    r = ApiGetKernelSessionStatusRequest()
    r.user_name = owner; r.kernel_slug = slug
    resp = c.kernels.kernels_api_client.get_kernel_session_status(r)
    return str(getattr(resp, 'status', resp))

def download_kernel_file(c, owner, slug, file_name, out_dir='/tmp/work'):
    r = ApiDownloadKernelOutputRequest()
    r.owner_slug = owner; r.kernel_slug = slug; r.file_path = file_name
    red = c.kernels.kernels_api_client.download_kernel_output(r)
    url = red.url
    data = requests.get(url, timeout=600)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{slug}_{file_name}')
    open(path, 'wb').write(data.content)
    return path

def leaderboard(c, competition='ms-capital-real-financial-market-forecasting'):
    import io, zipfile, csv, json
    tok = os.environ['KAGGLE_API_TOKEN']
    url = f'https://api.kaggle.com/v1/competitions.CompetitionApiService/DownloadLeaderboard'
    r = requests.post(url, data=json.dumps({'competition_name': competition}),
                      headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json'}, timeout=180)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding='utf-8')))

def gpu_quota():
    tok = os.environ['KAGGLE_API_TOKEN']
    r = requests.post('https://www.kaggle.com/api/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics',
                      headers={'Authorization': f'Bearer {tok}', 'Content-Type':'application/json'}, json={}, timeout=60)
    return r.json()
