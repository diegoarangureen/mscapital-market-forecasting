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
