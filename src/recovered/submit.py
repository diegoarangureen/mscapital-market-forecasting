import os, sys, json, time
os.environ['KAGGLE_API_TOKEN'] = open(os.path.expanduser('~/.kaggle/access_token')).read().strip()
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest, ApiListSubmissionsRequest)
import requests
COMP='ms-capital-real-financial-market-forecasting'
path='/tmp/work/submission_gbm.csv'; size=os.path.getsize(path)
c=KaggleClient()
req=ApiStartSubmissionUploadRequest()
req.competition_name=COMP; req.file_name='submission.csv'; req.content_length=size
req.last_modified_epoch_seconds=int(os.path.getmtime(path))
resp=c.competitions.competition_api_client.start_submission_upload(req)
r=requests.put(resp.create_url, data=open(path,'rb').read(),
               headers={'Content-Type':'application/octet-stream','Content-Length':str(size)})
print('PUT', r.status_code, flush=True)
sr=ApiCreateSubmissionRequest()
sr.competition_name=COMP
sr.blob_file_tokens=resp.token
sr.submission_description='GBM v3: LightGBM on ~64 microstructure+market-bar features (multi-horizon returns, volume windows, slopes, book/trade/order flow), no-data samples zeroed, val cosine TBD'
print(c.competitions.competition_api_client.create_submission(sr), flush=True)
for i in range(10):
    time.sleep(20)
    rq=ApiListSubmissionsRequest(); rq.competition_name=COMP
    subs=c.competitions.competition_api_client.list_submissions(rq).submissions
    s=subs[0]
    print('poll', i, str(s)[:400], flush=True)
    if 'complete' in str(s).lower() or 'error' in str(s).lower():
        break