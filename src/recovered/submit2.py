import os, sys, json, time
os.environ['KAGGLE_API_TOKEN'] = open(os.path.expanduser('~/.kaggle/access_token')).read().strip()
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest, ApiListSubmissionsRequest)
import requests
COMP='ms-capital-real-financial-market-forecasting'
path='/tmp/work/submission_blend.csv'; size=os.path.getsize(path)
c=KaggleClient()
# quota check first: list recent submissions
rq=ApiListSubmissionsRequest(); rq.competition_name=COMP
subs=c.competitions.competition_api_client.list_submissions(rq).submissions
print('existing submissions:', len(subs))
for s in subs[:6]:
    print(' -', str(s)[:300].replace('\n',' '))
if '--submit' not in sys.argv:
    print('DRY RUN ONLY - pass --submit to upload'); sys.exit(0)
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
sr.submission_description='GBM v4 + MLP blend: 50/50 unit blend of LightGBM (58 microstructure feats, bag0.7, 3-seed refit 0-70) and MLP (121 feats: X2+bar-series fingerprints+X6, 3-seed refit). val 0.1311 centered, robust on shifted split'
print(c.competitions.competition_api_client.create_submission(sr), flush=True)
for i in range(15):
    time.sleep(20)
    rq=ApiListSubmissionsRequest(); rq.competition_name=COMP
    subs=c.competitions.competition_api_client.list_submissions(rq).submissions
    s=subs[0]
    print('poll', i, str(s)[:500].replace('\n',' '), flush=True)
    if 'complete' in str(s).lower() or 'error' in str(s).lower():
        break
