#!/bin/bash
# Full recovery after sandbox wipe. Credentials are NOT stored here:
# set KGAT (Kaggle token) and GHP (GitHub PAT) first, e.g. from the agent's wake prompt.
# Usage: KGAT=... GHP=... bash recover.sh
set -e
mkdir -p ~/.kaggle /tmp/work /tmp/mscapital && chmod 700 ~/.kaggle
printf '%s' "$KGAT" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
[ -d /tmp/repo ] || git clone --quiet "https://x-access-token:${GHP}@github.com/diegoarangureen/mscapital-market-forecasting.git" /tmp/repo
cd /tmp/repo && git config user.email agent@instinct.com && git config user.name 'Instinct Agent' && git pull -q --rebase || true
pip install -q lightgbm pyarrow pandas torch kagglesdk lz4 zstandard --index-url https://pypi.org/simple
cd /tmp/mscapital && bash /tmp/repo/src/download_data.sh
cp /tmp/repo/src/recovered/{features20.py,gbm_x20_ab.py,gbm_xcos_ab.py,mlp246_v2.py,master_rebuild.sh,stage_driver.sh} /tmp/work/
echo RECOVERY_DONE
