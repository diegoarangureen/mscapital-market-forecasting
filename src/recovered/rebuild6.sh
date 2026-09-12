#!/bin/bash
cd /tmp/work
TOK=$(cat ~/.kaggle/access_token)
BASE='https://www.kaggle.com/api/v1/competitions/data/download/ms-capital-real-financial-market-forecasting'
FILES="train/label.feather train/transaction.feather train/order.feather train/market.feather test/transaction.feather test/order.feather test/market.feather"
# wait for the running download to finish
while pgrep -f download_data.sh > /dev/null; do sleep 20; done
# integrity: every feather must start with ARROW1; redownload (fresh) until good
for round in 1 2 3 4 5; do
  bad=0
  for f in $FILES; do
    p="/tmp/mscapital/$f"
    if [ ! -s "$p" ] || [ "$(head -c 6 "$p")" != "ARROW1" ]; then
      echo "redownload $f (round $round)"; rm -f "$p"
      enc=${f//\//%2F}
      curl -sL --retry 3 -H "Authorization: Bearer $TOK" -o "$p" "$BASE/$enc"
      bad=1
    fi
  done
  [ "$bad" = "0" ] && break
done
ok=1
for f in $FILES; do p="/tmp/mscapital/$f"; [ "$(head -c 6 "$p" 2>/dev/null)" != "ARROW1" ] && { echo "STILL_BAD $f"; ok=0; }; done
[ "$ok" = "1" ] && echo DATA_OK || { echo DATA_FAIL; exit 1; }
set -e
python3 -u part_txord.py train 1257637
python3 -u part_txord.py test 647896
python3 -u combine.py train 1257637
python3 -u combine.py test 647896
python3 -u verify_x2.py
python3 -u features6.py train 1257637
python3 -u features6.py test 647896
python3 -u collect_fp.py
python3 -u eval_x2base.py
echo REBUILD_ALL_DONE
