#!/bin/bash
# Download the competition data with a Kaggle API Bearer token
# (new-style KGAT_ token; the legacy CLI is too old for it).
# NOTE: slashes in nested paths must be %2F-encoded or the API 404s.
TOK=$(cat ~/.kaggle/access_token)
BASE='https://www.kaggle.com/api/v1/competitions/data/download/ms-capital-real-financial-market-forecasting'
mkdir -p train test
for f in submission.csv train/label.feather train/transaction.feather train/order.feather train/market.feather test/transaction.feather test/order.feather test/market.feather; do
  enc=${f//\//%2F}
  curl -sL -C - --retry 3 -H "Authorization: Bearer $TOK" -o "$f" "$BASE/$enc"
done
