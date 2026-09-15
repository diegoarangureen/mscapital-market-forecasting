#!/bin/bash
cd /tmp/work
export MALLOC_ARENA_MAX=1
R=/tmp/repo/src/recovered
# self-contained (both splits)
for f in features7 features8 features9; do
  X=$(echo $f | sed 's/features/X/')
  if [ -f /tmp/work/${X}_train.npy ] && [ -f /tmp/work/${X}_test.npy ]; then echo "SKIP $f"; continue; fi
  echo "RUN $f $(date -u +%H:%M)"; python3 -u $R/$f.py || { echo "FAIL $f"; exit 1; }
done
# arg-driven (split ns)
for f in features2 features12 features13 features16; do
  X=$(echo $f | sed 's/features/X/')
  for args in "train 1257637" "test 647896"; do
    set -- $args
    if [ -f /tmp/work/${X}_$1.npy ]; then echo "SKIP $f $1"; continue; fi
    echo "RUN $f $1 $(date -u +%H:%M)"; python3 -u $R/$f.py $1 $2 || { echo "FAIL $f $1"; exit 1; }
  done
done
echo "MATRICES_DONE $(date -u +%H:%M)"
python3 -u gbm_xcos_ab.py
echo "XCOS_CHAIN_DONE"
python3 -u mlp246_v2.py
echo "ALL_DONE"
