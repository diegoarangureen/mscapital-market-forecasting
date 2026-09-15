#!/bin/bash
# Idempotent stage driver: run the next pending stage. Safe to call repeatedly.
cd /tmp/work
export MALLOC_ARENA_MAX=1
if [ ! -f /tmp/work/mlp246v2_val.npy ] && ! pgrep -f mlp246_v2.py >/dev/null; then
  setsid nohup python3 -u /tmp/work/mlp246_v2.py >> /tmp/work/mlp246v2.log 2>&1 < /dev/null &
  echo "launched mlp246_v2"; exit 0
fi
if [ -f /tmp/work/mlp246v2_val.npy ] && [ ! -f /tmp/work/X20AB_DONE ] && ! pgrep -f 'gbm_x20_ab.py' >/dev/null; then
  if [ ! -f /tmp/work/X20_train.npy ] && ! pgrep -f features20.py >/dev/null; then
    setsid nohup python3 -u /tmp/work/features20.py >> /tmp/work/x20.log 2>&1 < /dev/null &
    echo "launched features20"; exit 0
  fi
  if [ -f /tmp/work/X20_train.npy ]; then
    setsid nohup bash -c 'python3 -u /tmp/work/gbm_x20_ab.py; touch /tmp/work/X20AB_DONE' >> /tmp/work/x20.log 2>&1 < /dev/null &
    echo "launched gbm_x20_ab"; exit 0
  fi
fi
echo "nothing to do"
