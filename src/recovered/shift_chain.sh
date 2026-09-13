#!/bin/bash
cd /tmp/work
set -e
python3 -u eval_shift.py > /tmp/shift.log 2>&1
python3 -u mlp_shift.py > /tmp/mlpshift.log 2>&1
echo SHIFT_CHAIN_DONE
