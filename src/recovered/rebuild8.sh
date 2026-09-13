#!/bin/bash
cd /tmp/work
set -e
python3 -u collect_fp.py
python3 -u eval_x2base.py
echo REBUILD_ALL_DONE
