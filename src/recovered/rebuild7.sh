#!/bin/bash
cd /tmp/work
set -e
python3 -u combine.py train 1257637
python3 -u combine.py test 647896
python3 -u verify_x2.py
python3 -u features6.py train 1257637
python3 -u features6.py test 647896
python3 -u collect_fp.py
python3 -u eval_x2base.py
echo REBUILD_ALL_DONE
