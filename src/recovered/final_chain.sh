#!/bin/bash
cd /tmp/work
set -e
python3 -u final_mlp_train.py
python3 -u final_mlp_test.py
