#!/bin/bash
while ! grep -q XCOS_DONE /tmp/work/xcos.log 2>/dev/null; do sleep 60; done
export MALLOC_ARENA_MAX=1
exec python3 /tmp/work/mlp246_v2.py
