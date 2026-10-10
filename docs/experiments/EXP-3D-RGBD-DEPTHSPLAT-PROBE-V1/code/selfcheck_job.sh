#!/bin/bash
# Camera-convention self-check of DepthSplat on the DEV probe: three variants in sequence on the assigned GPU.
set -u
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_dev/manifest_dev.json
for variant in none w2c pixel_center; do
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_selfcheck.py "$manifest" DEV --variant "$variant" --queries \
    > ~/cv_work/depthsplat/selfcheck_$variant.log 2>&1 || exit 1
done
