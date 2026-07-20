#!/bin/bash
set -u

cd /home/ubuntu/dev/martiandating
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/martiandating-jax-cache

while systemctl --user is-active --quiet codex-flex-sensitivity-queue; do
    sleep 15
done

tag=sensitivity_sigma20
mkdir -p "results/flexible/${tag}"
.venv-a100/bin/python run_flexible_mdd.py \
    --chains 256 --warmup 0 --settle-draws 500 \
    --draws 2000 --chunk-size 250 \
    --bins 28 --grid-min -4 --grid-max 14 \
    --kernel-length 1.0 --flex-scale 1.5 --relative-sigma 0.20 \
    --affine-precondition --fixed-step-size 0.05 --init-scale 0.25 \
    --seed 20260728 --kill-gates --strict-final --tag "${tag}" \
    > "results/flexible/${tag}/runner.log" 2>&1
