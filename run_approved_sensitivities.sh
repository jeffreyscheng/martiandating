#!/bin/bash
set -u

cd /home/ubuntu/dev/martiandating
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/martiandating-jax-cache

while systemctl --user is-active --quiet codex-flex-rough; do
    sleep 15
done

run_sensitivity() {
    local tag="$1"
    shift
    mkdir -p "results/flexible/${tag}"
    echo "$(date -u +%FT%TZ) starting ${tag}" >> results/flexible/sensitivity_queue.log
    .venv-a100/bin/python run_flexible_mdd.py \
        --chains 256 --warmup 0 --settle-draws 500 \
        --draws 2000 --chunk-size 250 \
        --grid-min -4 --grid-max 14 \
        --affine-precondition --init-scale 0.25 \
        --kill-gates --strict-final --tag "${tag}" "$@" \
        > "results/flexible/${tag}/runner.log" 2>&1
    local status=$?
    echo "$(date -u +%FT%TZ) finished ${tag} status=${status}" >> results/flexible/sensitivity_queue.log
}

run_sensitivity sensitivity_sigma15 \
    --bins 28 --kernel-length 1.0 --flex-scale 1.5 \
    --relative-sigma 0.15 --fixed-step-size 0.04 --seed 20260725

run_sensitivity sensitivity_grid20 \
    --bins 20 --kernel-length 1.0 --flex-scale 1.5 \
    --relative-sigma 0.10 --fixed-step-size 0.05 --seed 20260726

run_sensitivity sensitivity_grid36 \
    --bins 36 --kernel-length 1.0 --flex-scale 1.5 \
    --relative-sigma 0.10 --fixed-step-size 0.03 --seed 20260727
