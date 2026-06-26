#!/bin/bash
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== High-domain runs: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="f2_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

submit F2_10es_r1 --num_domains 10 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_10es_r1_gpu0
submit F2_10es_r2 --num_domains 10 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_10es_r2_gpu0
submit F2_15es_r1 --num_domains 15 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_15es_r1_gpu0
submit F2_20es_r1 --num_domains 20 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_20es_r1_gpu0

echo "=== Done: $(date) ==="
