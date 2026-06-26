#!/bin/bash
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Pathfinder warmup sweep: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="pf_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo "--- 3-domain shared Ea + pathfinder ---"
submit PF_3es_r1 --num_domains 3 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --tag PF_3es_r1_gpu0
submit PF_3es_r2 --num_domains 3 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --tag PF_3es_r2_gpu0
submit PF_3es_r3 --num_domains 3 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 100 --seed 271 --ea_mode shared --tag PF_3es_r3_gpu0

echo "--- 4-domain shared Ea + pathfinder ---"
submit PF_4es_r1 --num_domains 4 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --tag PF_4es_r1_gpu0
submit PF_4es_r2 --num_domains 4 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --tag PF_4es_r2_gpu0

echo "--- 6-domain shared Ea + pathfinder ---"
submit PF_6es_r1 --num_domains 6 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --tag PF_6es_r1_gpu0

echo ""
echo "=== 6 jobs submitted: $(date) ==="
