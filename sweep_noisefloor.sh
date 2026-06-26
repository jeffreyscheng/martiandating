#!/bin/bash
# Noise floor sweep: fixed σ_floor × domain count × Ea mode
# All with pathfinder warmup, continuous adaptation, 8xH100
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Noise Floor Sweep: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"
    sbatch --job-name="nf_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

COMMON="--num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 20"

echo "--- 3 domains × noise floors × shared Ea ---"
submit NF3_f3_es  --num_domains 3 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.0003 --tag NF3_f3_es_gpu0
submit NF3_f10_es --num_domains 3 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.001  --tag NF3_f10_es_gpu0
submit NF3_f30_es --num_domains 3 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.003  --tag NF3_f30_es_gpu0

echo "--- 4 domains × noise floors × shared Ea ---"
submit NF4_f3_es  --num_domains 4 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.0003 --tag NF4_f3_es_gpu0
submit NF4_f10_es --num_domains 4 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.001  --tag NF4_f10_es_gpu0
submit NF4_f30_es --num_domains 4 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.003  --tag NF4_f30_es_gpu0

echo "--- 5 domains × noise floors × shared Ea ---"
submit NF5_f10_es --num_domains 5 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.001  --tag NF5_f10_es_gpu0
submit NF5_f30_es --num_domains 5 $COMMON --seed 42 --ea_mode shared --sigma_floor 0.003  --tag NF5_f30_es_gpu0

echo "--- 3 domains × noise floors × FIXED Ea (does floor make fixed Ea work?) ---"
submit NF3_f10_ef --num_domains 3 $COMMON --seed 42 --ea_mode fixed --sigma_floor 0.001  --tag NF3_f10_ef_gpu0
submit NF3_f30_ef --num_domains 3 $COMMON --seed 42 --ea_mode fixed --sigma_floor 0.003  --tag NF3_f30_ef_gpu0

echo "--- Reproducibility: 3-dom, floor=0.001, shared Ea, different seeds ---"
submit NF3_f10_s2 --num_domains 3 $COMMON --seed 137 --ea_mode shared --sigma_floor 0.001 --tag NF3_f10_s2_gpu0
submit NF3_f10_s3 --num_domains 3 $COMMON --seed 271 --ea_mode shared --sigma_floor 0.001 --tag NF3_f10_s3_gpu0

echo ""
echo "=== 12 jobs submitted: $(date) ==="
