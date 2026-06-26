#!/bin/bash
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Fine Noise Floor Sweep: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"
    sbatch --job-name="nff_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=01:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

COMMON="--num_domains 3 --num_chains 104 --num_warmup 500 --steps_per_loop 1000 --num_loops 20 --seed 42 --ea_mode shared"

submit NFF_03  $COMMON --sigma_floor 0.00003  --tag NFF_03_gpu0
submit NFF_05  $COMMON --sigma_floor 0.00005  --tag NFF_05_gpu0
submit NFF_08  $COMMON --sigma_floor 0.00008  --tag NFF_08_gpu0
submit NFF_10  $COMMON --sigma_floor 0.0001   --tag NFF_10_gpu0
submit NFF_15  $COMMON --sigma_floor 0.00015  --tag NFF_15_gpu0
submit NFF_20  $COMMON --sigma_floor 0.0002   --tag NFF_20_gpu0
submit NFF_25  $COMMON --sigma_floor 0.00025  --tag NFF_25_gpu0
submit NFF_30  $COMMON --sigma_floor 0.0003   --tag NFF_30_gpu0

echo ""
echo "=== 8 jobs submitted: $(date) ==="
echo "=== ETA: ~23 min from now ==="
