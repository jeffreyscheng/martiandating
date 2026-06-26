#!/bin/bash
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== AW4 Variants (8xH100): $(date) ==="

submit() {
    local TAG=$1; shift
    local SCRIPT=$1; shift
    local FLAGS="$@"

    sbatch --job-name="aw_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} ${SCRIPT} ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo "--- Init variants ---"
submit AW4r  run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0 --tag AW4r_gpu0
submit AW4s2 run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --tag AW4s2_gpu0

echo "--- Domain count variants ---"
submit AW3   run_gpu.py --num_domains 3 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --tag AW3_gpu0
submit AW5   run_gpu.py --num_domains 5 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --tag AW5_gpu0

echo "--- Ea variants ---"
submit AW4ep run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode per_domain --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --tag AW4ep_gpu0
submit AW4es run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --tag AW4es_gpu0

echo "--- Sampler tuning ---"
submit AW4L  run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 2000 --num_loops 50 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --tag AW4L_gpu0
submit AW4d  run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --diagonal_mass --tag AW4d_gpu0

echo "--- Remaining sweep v2 ---"
submit TN4  run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --temperature 5.0 --tag TN4_gpu0
submit RP4  run_reparam.py --tag RP4 --num_chains 100 --num_loops 100
submit NS4  run_nested.py --tag NS4 --num_domains 4

echo ""
echo "=== All submitted: $(date) ==="
