#!/bin/bash
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== AW4 Variants Sweep: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="aw_${TAG}" --partition=gnode-interactive \
        --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=64G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} --tag ${TAG}_gpu0 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log && cp results/fit_${TAG}_gpu0.pkl results/fit_${TAG}_combined.pkl 2>/dev/null" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo "--- Init variants ---"
submit AW4r --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0
submit AW4s2 --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0

echo "--- Domain count variants ---"
submit AW3 --num_domains 3 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0
submit AW5 --num_domains 5 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0

echo "--- Ea variants ---"
submit AW4ep --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode per_domain --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0
submit AW4es --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0

echo "--- Sampler tuning ---"
submit AW4L --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 2000 --num_loops 50 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0
submit AW4d --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0 --diagonal_mass

echo ""
echo "=== All submitted: $(date) ==="
