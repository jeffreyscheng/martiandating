#!/bin/bash
# Follow-up sweep 2: More domains + continuous distribution
# Goal: improve median Arrhenius fit at 525-587°C transition zone
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Follow-up Sweep 2: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="f2_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo "--- More domains with shared Ea (smoother LRD->HRD transition) ---"
submit F2_5es_r1 --num_domains 5 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_5es_r1_gpu0
submit F2_5es_r2 --num_domains 5 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_5es_r2_gpu0
submit F2_6es_r1 --num_domains 6 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_6es_r1_gpu0
submit F2_6es_r2 --num_domains 6 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_6es_r2_gpu0
submit F2_8es_r1 --num_domains 8 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_8es_r1_gpu0
submit F2_8es_r2 --num_domains 8 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_8es_r2_gpu0

echo "--- Continuous domain distribution (Gaussian over logD) ---"
submit F2_cont_r1 --continuous --n_bins 20 --num_domains 1 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_cont_r1_gpu0
submit F2_cont_r2 --continuous --n_bins 20 --num_domains 1 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_cont_r2_gpu0
submit F2_cont30 --continuous --n_bins 30 --num_domains 1 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag F2_cont30_gpu0
submit F2_contf --continuous --n_bins 20 --num_domains 1 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0 --tag F2_contf_gpu0

echo ""
echo "=== 10 jobs submitted: $(date) ==="
