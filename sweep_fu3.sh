#!/bin/bash
# Follow-up 3: phi reparameterization + Hessian init
# Fix: phi_scale=0.03 makes phi_raw posterior width ~1 (matching logD_raw)
# Fix: hessian_init calibrates initial step size for all parameter scales
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Follow-up 3 (phi fix): $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="f3_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo "--- 3-domain shared Ea + phi_scale=0.03 + hessian_init (main fix) ---"
submit F3_3es_r1 --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --phi_scale 0.03 --hessian_init --tag F3_3es_r1_gpu0
submit F3_3es_r2 --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --phi_scale 0.03 --hessian_init --tag F3_3es_r2_gpu0
submit F3_3es_r3 --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 271 --ea_mode shared --phi_scale 0.03 --hessian_init --tag F3_3es_r3_gpu0

echo "--- 4-domain shared Ea + phi fix ---"
submit F3_4es_r1 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --phi_scale 0.03 --hessian_init --tag F3_4es_r1_gpu0
submit F3_4es_r2 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --phi_scale 0.03 --hessian_init --tag F3_4es_r2_gpu0

echo "--- 6-domain shared Ea + phi fix ---"
submit F3_6es_r1 --num_domains 6 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --phi_scale 0.03 --hessian_init --tag F3_6es_r1_gpu0

echo "--- phi_scale sweep (is 0.03 the right scale?) ---"
submit F3_ps01 --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --phi_scale 0.01 --hessian_init --tag F3_ps01_gpu0
submit F3_ps1  --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --phi_scale 0.1 --hessian_init --tag F3_ps1_gpu0

echo "--- Without hessian_init (test if phi_scale alone is enough) ---"
submit F3_noh --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --phi_scale 0.03 --tag F3_noh_gpu0

echo "--- Baseline: no phi fix, just randomized phi init ---"
submit F3_base --num_domains 3 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --tag F3_base_gpu0

echo ""
echo "=== 10 jobs submitted: $(date) ==="
