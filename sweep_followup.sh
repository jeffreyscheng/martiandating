#!/bin/bash
# Follow-up sweep: Ea freedom + random init + continuous adaptation variants
# Key finding: random init → R-hat 8.7, shared/per-domain Ea → 18/18 fit
# Goal: combine both for R-hat < 10 AND 18/18 Arrhenius
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Follow-up Sweep: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="fu_${TAG}" --partition=gnode-hi-pri \
        --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=256G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && stdbuf -oL ${PY} run_gpu.py ${FLAGS} 2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo "--- Shared Ea + random init (the key combo) ---"
submit FU_es_r1 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_es_r1_gpu0
submit FU_es_r2 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_es_r2_gpu0
submit FU_es_r3 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 271 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_es_r3_gpu0

echo "--- Per-domain Ea + random init ---"
submit FU_ep_r1 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode per_domain --warmup_sigma_inflate 1.0 --tag FU_ep_r1_gpu0
submit FU_ep_r2 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode per_domain --warmup_sigma_inflate 1.0 --tag FU_ep_r2_gpu0

echo "--- Shared Ea + random init + diagonal mass (AW4d had good step sizes) ---"
submit FU_esd_r1 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --diagonal_mass --warmup_sigma_inflate 1.0 --tag FU_esd_r1_gpu0
submit FU_esd_r2 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --diagonal_mass --warmup_sigma_inflate 1.0 --tag FU_esd_r2_gpu0

echo "--- 3-domain shared Ea + random init (fewer params = easier convergence) ---"
submit FU_3es_r1 --num_domains 3 --num_chains 99 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_3es_r1_gpu0
submit FU_3es_r2 --num_domains 3 --num_chains 99 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 137 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_3es_r2_gpu0

echo "--- Shared Ea + random init + longer runs (200 loops for better R-hat) ---"
submit FU_es_long --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 200 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_es_long_gpu0

echo "--- Shared Ea + random init + more warmup ---"
submit FU_es_w20k --num_domains 4 --num_chains 104 --num_warmup 20000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_es_w20k_gpu0

echo "--- Shared Ea + random init + 2000 steps/chunk ---"
submit FU_es_L --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 2000 --num_loops 50 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_es_L_gpu0

echo "--- 5-domain shared Ea + random init ---"
submit FU_5es_r1 --num_domains 5 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode shared --warmup_sigma_inflate 1.0 --tag FU_5es_r1_gpu0

echo "--- Per-domain Ea + random init + diagonal mass ---"
submit FU_epd_r1 --num_domains 4 --num_chains 104 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode per_domain --diagonal_mass --warmup_sigma_inflate 1.0 --tag FU_epd_r1_gpu0

echo ""
echo "=== 15 jobs submitted: $(date) ==="
