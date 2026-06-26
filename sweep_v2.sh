#!/bin/bash
# Sweep v2: 8 diverse experiments to find a well-fit, well-converged model
# Experiments: MAP-Laplace, Emcee, Nested, EaTdep, AW6, PerEa-MAP, TemperedNUTS, ReParam
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Sweep v2: $(date) ==="

submit() {
    local TAG=$1; shift
    local SCRIPT=$1; shift
    local FLAGS="$@"

    sbatch --job-name="sv2_${TAG}" --partition=gnode-interactive \
        --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=64G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && \
        stdbuf -oL ${PY} ${SCRIPT} ${FLAGS} \
        2>&1 | tee ${DIR}/results/${TAG}_gpu0.log" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo ""
echo "--- #2: MAP-Laplace (diagnostic: does the MAP fit all 18 points?) ---"
submit MAP_L4 run_map_laplace.py --tag MAP_L4 --num_domains 4

echo ""
echo "--- #3: Emcee (affine-invariant ensemble sampler, no mass matrix) ---"
submit EM4 run_emcee.py --tag EM4 --num_domains 4 --num_walkers 200 --num_steps 50000

echo ""
echo "--- #4: Nested sampling (handles multimodality, gives evidence) ---"
submit NS4 run_nested.py --tag NS4 --num_domains 4

echo ""
echo "--- #5: Temperature-dependent Ea + continuous adaptation ---"
submit AW4t run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 \
    --steps_per_loop 1000 --num_loops 100 --seed 42 \
    --ea_mode t_dependent --init_from results/fit_D3_gpu.pkl --init_mode pkl \
    --warmup_sigma_inflate 1.0 --tag AW4t_gpu0

echo ""
echo "--- #6: 6 domains + continuous adaptation ---"
submit AW6 run_gpu.py --num_domains 6 --num_chains 100 --num_warmup 5000 \
    --steps_per_loop 1000 --num_loops 100 --seed 42 \
    --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl \
    --warmup_sigma_inflate 1.0 --tag AW6_gpu0

echo ""
echo "--- #7: Per-domain Ea + improved MAP init + continuous adaptation ---"
submit PM4e run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 \
    --steps_per_loop 1000 --num_loops 100 --seed 42 \
    --ea_mode per_domain --init_mode map \
    --warmup_sigma_inflate 1.0 --tag PM4e_gpu0

echo ""
echo "--- #8: Tempered NUTS (T=5, flatter posterior for mixing) ---"
submit TN4 run_gpu.py --num_domains 4 --num_chains 100 --num_warmup 5000 \
    --steps_per_loop 1000 --num_loops 100 --seed 42 \
    --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl \
    --warmup_sigma_inflate 1.0 --temperature 5.0 --tag TN4_gpu0

echo ""
echo "--- #9: Reparameterized (whitened param space from D3_gpu covariance) ---"
submit RP4 run_reparam.py --tag RP4 --num_chains 100 --num_loops 100

echo ""
echo "=== All 8 submitted. Monitor with: ==="
echo "  watch -n 60 '$PY evaluate_sweep.py MAP_L4 EM4 NS4 AW4t AW6 PM4e TN4 RP4'"
echo "=== $(date) ==="
