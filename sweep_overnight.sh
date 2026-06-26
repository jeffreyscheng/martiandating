#!/bin/bash
cd /home/jefcheng/dev/martiandating
DIR=/home/jefcheng/dev/martiandating
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "=== Overnight Sweep: $(date) ==="

submit() {
    local TAG=$1; shift
    local FLAGS="$@"

    sbatch --job-name="sweep_${TAG}" --partition=gnode-interactive \
        --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=64G \
        --time=04:00:00 --output="${DIR}/results/slurm_${TAG}_%j.log" \
        --wrap="cd ${DIR} && export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax && export PYTHONUNBUFFERED=1 && \
        stdbuf -oL ${PY} run_gpu.py ${FLAGS} --tag ${TAG}_gpu0 \
        2>&1 | tee ${DIR}/results/${TAG}_gpu0.log && \
        cp results/fit_${TAG}_gpu0.pkl results/fit_${TAG}_combined.pkl 2>/dev/null" 2>&1 | tail -1
    echo "  Submitted $TAG"
}

echo ""
echo "--- Strategy A: Init from D3_gpu ---"
submit A2 --num_domains 2 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0
submit A3 --num_domains 3 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0
submit A4 --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_from results/fit_D3_gpu.pkl --init_mode pkl --warmup_sigma_inflate 1.0

echo ""
echo "--- Strategy B: MAP-first ---"
submit B2 --num_domains 2 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_mode map --warmup_sigma_inflate 1.0
submit B3 --num_domains 3 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_mode map --warmup_sigma_inflate 1.0
submit B4 --num_domains 4 --num_chains 100 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --init_mode map --warmup_sigma_inflate 1.0

echo ""
echo "--- Strategy C: Random init, long warmup ---"
submit C2 --num_domains 2 --num_chains 100 --num_warmup 20000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0
submit C3 --num_domains 3 --num_chains 100 --num_warmup 20000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0
submit C4 --num_domains 4 --num_chains 100 --num_warmup 20000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0

echo ""
echo "--- Strategy D: Random init, few chains ---"
submit D2f --num_domains 2 --num_chains 10 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0
submit D3f --num_domains 3 --num_chains 10 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0
submit D4f --num_domains 4 --num_chains 10 --num_warmup 5000 --steps_per_loop 1000 --num_loops 100 --seed 42 --ea_mode fixed --warmup_sigma_inflate 1.0

echo ""
echo "=== All submitted. Starting monitor loop ==="
echo ""

# Monitor loop
while true; do
    echo "--- $(date) ---"

    # Count completed
    completed=0
    for TAG in A2 A3 A4 B2 B3 B4 C2 C3 C4 D2f D3f D4f; do
        pkl="results/fit_${TAG}_gpu0.pkl"
        if [ -f "$pkl" ] && [ $(stat -c%s "$pkl" 2>/dev/null || echo 0) -gt 1000 ]; then
            completed=$((completed + 1))
        fi
    done
    echo "Completed: $completed/12"

    # Run evaluation on completed ones
    if [ "$completed" -gt 0 ]; then
        $PY evaluate_sweep.py A2 A3 A4 B2 B3 B4 C2 C3 C4 D2f D3f D4f 2>&1 | tee results/sweep_status.txt

        # Check if any passed
        if grep -q "PASSES ALL CRITERIA" results/sweep_status.txt 2>/dev/null; then
            echo ""
            echo "=== SUCCESS! A model passed all criteria ==="
            winner=$(grep "PASSES ALL CRITERIA" results/sweep_status.txt | head -1 | awk '{print $2}')
            echo "Winner: $winner"

            # Generate final plot
            cp results/fit_${winner}_gpu0.pkl results/fit_${winner}_combined.pkl 2>/dev/null
            $PY plot_final.py 2>&1 | grep "$winner"
            echo "Plot: results/${winner}_final.png"
            break
        fi
    fi

    # Check if all jobs finished
    running=$(squeue -u $USER -h | grep sweep_ | wc -l)
    if [ "$running" -eq 0 ] && [ "$completed" -ge 12 ]; then
        echo "All jobs finished, none passed. Final evaluation:"
        $PY evaluate_sweep.py A2 A3 A4 B2 B3 B4 C2 C3 C4 D2f D3f D4f 2>&1 | tee results/sweep_status.txt
        break
    fi

    sleep 600
done

echo "=== Sweep complete: $(date) ==="
