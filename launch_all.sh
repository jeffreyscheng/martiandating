#!/bin/bash
# Launch one 8xH100 SLURM job per model variant
cd /home/jefcheng/dev/martiandating

CHAINS=1000
WARMUP=5000
STEPS_PER_LOOP=1000
LOOPS=100

submit_model() {
    local TAG=$1
    shift
    local EXTRA_FLAGS="$@"

    sbatch --job-name="mdd_${TAG}" \
           --partition=gnode \
           --nodes=1 \
           --gpus-per-node=8 \
           --cpus-per-task=32 \
           --time=02:00:00 \
           --output="results/slurm_${TAG}_%j.log" \
           --wrap="
cd /home/jefcheng/dev/martiandating
export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax
echo \"=== Model ${TAG} on \$(hostname) at \$(date) ===\"
for gpu_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=\$gpu_id python run_gpu.py \
        --num_chains ${CHAINS} --num_warmup ${WARMUP} \
        --steps_per_loop ${STEPS_PER_LOOP} --num_loops ${LOOPS} \
        --tag ${TAG}_gpu\${gpu_id} --seed \$((42 + gpu_id)) \
        ${EXTRA_FLAGS} \
        > results/${TAG}_gpu\${gpu_id}.log 2>&1 &
done
wait
echo \"All GPUs done for ${TAG} at \$(date)\"

python -c \"
import pickle, numpy as np
all_samples = {}
for gpu_id in range(8):
    try:
        d = pickle.load(open(f'results/fit_${TAG}_gpu{gpu_id}.pkl','rb'))
        for key in d['samples']:
            if key not in all_samples: all_samples[key] = []
            all_samples[key].append(d['samples'][key])
    except Exception as e:
        print(f'GPU {gpu_id}: {e}')
combined = {k: np.concatenate(v, axis=1) for k, v in all_samples.items()}
pickle.dump({'samples': combined, 'tag': '${TAG}'}, open('results/fit_${TAG}_combined.pkl', 'wb'))
print(f'${TAG}: combined {combined[list(combined.keys())[0]].shape}')
\"
"
    echo "Submitted ${TAG}"
}

echo "=== Launching model sweep ==="

# Discrete domains, fixed Ea=117
submit_model D1 --num_domains 1 --ea_mode fixed
submit_model D2 --num_domains 2 --ea_mode fixed
submit_model D3 --num_domains 3 --ea_mode fixed
submit_model D4 --num_domains 4 --ea_mode fixed

# Discrete domains, shared Ea sampled
submit_model D5 --num_domains 2 --ea_mode shared
submit_model D6 --num_domains 3 --ea_mode shared

# Discrete domains, per-domain Ea
submit_model D7 --num_domains 2 --ea_mode per_domain
submit_model D8 --num_domains 3 --ea_mode per_domain

# Continuous distribution, fixed Ea
submit_model C1 --num_domains 1 --ea_mode fixed --continuous --n_bins 20
submit_model C3 --num_domains 1 --ea_mode shared --continuous --n_bins 20

# With learned noise
submit_model N1 --num_domains 2 --ea_mode per_domain --learn_sigma
submit_model N2 --num_domains 3 --ea_mode fixed --learn_sigma

echo ""
echo "=== All submitted! Monitor with: squeue -u $USER ==="
