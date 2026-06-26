#!/bin/bash
#SBATCH --job-name=mdd_mcmc
#SBATCH --partition=gnode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_%j.log

cd /home/jefcheng/dev/martiandating
export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax

echo "Starting on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2

# 8 GPUs × 1000 chains × 100K steps = 800M chain-steps
for gpu_id in 0 1 2 3 4 5 6 7; do
    echo "[$(date +%H:%M:%S)] Launching GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python run_gpu.py \
        --num_domains 3 \
        --num_chains 1000 \
        --num_warmup 5000 \
        --steps_per_loop 1000 \
        --num_loops 100 \
        --tag D3_gpu${gpu_id} \
        --seed $((42 + gpu_id)) \
        > results/gpu${gpu_id}.log 2>&1 &
done

echo "Launched 8 GPU jobs, waiting..."
wait
echo "All GPUs done at $(date)"

# Combine results
python -c "
import pickle, numpy as np

all_samples = {}
for gpu_id in range(8):
    path = f'results/fit_D3_gpu{gpu_id}.pkl'
    try:
        d = pickle.load(open(path, 'rb'))
        for key in d['samples']:
            if key not in all_samples:
                all_samples[key] = []
            all_samples[key].append(d['samples'][key])
        print(f'GPU {gpu_id}: loaded')
    except Exception as e:
        print(f'GPU {gpu_id}: FAILED - {e}')

combined = {k: np.concatenate(v, axis=1) for k, v in all_samples.items()}
n_loops, n_chains, n_domains = combined['logD0_r2'].shape
print(f'Combined: {n_loops} loops x {n_chains} chains x {n_domains} domains')

def split_rhat(samples):
    T, C = samples.shape
    half = T // 2
    chains = np.concatenate([samples[:half].T, samples[half:half*2].T], axis=0)
    n, m = half, chains.shape[0]
    chain_means = chains.mean(axis=1)
    B = n/(m-1)*np.sum((chain_means - chain_means.mean())**2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    return np.sqrt(((n-1)/n*W + B/n)/W) if W > 0 else np.nan

print()
print('=== Final Convergence ===')
for key in ['logD0_r2', 'phi']:
    for d in range(combined[key].shape[2]):
        rhat = split_rhat(combined[key][:, :, d])
        print(f'  {key}[{d}]: R-hat = {rhat:.4f} {\"OK\" if rhat < 1.1 else \"WARNING\"}')

from scipy.optimize import brentq
R_gas = 8.314
Ea_fixed = 117.0

def max_temp(logD0_r2_val, dur_Ma, max_loss=0.01):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD0_r2_val)
    def f(T_C):
        T_K = T_C + 273.15
        y = D0r2 * np.exp(-Ea_fixed*1e3/(R_gas*T_K)) * dur_s
        F = 6*np.sqrt(y/np.pi)-3*y if y<0.3 else 1-(6/np.pi**2)*np.exp(-np.pi**2*y)
        return np.clip(F,0,1) - max_loss
    try: return brentq(f, -273, 1000, xtol=0.1)
    except: return np.nan

hrd_logD = combined['logD0_r2'][-1, :, 0]  # ordered, [0] is HRD (smallest)
print()
print('=== T-t Constraints (8000 chains) ===')
for dur, label in [(10,'10 My'),(100,'100 My'),(200,'200 My'),(500,'500 My'),(1300,'Isothermal')]:
    temps = np.array([max_temp(ld, dur) for ld in hrd_logD])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med = np.median(valid)
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f'  {label:12s}: {med:6.1f}C  95%CI=[{lo:.1f}, {hi:.1f}]  crosses 0C: {hi > 0}')

pickle.dump({'samples': combined}, open('results/fit_D3_combined.pkl', 'wb'))
print()
print('Saved results/fit_D3_combined.pkl')
"

echo "Job complete at $(date)"
