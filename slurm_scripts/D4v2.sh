#!/bin/bash
#SBATCH --job-name=mdd_D4v2
#SBATCH --partition=gnode-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/jefcheng/dev/martiandating/results/slurm_D4v2_%j.log

cd /home/jefcheng/dev/martiandating
export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax
echo "Starting D4v2 on $(hostname) at $(date)"

for gpu_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu_id /apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python run_gpu.py \
        --num_chains 50 --num_warmup 200 \
        --steps_per_loop 1000 --num_loops 100 \
        --tag D4v2_gpu${gpu_id} --seed $((42 + gpu_id)) \
        --num_domains 4 --ea_mode fixed \
        > /home/jefcheng/dev/martiandating/results/D4v2_gpu${gpu_id}.log 2>&1 &
done
wait
echo "All GPUs done at $(date)"

/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python -c "
import pickle, numpy as np
all_s = {}
for g in range(8):
    try:
        d = pickle.load(open('/home/jefcheng/dev/martiandating/results/fit_D4v2_gpu' + str(g) + '.pkl','rb'))
        for k in d['samples']:
            all_s.setdefault(k,[]).append(d['samples'][k])
    except Exception as e: print(f'GPU {g}: {e}')
if all_s:
    combined = {k: np.concatenate(v, axis=1) for k, v in all_s.items()}
    pickle.dump({'samples': combined, 'tag': 'D4v2'}, open('/home/jefcheng/dev/martiandating/results/fit_D4v2_combined.pkl', 'wb'))
    print(f'D4v2: {combined[list(combined.keys())[0]].shape}')
    if 'phi' in combined:
        for d in range(combined['phi'].shape[2]):
            print(f'  phi[{d}]={np.median(combined[\"phi\"][-1,:,d]):.3f}')
    if 'log_sigma_scale' in combined:
        print(f'  noise={np.exp(np.median(combined[\"log_sigma_scale\"][-1])):.2f}x')
    if 'logD0_r2' in combined:
        for d in range(combined['logD0_r2'].shape[2]):
            print(f'  logD[{d}]={np.median(combined[\"logD0_r2\"][-1,:,d]):.2f}')
"

/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python plot_final.py 2>&1 | grep "D4v2"
