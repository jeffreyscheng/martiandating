#!/bin/bash
#SBATCH --job-name=mdd_C1
#SBATCH --partition=gnode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/jefcheng/dev/martiandating/results/slurm_C1_%j.log

cd /home/jefcheng/dev/martiandating
export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax

echo "Starting C1 on $(hostname) at $(date)"

for gpu_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu_id python run_gpu.py \
        --num_chains 100 --num_warmup 5000 \
        --steps_per_loop 1000 --num_loops 100 \
        --tag C1_gpu${gpu_id} --seed $((42 + gpu_id)) \
        --num_domains 1 --ea_mode fixed --continuous --n_bins 20 \
        > /home/jefcheng/dev/martiandating/results/C1_gpu${gpu_id}.log 2>&1 &
done
wait
echo "All GPUs done for C1 at $(date)"

python -c "
import pickle, numpy as np
all_s = {}
for g in range(8):
    try:
        d = pickle.load(open('/home/jefcheng/dev/martiandating/results/fit_C1_gpu' + str(g) + '.pkl','rb'))
        for k in d['samples']:
            all_s.setdefault(k,[]).append(d['samples'][k])
    except Exception as e: print(f'GPU {g}: {e}')
if all_s:
    combined = {k: np.concatenate(v, axis=1) for k, v in all_s.items()}
    pickle.dump({'samples': combined, 'tag': 'C1'}, open('/home/jefcheng/dev/martiandating/results/fit_C1_combined.pkl', 'wb'))
    print(f'C1: {combined[list(combined.keys())[0]].shape}')
"
