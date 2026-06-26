#!/bin/bash
#SBATCH --job-name=mdd_N2
#SBATCH --partition=gnode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/jefcheng/dev/martiandating/results/slurm_N2_%j.log

cd /home/jefcheng/dev/martiandating
export JAX_COMPILATION_CACHE_DIR=/home/jefcheng/.cache/jax
PY=/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python

echo "Starting N2 on $(hostname) at $(date)"
$PY -c "import jax; print('Backend:', jax.default_backend())"

for gpu_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu_id $PY run_gpu.py \
        --num_chains 100 --num_warmup 1000 \
        --steps_per_loop 1000 --num_loops 100 \
        --tag N2_gpu${gpu_id} --seed $((42 + gpu_id)) \
        --num_domains 3 --ea_mode fixed --learn_sigma \
        > /home/jefcheng/dev/martiandating/results/N2_gpu${gpu_id}.log 2>&1 &
done
wait
echo "All GPUs done at $(date)"

$PY -c "
import pickle, numpy as np
all_s = {}
for g in range(8):
    try:
        d = pickle.load(open('/home/jefcheng/dev/martiandating/results/fit_N2_gpu' + str(g) + '.pkl','rb'))
        for k in d['samples']:
            all_s.setdefault(k,[]).append(d['samples'][k])
    except Exception as e: print(f'GPU {g}: {e}')
if all_s:
    combined = {k: np.concatenate(v, axis=1) for k, v in all_s.items()}
    pickle.dump({'samples': combined, 'tag': 'N2'}, open('/home/jefcheng/dev/martiandating/results/fit_N2_combined.pkl', 'wb'))
    print(f'N2: {combined[list(combined.keys())[0]].shape}')
"
