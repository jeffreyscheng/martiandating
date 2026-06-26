"""Evaluate sweep results against success criteria."""
import os, sys, pickle, glob
import numpy as np
import pandas as pd

R_gas = 8.314
Ea_fixed = 117.0

# Load data
df = pd.read_csv('data/nakhla1_parsed_fitted.csv')
total = df['39Ar'].sum()
df['dF'] = df['39Ar'] / total
df['F'] = df['dF'].cumsum()
df['T_K'] = df['Temp'] + 273.15

def yF(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6 / np.sqrt(np.pi)
        return ((a - np.sqrt(max(a * a - 12 * F, 0))) / 6) ** 2
    return -(1 / np.pi ** 2) * np.log(max((1 - F) * np.pi ** 2 / 6, 1e-30))

df['y'] = df['F'].apply(yF)
filt = df.dropna(subset=['seconds_per_extraction_step'])
fitted = filt[filt['Temp'] >= 350].copy()
F_before = filt[filt['Temp'] < 350]['dF'].sum()
fitted_F_cum = F_before + np.cumsum(fitted['dF'].values)
tK = fitted['T_K'].values
dt_arr = fitted['seconds_per_extraction_step'].values
y_before = yF(F_before)
y_obs = np.array([yF(f) for f in fitted_F_cum])
dy_obs = np.diff(y_obs, prepend=y_before)
D_obs = dy_obs / dt_arr
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)
frac_fitted = fitted_F_cum[-1] - F_before
n_data = len(tK)
temps_C = fitted['Temp'].values

def fr_np(y):
    return max(6*np.sqrt(y/np.pi)-3*y, 0) if y < 0.3 else min(1-(6/np.pi**2)*np.exp(-np.pi**2*y), 1)

def split_rhat(samples, discard_frac=0.5):
    """Compute split R-hat, discarding first discard_frac of samples as burn-in."""
    T, C = samples.shape
    start = int(T * discard_frac)
    post_burnin = samples[start:]
    T2 = post_burnin.shape[0]
    half = T2 // 2
    if half < 2: return np.nan
    chains = np.concatenate([post_burnin[:half].T, post_burnin[half:half*2].T], axis=0)
    n, m = half, chains.shape[0]
    chain_means = chains.mean(axis=1)
    B = n / (m - 1) * np.sum((chain_means - chain_means.mean()) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    return np.sqrt(((n - 1) / n * W + B / n) / W) if W > 0 else np.nan

def compute_effective_arrhenius(logD_vals, phi_vals, nd, Ea_val=None):
    """Compute effective ln(D/rho2) for one posterior sample."""
    Ea_use = Ea_val if Ea_val is not None else Ea_fixed
    Dr2 = np.exp(logD_vals)[:, None]
    yi = Dr2 * np.exp(-Ea_use * 1e3 / (R_gas * tK[None, :])) * dt_arr
    yc = np.cumsum(yi, axis=1)
    Fc = np.array([[fr_np(y) for y in yc[d]] for d in range(nd)])
    dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((nd, 1)))
    dF_mix = phi_vals @ dF_dom
    Fm = np.cumsum(dF_mix)
    if Fm[-1] < 1e-30: return None
    Fs = F_before + Fm / Fm[-1] * frac_fitted
    ye = np.array([yF(f) for f in Fs])
    dye = np.diff(ye, prepend=y_before)
    De = dye / dt_arr
    De[De <= 0] = np.nan
    v = np.log(De)
    if np.all(np.isfinite(v)) and np.all(v > -30):
        return v
    return None

def evaluate(tag):
    """Evaluate a single model run."""
    for path in [f'results/fit_{tag}_combined.pkl', f'results/fit_{tag}_gpu0.pkl', f'results/fit_{tag}.pkl']:
        if os.path.exists(path) and os.path.getsize(path) > 100:
            break
    else:
        return None

    d = pickle.load(open(path, 'rb'))
    fs = d['samples']
    if 'logD0_r2' not in fs:
        return None

    nd = fs['logD0_r2'].shape[2]
    nc = fs['logD0_r2'].shape[1]
    nl = fs['logD0_r2'].shape[0]

    result = {'tag': tag, 'nd': nd, 'nc': nc, 'nl': nl}

    # 1. Arrhenius coverage
    all_lnD = []
    for c in range(min(nc, 500)):
        logD = fs['logD0_r2'][-1, c]
        phi = fs['phi'][-1, c] if 'phi' in fs else np.ones(nd) / nd
        Ea_c = fs['Ea_shared'][-1, c] if 'Ea_shared' in fs else None
        v = compute_effective_arrhenius(logD, phi, nd, Ea_val=Ea_c)
        if v is not None:
            all_lnD.append(v)

    if len(all_lnD) < 10:
        result['arrhenius_pass'] = False
        result['arrhenius_detail'] = 'Too few valid samples'
        return result

    all_lnD = np.array(all_lnD)
    lo = np.percentile(all_lnD, 2.5, axis=0)
    hi = np.percentile(all_lnD, 97.5, axis=0)

    covered = np.array([(lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i]) if np.isfinite(lnD_obs[i]) else True
                        for i in range(n_data)])
    n_covered = covered.sum()
    result['arrhenius_covered'] = int(n_covered)
    result['arrhenius_total'] = n_data
    result['arrhenius_pass'] = (n_covered == n_data)
    if not result['arrhenius_pass']:
        missed = [f"{temps_C[i]}°C" for i in range(n_data) if not covered[i]]
        result['arrhenius_detail'] = f'Missed: {", ".join(missed)}'
    else:
        result['arrhenius_detail'] = 'All covered'

    # 2. R-hat
    rhats = {}
    for key in ['logD0_r2', 'phi']:
        if key in fs and fs[key].ndim == 3:
            for dim in range(fs[key].shape[2]):
                rhats[f'{key}[{dim}]'] = split_rhat(fs[key][:, :, dim])
    logD_rhats = [v for k, v in rhats.items() if 'logD' in k and np.isfinite(v)]
    result['max_rhat'] = max(logD_rhats) if logD_rhats else np.nan
    result['rhat_pass'] = result['max_rhat'] < 10 if np.isfinite(result['max_rhat']) else False

    # 3. Domain structure
    phi_median = np.median(fs['phi'][-1], axis=0) if 'phi' in fs else np.ones(nd) / nd
    n_active = np.sum(phi_median > 0.05)
    phi_init_base = [0.5, 0.25, 0.125, 0.0625]
    phi_init = np.array((phi_init_base * ((nd // len(phi_init_base)) + 1))[:nd])
    phi_moved = np.max(np.abs(phi_median - phi_init)) > 0.02
    result['n_active_domains'] = int(n_active)
    result['phi_moved'] = phi_moved
    result['domain_pass'] = (n_active >= 2) and phi_moved
    result['phi_median'] = phi_median.tolist()

    # 4. Physical parameters
    logD_median = np.median(fs['logD0_r2'][-1], axis=0)
    in_range = np.all((logD_median > 3) & (logD_median < 12))
    not_collapsed = (logD_median.max() - logD_median.min()) > 0.5
    result['logD_median'] = logD_median.tolist()
    result['params_pass'] = in_range and not_collapsed

    # 5. Phi movement
    phi_stds = np.std(fs['phi'][-1], axis=0) if 'phi' in fs else np.zeros(nd)
    n_moving = np.sum(phi_stds > 0.01)
    result['phi_stds'] = phi_stds.tolist()
    result['phi_move_pass'] = n_moving >= 2

    # Overall
    result['ALL_PASS'] = all([
        result['arrhenius_pass'],
        result['rhat_pass'],
        result['domain_pass'],
        result['params_pass'],
        result['phi_move_pass'],
    ])

    return result

# Evaluate all runs
tags = sys.argv[1:] if len(sys.argv) > 1 else [
    f.replace('results/fit_', '').replace('_combined.pkl', '').replace('_gpu0.pkl', '').replace('.pkl', '')
    for f in sorted(glob.glob('results/fit_*.pkl'))
    if 'test' not in f and 'multitest' not in f and 'quicktest' not in f and 'slurm' not in f
]
# Deduplicate
tags = list(dict.fromkeys(tags))

print(f"{'Tag':>10s} | {'Arr':>5s} | {'Rhat':>8s} | {'Doms':>4s} | {'Params':>6s} | {'PhiMv':>5s} | {'PASS':>4s} | Details")
print("-" * 90)

for tag in tags:
    r = evaluate(tag)
    if r is None:
        print(f"{tag:>10s} | {'N/A':>5s} | {'N/A':>8s} | {'N/A':>4s} | {'N/A':>6s} | {'N/A':>5s} | {'N/A':>4s} |")
        continue

    arr = f"{r['arrhenius_covered']}/{r['arrhenius_total']}" if 'arrhenius_covered' in r else 'N/A'
    rhat = f"{r['max_rhat']:.1f}" if np.isfinite(r.get('max_rhat', np.nan)) else 'N/A'
    doms = str(r.get('n_active_domains', '?'))
    params = 'OK' if r.get('params_pass') else 'FAIL'
    phimv = 'OK' if r.get('phi_move_pass') else 'FAIL'
    passed = 'YES' if r.get('ALL_PASS') else 'no'
    detail = r.get('arrhenius_detail', '')

    print(f"{tag:>10s} | {arr:>5s} | {rhat:>8s} | {doms:>4s} | {params:>6s} | {phimv:>5s} | {passed:>4s} | {detail}")

    if r.get('ALL_PASS'):
        print(f"  *** {tag} PASSES ALL CRITERIA ***")
        print(f"  logD: {r['logD_median']}")
        print(f"  phi:  {r['phi_median']}")
