"""Multi-start Pathfinder: find modes via L-BFGS, form Gaussian mixture, importance resample."""
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/jefcheng/.cache/jax'
import time, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
import blackjax
import blackjax.vi.pathfinder as pathfinder
from scipy.optimize import brentq
from scipy.stats import multivariate_normal

print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")

parser = argparse.ArgumentParser()
parser.add_argument("--num_domains", type=int, default=3)
parser.add_argument("--num_starts", type=int, default=1000)
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--tag", type=str, default="PFM")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ea_mode", type=str, default="shared", choices=["fixed", "shared"])
parser.add_argument("--cluster_threshold", type=float, default=0.5)
parser.add_argument("--sigma_floor", type=float, default=0.0)
args = parser.parse_args()

num_domains = args.num_domains

# ── Data ──
df = pd.read_csv('data/nakhla1_parsed_fitted.csv')
total_39Ar = df['39Ar'].sum()
df['dF'] = df['39Ar'] / total_39Ar
df['F'] = df['dF'].cumsum()
df['T_K'] = df['Temp'] + 273.15

def y_from_F(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6/np.sqrt(np.pi)
        return ((a - np.sqrt(max(a*a-12*F,0)))/6)**2
    return -(1/np.pi**2)*np.log(max((1-F)*np.pi**2/6,1e-30))

df['y'] = df['F'].apply(y_from_F)
filt = df.dropna(subset=['seconds_per_extraction_step'])
filt = filt[filt['Temp'] >= 350]

temps_K = jnp.array(filt['T_K'].values)
f_obs = jnp.array(filt['dF'].values)
sigma_meas = jnp.array(filt['std_39Ar'].values / total_39Ar + 1e-12)
sigma_obs = jnp.sqrt(sigma_meas**2 + args.sigma_floor**2) if args.sigma_floor > 0 else sigma_meas
dt = jnp.array(filt['seconds_per_extraction_step'].values)
n_data = len(temps_K)
temps_C = filt['Temp'].values

F_before = df[df['Temp'] < 350].dropna(subset=['seconds_per_extraction_step'])['dF'].sum() if len(df[df['Temp'] < 350].dropna(subset=['seconds_per_extraction_step'])) > 0 else 0.0
fitted_dF = filt['dF'].values
fitted_F_cum = F_before + np.cumsum(fitted_dF)
frac_fitted = fitted_F_cum[-1] - F_before
y_before = y_from_F(F_before)
y_obs_arr = np.array([y_from_F(f) for f in fitted_F_cum])
dy_obs = np.diff(y_obs_arr, prepend=y_before)
D_obs = dy_obs / filt['seconds_per_extraction_step'].values
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)

print(f"Data: {n_data} steps, {num_domains} domains")
if args.sigma_floor > 0:
    print(f"Noise floor: {args.sigma_floor:.5f}")

# ── Model ──
R_gas = 8.314
Ea_fixed_val = 117.0

def stick_breaking(phi_raw):
    probs = jax.nn.sigmoid(phi_raw)
    remaining = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(1.0 - probs)])
    return jnp.concatenate([probs * remaining[:-1], remaining[-1:]])

def fractional_release(y):
    F_short = 6 * jnp.sqrt(y / jnp.pi) - 3 * y
    F_long = 1 - (6 / jnp.pi**2) * jnp.exp(-jnp.pi**2 * y)
    return jnp.clip(jnp.where(y < 0.3, F_short, F_long), 0.0, 1.0)

def log_prob(params):
    logD_raw = params["logD_raw"]
    logD = jnp.cumsum(jax.nn.softplus(logD_raw)) + logD_raw[0]
    phi_raw = params["phi_raw"]
    phi = stick_breaking(phi_raw)
    D_r2 = jnp.exp(logD)[:, None]

    if args.ea_mode == "shared":
        Ea_val = params["Ea_shared"]
    else:
        Ea_val = Ea_fixed_val

    T = temps_K[None, :]
    Ea_broadcast = jnp.full((num_domains, 1), Ea_val)
    y_inc = D_r2 * jnp.exp(-Ea_broadcast * 1e3 / (R_gas * T)) * dt
    y_cum = jnp.cumsum(y_inc, axis=1)
    Fcum = fractional_release(y_cum)
    dF = jnp.diff(Fcum, axis=1, prepend=jnp.zeros((num_domains, 1)))
    dF_mix = phi @ dF
    F_final_mix = phi @ Fcum[..., -1]
    f_pred = dF_mix / (F_final_mix + 1e-30)

    ll = jnp.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2
                 - jnp.log(sigma_obs) - 0.5 * jnp.log(2 * jnp.pi))
    # Tighter priors based on feldspar diffusion literature
    # logD in [3, 10] for feldspars; Ea in [100, 140] kJ/mol
    prior = jnp.sum(-0.5 * (logD - 6.0)**2 / 1.5**2)
    if args.ea_mode == "shared":
        prior += -0.5 * (Ea_val - 117.0)**2 / 10.0**2
    return ll + prior

def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    # logD_raw → logD via cumsum(softplus(raw)) + raw[0]
    # To get logD ≈ [4, 5.5, 7], need raw ≈ [2, 0.5, 0.5]
    # Use broad random init centered in the physical range
    params = {
        "logD_raw": jnp.array([2.0, 0.5, 0.3])[:num_domains] + random.normal(k1, (num_domains,)) * 1.5,
        "phi_raw": random.normal(k2, (num_domains - 1,)) * 1.0,
    }
    if args.ea_mode == "shared":
        params["Ea_shared"] = 117.0 + random.normal(k3) * 10.0
    return params

# ── Multi-start Pathfinder ──
print(f"\nRunning {args.num_starts} pathfinder starts...")
t0 = time.time()

modes = []
key = random.PRNGKey(args.seed)

log_prob_jit = jax.jit(log_prob)

for i in range(args.num_starts):
    start_key, sample_key, key = random.split(key, 3)
    init = init_params(start_key)

    try:
        state, info = pathfinder.approximate(
            sample_key, log_prob, init,
            num_samples=50, maxiter=100)

        pos = state.position
        lp = float(log_prob_jit(pos))
        elbo = float(state.elbo)

        flat, unravel = ravel_pytree(pos)
        flat_np = np.array(flat)

        logD = np.cumsum(np.log(1 + np.exp(np.array(pos['logD_raw'])))) + np.array(pos['logD_raw'])[0]
        phi_raw_np = np.array(pos['phi_raw'])
        probs = 1 / (1 + np.exp(-phi_raw_np))
        remaining = np.concatenate([[1.0], np.cumprod(1 - probs)])
        phi = np.concatenate([probs * remaining[:-1], remaining[-1:]])

        # L-BFGS inverse Hessian
        alpha = np.array(state.alpha)
        beta = np.array(state.beta)
        gamma = np.array(state.gamma)
        inv_H = np.diag(alpha) + beta @ gamma @ beta.T

        # Check if positive definite
        eigvals = np.linalg.eigvalsh(inv_H)
        pd = np.all(eigvals > 0)
        if not pd:
            inv_H += np.eye(len(alpha)) * (abs(eigvals.min()) + 1e-8)

        modes.append({
            'flat': flat_np,
            'position': {k: np.array(v) for k, v in pos.items()},
            'logD': logD,
            'phi': phi,
            'log_prob': lp,
            'elbo': elbo,
            'inv_hessian': inv_H,
            'unravel': unravel,
        })
    except Exception as e:
        pass

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        valid = len(modes)
        print(f"  Start {i+1}/{args.num_starts}: {valid} valid modes, {elapsed:.0f}s")

pathfinder_time = time.time() - t0
print(f"Pathfinder done: {len(modes)} valid modes in {pathfinder_time:.0f}s")

# ── Cluster modes ──
print(f"\nClustering modes (threshold={args.cluster_threshold})...")
flats = np.array([m['flat'] for m in modes])
log_probs = np.array([m['log_prob'] for m in modes])

clusters = []
cluster_assignments = np.full(len(modes), -1)

for i in range(len(modes)):
    if cluster_assignments[i] >= 0:
        continue
    cluster_members = [i]
    for j in range(i + 1, len(modes)):
        if cluster_assignments[j] >= 0:
            continue
        dist = np.linalg.norm(flats[i] - flats[j])
        if dist < args.cluster_threshold:
            cluster_members.append(j)
            cluster_assignments[j] = len(clusters)
    cluster_assignments[i] = len(clusters)
    # Pick the member with highest log_prob as representative
    best_idx = max(cluster_members, key=lambda k: log_probs[k])
    clusters.append({
        'representative': modes[best_idx],
        'members': cluster_members,
        'size': len(cluster_members),
    })

print(f"Found {len(clusters)} unique modes")
for ci, c in enumerate(sorted(clusters, key=lambda x: -x['representative']['log_prob'])[:10]):
    m = c['representative']
    ea_str = f", Ea={m['position']['Ea_shared']:.1f}" if 'Ea_shared' in m['position'] else ""
    print(f"  Mode {ci}: logprob={m['log_prob']:.0f}, logD={m['logD'].round(2)}, "
          f"phi={m['phi'].round(3)}{ea_str}, size={c['size']}")

# ── Form Gaussian mixture ──
print(f"\nForming Gaussian mixture...")
# Weight by exp(log_prob) × sqrt(det(covariance)) × cluster_size
# The sqrt(det) accounts for the volume of the Gaussian
mixture_components = []
for c in clusters:
    m = c['representative']
    mean = m['flat']
    cov = m['inv_hessian']
    lp = m['log_prob']

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        continue

    log_weight = lp + 0.5 * logdet + np.log(c['size'])
    mixture_components.append({
        'mean': mean,
        'cov': cov,
        'log_weight': log_weight,
        'mode': m,
        'cluster_size': c['size'],
    })

# Normalize weights
log_weights = np.array([mc['log_weight'] for mc in mixture_components])
log_weights -= np.max(log_weights)  # numerical stability
weights = np.exp(log_weights)
weights /= weights.sum()

print(f"Mixture has {len(mixture_components)} components")
for i, (mc, w) in enumerate(sorted(zip(mixture_components, weights), key=lambda x: -x[1])[:10]):
    print(f"  Component {i}: weight={w:.4f}, logprob={mc['mode']['log_prob']:.0f}, "
          f"logD={mc['mode']['logD'].round(2)}")

# ── Draw samples from mixture ──
print(f"\nDrawing {args.num_samples} samples from mixture...")
rng = np.random.RandomState(args.seed)

n_per_component = rng.multinomial(args.num_samples, weights)
all_flat_samples = []
component_indices = []

for i, (mc, n) in enumerate(zip(mixture_components, n_per_component)):
    if n == 0:
        continue
    try:
        samples = rng.multivariate_normal(mc['mean'], mc['cov'], size=n)
        all_flat_samples.append(samples)
        component_indices.extend([i] * n)
    except np.linalg.LinAlgError:
        cov_reg = mc['cov'] + np.eye(len(mc['mean'])) * 1e-8
        samples = rng.multivariate_normal(mc['mean'], cov_reg, size=n)
        all_flat_samples.append(samples)
        component_indices.extend([i] * n)

flat_samples = np.concatenate(all_flat_samples, axis=0)
print(f"Drew {len(flat_samples)} samples from {np.sum(n_per_component > 0)} components")

# ── Importance resampling ──
print(f"\nComputing importance weights...")
# For each sample: w = exp(log_prob(θ)) / q(θ)
# q(θ) = Σ_k weight_k × N(θ; mean_k, cov_k)

# Unravel one sample to get the unravel function
dummy_unravel = modes[0]['unravel']

log_target = np.zeros(len(flat_samples))
log_proposal = np.zeros(len(flat_samples))

for si in range(len(flat_samples)):
    theta = flat_samples[si]
    params = dummy_unravel(jnp.array(theta))
    log_target[si] = float(log_prob_jit(params))

    # Mixture density
    log_components = []
    for mc, w in zip(mixture_components, weights):
        if w < 1e-30:
            continue
        try:
            log_comp = np.log(w) + multivariate_normal.logpdf(theta, mc['mean'], mc['cov'])
            log_components.append(log_comp)
        except:
            pass
    if log_components:
        max_lc = max(log_components)
        log_proposal[si] = max_lc + np.log(sum(np.exp(lc - max_lc) for lc in log_components))
    else:
        log_proposal[si] = -np.inf

    if (si + 1) % 1000 == 0:
        print(f"  {si+1}/{len(flat_samples)}")

log_iw = log_target - log_proposal
log_iw -= np.max(log_iw)
iw = np.exp(log_iw)
iw /= iw.sum()

ess = 1.0 / np.sum(iw**2)
print(f"Effective Sample Size (ESS): {ess:.0f} / {len(flat_samples)}")
print(f"ESS ratio: {ess/len(flat_samples):.3f}")

# ── Transform samples to physical parameters ──
print(f"\nTransforming samples to physical parameters...")

def np_stick_breaking(pr):
    probs = 1 / (1 + np.exp(-pr))
    remaining = np.concatenate([[1.0], np.cumprod(1 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])

all_logD = []
all_phi = []
all_Ea = []

for si in range(len(flat_samples)):
    params = dummy_unravel(jnp.array(flat_samples[si]))
    logD_raw = np.array(params['logD_raw'])
    logD = np.cumsum(np.log(1 + np.exp(logD_raw))) + logD_raw[0]
    phi = np_stick_breaking(np.array(params['phi_raw']))
    all_logD.append(logD)
    all_phi.append(phi)
    if 'Ea_shared' in params:
        all_Ea.append(float(params['Ea_shared']))
    else:
        all_Ea.append(Ea_fixed_val)

all_logD = np.array(all_logD)
all_phi = np.array(all_phi)
all_Ea = np.array(all_Ea)

# ── Effective Arrhenius ──
print(f"\nComputing effective Arrhenius for {len(flat_samples)} samples...")

def fr_np(y):
    return max(6*np.sqrt(y/np.pi)-3*y, 0) if y < 0.3 else min(1-(6/np.pi**2)*np.exp(-np.pi**2*y), 1)

tK = filt['T_K'].values
dt_arr = filt['seconds_per_extraction_step'].values
all_lnD_eff = []

for si in range(len(flat_samples)):
    logD = all_logD[si]
    phi = all_phi[si]
    Ea = all_Ea[si]
    Dr2 = np.exp(logD)[:, None]
    yi = Dr2 * np.exp(-Ea * 1e3 / (R_gas * tK[None, :])) * dt_arr
    yc = np.cumsum(yi, axis=1)
    Fc = np.array([[fr_np(y) for y in yc[d]] for d in range(num_domains)])
    dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((num_domains, 1)))
    dF_mix = phi @ dF_dom
    Fm = np.cumsum(dF_mix)
    if Fm[-1] < 1e-30:
        continue
    Fs = F_before + Fm / Fm[-1] * frac_fitted
    ye = np.array([y_from_F(f) for f in Fs])
    dye = np.diff(ye, prepend=y_before)
    De = dye / dt_arr
    De[De <= 0] = np.nan
    v = np.log(De)
    if np.all(np.isfinite(v)) and np.all(v > -30):
        all_lnD_eff.append(v)

all_lnD_eff = np.array(all_lnD_eff)
print(f"Valid Arrhenius samples: {len(all_lnD_eff)}")

# Coverage
if len(all_lnD_eff) > 0:
    med = np.median(all_lnD_eff, axis=0)
    lo = np.percentile(all_lnD_eff, 2.5, axis=0)
    hi = np.percentile(all_lnD_eff, 97.5, axis=0)
    covered = np.array([(lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i]) if np.isfinite(lnD_obs[i]) else True
                        for i in range(n_data)])
    n_covered = covered.sum()
    print(f"\nArrhenius coverage: {n_covered}/{n_data}")
    if n_covered < n_data:
        missed = [f"{temps_C[i]}°C" for i in range(n_data) if not covered[i]]
        print(f"  Missed: {', '.join(missed)}")

# ── T-t constraints ──
def max_temp(Ea_kJ, logD0_r2_val, dur_Ma, max_loss=0.01):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD0_r2_val)
    def f(T_C):
        T_K = T_C + 273.15
        y = D0r2 * np.exp(-Ea_kJ*1e3/(R_gas*T_K)) * dur_s
        F = fr_np(y)
        return np.clip(F, 0, 1) - max_loss
    try: return brentq(f, -273, 1000, xtol=0.1)
    except: return np.nan

hrd_logD = all_logD[:, 0]
hrd_Ea = all_Ea

print(f"\n=== T-t Constraints ({len(hrd_logD)} samples) ===")
for dur, label in [(10,"10 My"),(100,"100 My"),(200,"200 My"),(500,"500 My"),(1300,"Isothermal")]:
    temps = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med_t = np.median(valid)
        lo_t, hi_t = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med_t:6.1f}°C  95%CI=[{lo_t:.1f}, {hi_t:.1f}]  crosses 0°C: {hi_t > 0}")

# ── Parameter summaries ──
print(f"\n=== Parameter Summaries ===")
print(f"logD: median={np.median(all_logD, axis=0).round(2)}, std={np.std(all_logD, axis=0).round(3)}")
print(f"phi:  median={np.median(all_phi, axis=0).round(3)}, std={np.std(all_phi, axis=0).round(3)}")
print(f"Ea:   median={np.median(all_Ea):.2f}, std={np.std(all_Ea):.2f}, 95%CI=[{np.percentile(all_Ea, 2.5):.1f}, {np.percentile(all_Ea, 97.5):.1f}]")

# ── Save ──
# Format compatible with evaluate_sweep.py
fs_np = {
    'logD0_r2': all_logD.reshape(1, -1, num_domains),
    'phi': all_phi.reshape(1, -1, num_domains),
}
if args.ea_mode == "shared":
    fs_np['Ea_shared'] = all_Ea.reshape(1, -1)

result = {
    'samples': fs_np,
    'args': vars(args),
    'pathfinder_time': pathfinder_time,
    'n_modes': len(clusters),
    'n_components': len(mixture_components),
    'ess': ess,
    'importance_weights': iw,
    'mixture_weights': weights,
    'modes_summary': [{
        'logD': c['representative']['logD'].tolist(),
        'phi': c['representative']['phi'].tolist(),
        'log_prob': c['representative']['log_prob'],
        'cluster_size': c['size'],
    } for c in sorted(clusters, key=lambda x: -x['representative']['log_prob'])[:20]],
}

out_path = f"results/fit_{args.tag}.pkl"
pickle.dump(result, open(out_path, 'wb'))
print(f"\nSaved {out_path}")
print(f"Total time: {time.time() - t0:.0f}s")
