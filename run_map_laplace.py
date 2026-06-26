"""MAP + Laplace approximation for MDD thermochronology model.

Finds MAP via multi-restart Adam, computes Hessian for Gaussian posterior,
draws samples for uncertainty quantification.
"""
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/jefcheng/.cache/jax'
import time, pickle, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
import optax

print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="MAP_L4")
parser.add_argument("--num_domains", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_restarts", type=int, default=500)
parser.add_argument("--n_opt_steps", type=int, default=5000)
parser.add_argument("--n_samples", type=int, default=10000)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

num_domains = args.num_domains

# ── Data ────────────────────────────────────────────────────────
df = pd.read_csv('data/nakhla1_parsed_fitted.csv')
total_39Ar = df['39Ar'].sum()
df['dF'] = df['39Ar'] / total_39Ar
df['F'] = df['dF'].cumsum()
df['T_K'] = df['Temp'] + 273.15

def y_from_F(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6 / np.sqrt(np.pi)
        return ((a - np.sqrt(max(a * a - 12 * F, 0))) / 6) ** 2
    return -(1 / np.pi**2) * np.log(max((1 - F) * np.pi**2 / 6, 1e-30))

df['y'] = df['F'].apply(y_from_F)
filt = df.dropna(subset=['seconds_per_extraction_step'])
filt = filt[filt['Temp'] >= 350]

temps_K = jnp.array(filt['T_K'].values)
f_obs = jnp.array(filt['dF'].values)
sigma_obs = jnp.array(filt['std_39Ar'].values / total_39Ar + 1e-12)
dt = jnp.array(filt['seconds_per_extraction_step'].values)
n_data = len(temps_K)
temps_C = filt['Temp'].values

# For effective Arrhenius computation
F_before = df[df['Temp'] < 350].dropna(subset=['seconds_per_extraction_step'])['dF'].sum()
# Actually need all rows below 350 regardless of dt for cumulative F
F_before_all = df[df['Temp'] < 350]['dF'].sum()
fitted_dF = filt['dF'].values
fitted_F_cum = F_before_all + np.cumsum(fitted_dF)
frac_fitted = fitted_F_cum[-1] - F_before_all
tK_np = filt['T_K'].values
dt_np = filt['seconds_per_extraction_step'].values
y_before = y_from_F(F_before_all)
y_obs_arr = np.array([y_from_F(f) for f in fitted_F_cum])
dy_obs_arr = np.diff(y_obs_arr, prepend=y_before)
D_obs = dy_obs_arr / dt_np
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)

print(f"Data: {n_data} steps, {num_domains} domains")
print(f"F_before (T<350): {F_before_all:.6f}")

# ── Model ───────────────────────────────────────────────────────
Ea_fixed_val = 117.0
R_gas = 8.314
logD_prior_mean = 5.7
logD_prior_std = 2.0

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
    phi_raw = params["phi_raw"]

    # Order constraint on logD
    logD = jnp.cumsum(jax.nn.softplus(logD_raw)) + logD_raw[0]
    phi = stick_breaking(phi_raw)

    D_r2 = jnp.exp(logD)[:, None]  # (nd, 1)
    T = temps_K[None, :]            # (1, n_data)

    Ea_broadcast = jnp.full((num_domains, 1), Ea_fixed_val)

    y_inc = D_r2 * jnp.exp(-Ea_broadcast * 1e3 / (R_gas * T)) * dt
    y_cum = jnp.cumsum(y_inc, axis=1)
    F_cum = fractional_release(y_cum)
    dF = jnp.diff(F_cum, axis=1, prepend=jnp.zeros((num_domains, 1)))

    dF_mix = phi @ dF  # (n_data,)
    f_pred = dF_mix / (jnp.sum(dF_mix) + 1e-30)

    ll = jnp.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2
                 - jnp.log(sigma_obs))

    # Priors
    prior = jnp.sum(-0.5 * (logD - logD_prior_mean)**2 / logD_prior_std**2)
    # Dirichlet(1) prior on phi: (alpha-1)*log(phi) = 0 for alpha=1
    # So Dirichlet(1) adds no contribution, but we keep the term for clarity
    prior += jnp.sum((1.0 - 1.0) * jnp.log(phi))

    return prior + ll

# ── MAP via multi-restart Adam ──────────────────────────────────
def init_params(key):
    k1, k2 = random.split(key, 2)
    return {
        "logD_raw": random.normal(k1, (num_domains,)) * 1.0,
        "phi_raw": jnp.zeros((num_domains - 1,)),
    }

neg_log_prob = jax.jit(lambda p: -log_prob(p))
grad_fn = jax.jit(jax.grad(neg_log_prob))

print(f"\nFinding MAP estimate via Adam ({args.n_restarts} restarts x {args.n_opt_steps} steps)...")
t0 = time.time()

best_lp = -jnp.inf
best_params = None

for trial in range(args.n_restarts):
    trial_key = random.PRNGKey(args.seed + trial)
    params = init_params(trial_key)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    for step in range(args.n_opt_steps):
        g = grad_fn(params)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

    lp = float(log_prob(params))
    if lp > best_lp:
        best_lp = lp
        best_params = params

    if (trial + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Trial {trial+1}/{args.n_restarts}: best logprob={best_lp:.2f} ({elapsed:.0f}s)")

map_time = time.time() - t0
print(f"  MAP optimization done: {map_time:.1f}s")

# Extract MAP physical parameters
logD_map = jnp.cumsum(jax.nn.softplus(best_params['logD_raw'])) + best_params['logD_raw'][0]
phi_map = stick_breaking(best_params['phi_raw'])
print(f"\n=== MAP Parameters ===")
print(f"  logD (ln D0/rho2): {np.array(logD_map)}")
print(f"  phi (fractions):   {np.array(phi_map)}")
print(f"  Ea (fixed):        {Ea_fixed_val} kJ/mol")
print(f"  log_prob:          {best_lp:.2f}")

# ── Laplace approximation: Hessian at MAP ───────────────────────
print(f"\nComputing Hessian at MAP...")
t0 = time.time()

# Flatten parameters to a vector
flat_map, unflatten_fn = ravel_pytree(best_params)
n_params = len(flat_map)
print(f"  Parameter vector dimension: {n_params}")

def log_prob_flat(flat_params):
    return log_prob(unflatten_fn(flat_params))

# Compute the Hessian of the negative log-probability
hessian_fn = jax.jit(jax.hessian(lambda x: -log_prob_flat(x)))
H = np.array(hessian_fn(flat_map))

hessian_time = time.time() - t0
print(f"  Hessian computed in {hessian_time:.1f}s")
print(f"  Hessian shape: {H.shape}")

# Check positive definiteness and compute covariance
eigvals = np.linalg.eigvalsh(H)
print(f"  Hessian eigenvalues: min={eigvals.min():.4e}, max={eigvals.max():.4e}")

ridge_added = False
if eigvals.min() <= 0:
    ridge = 1e-6
    H += ridge * np.eye(n_params)
    eigvals_new = np.linalg.eigvalsh(H)
    print(f"  WARNING: Hessian not positive definite. Added ridge {ridge}.")
    print(f"  New eigenvalues: min={eigvals_new.min():.4e}, max={eigvals_new.max():.4e}")
    ridge_added = True

try:
    cov = np.linalg.inv(H)
    print(f"  Covariance matrix computed successfully.")
    print(f"  Diagonal (marginal variances): {np.diag(cov)}")
except np.linalg.LinAlgError:
    print(f"  ERROR: Could not invert Hessian. Using pseudo-inverse.")
    cov = np.linalg.pinv(H)

# ── Draw samples from Laplace approximation ────────────────────
print(f"\nDrawing {args.n_samples} samples from N(MAP, H^{{-1}})...")
rng = np.random.default_rng(args.seed)
flat_samples = rng.multivariate_normal(np.array(flat_map), cov, size=args.n_samples)

# Unflatten each sample and compute physical parameters
all_logD = []
all_phi = []
for i in range(args.n_samples):
    p = unflatten_fn(jnp.array(flat_samples[i]))
    logD_i = np.array(jnp.cumsum(jax.nn.softplus(p['logD_raw'])) + p['logD_raw'][0])
    phi_i = np.array(stick_breaking(p['phi_raw']))
    all_logD.append(logD_i)
    all_phi.append(phi_i)

all_logD = np.array(all_logD)   # (n_samples, nd)
all_phi = np.array(all_phi)     # (n_samples, nd)

print(f"  logD median: {np.median(all_logD, axis=0)}")
print(f"  logD std:    {np.std(all_logD, axis=0)}")
print(f"  phi median:  {np.median(all_phi, axis=0)}")
print(f"  phi std:     {np.std(all_phi, axis=0)}")

# ── Effective Arrhenius for each sample ─────────────────────────
print(f"\nComputing effective Arrhenius curves for {args.n_samples} samples...")
t0 = time.time()

def fr_np(y):
    return max(6 * np.sqrt(y / np.pi) - 3 * y, 0) if y < 0.3 else min(1 - (6 / np.pi**2) * np.exp(-np.pi**2 * y), 1)

def compute_effective_arrhenius(logD_vals, phi_vals, nd):
    """Compute effective ln(D/rho2) for one posterior sample."""
    Dr2 = np.exp(logD_vals)[:, None]
    yi = Dr2 * np.exp(-Ea_fixed_val * 1e3 / (R_gas * tK_np[None, :])) * dt_np
    yc = np.cumsum(yi, axis=1)
    Fc = np.array([[fr_np(y) for y in yc[d]] for d in range(nd)])
    dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((nd, 1)))
    dF_mix = phi_vals @ dF_dom
    Fm = np.cumsum(dF_mix)
    if Fm[-1] < 1e-30:
        return None
    Fs = F_before_all + Fm / Fm[-1] * frac_fitted
    ye = np.array([y_from_F(f) for f in Fs])
    dye = np.diff(ye, prepend=y_before)
    De = dye / dt_np
    De[De <= 0] = np.nan
    v = np.log(De)
    if np.all(np.isfinite(v)) and np.all(v > -30):
        return v
    return None

all_lnD_eff = []
n_valid = 0
for i in range(args.n_samples):
    v = compute_effective_arrhenius(all_logD[i], all_phi[i], num_domains)
    if v is not None:
        all_lnD_eff.append(v)
        n_valid += 1

arrhenius_time = time.time() - t0
print(f"  Valid Arrhenius curves: {n_valid}/{args.n_samples} ({arrhenius_time:.1f}s)")

# ── Arrhenius coverage check ────────────────────────────────────
if len(all_lnD_eff) >= 10:
    all_lnD_eff = np.array(all_lnD_eff)
    lo = np.percentile(all_lnD_eff, 2.5, axis=0)
    hi = np.percentile(all_lnD_eff, 97.5, axis=0)
    med = np.median(all_lnD_eff, axis=0)

    covered = np.array([
        (lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i]) if np.isfinite(lnD_obs[i]) else True
        for i in range(n_data)
    ])
    n_covered = covered.sum()

    print(f"\n=== Arrhenius Coverage (95% CI) ===")
    print(f"  Covered: {n_covered}/{n_data}")
    if n_covered < n_data:
        missed = [f"{temps_C[i]}C" for i in range(n_data) if not covered[i]]
        print(f"  Missed: {', '.join(missed)}")
    else:
        print(f"  All {n_data} observed ln(D/rho2) points covered!")
else:
    print(f"\n  WARNING: Too few valid Arrhenius samples ({len(all_lnD_eff)}) for coverage check.")
    n_covered = 0

# ── T-t constraints ─────────────────────────────────────────────
from scipy.optimize import brentq

def max_temp(Ea_kJ, logD0_r2_val, dur_Ma, max_loss=0.01):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD0_r2_val)
    def f(T_C):
        T_K = T_C + 273.15
        y = D0r2 * np.exp(-Ea_kJ * 1e3 / (R_gas * T_K)) * dur_s
        F = 6 * np.sqrt(y / np.pi) - 3 * y if y < 0.3 else 1 - (6 / np.pi**2) * np.exp(-np.pi**2 * y)
        return np.clip(F, 0, 1) - max_loss
    try:
        return brentq(f, -273, 1000, xtol=0.1)
    except:
        return np.nan

# HRD = smallest logD = logD[:, 0] (ordered by construction)
hrd_logD = all_logD[:, 0]  # smallest domain

print(f"\n=== T-t Constraints (HRD, {len(hrd_logD)} samples) ===")
for dur, label in [(10, "10 My"), (100, "100 My"), (200, "200 My"), (500, "500 My"), (1300, "Isothermal")]:
    temps_tt = np.array([max_temp(Ea_fixed_val, ld, dur) for ld in hrd_logD])
    valid = temps_tt[~np.isnan(temps_tt)]
    if len(valid) > 0:
        med_tt = np.median(valid)
        lo_tt, hi_tt = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med_tt:6.1f}C  95%CI=[{lo_tt:.1f}, {hi_tt:.1f}]  crosses 0C: {hi_tt > 0}")
    else:
        print(f"  {label:12s}: no valid solutions")

# ── Save results in run_gpu.py format ───────────────────────────
# Format: samples dict with shape (1, n_samples, nd) to match (num_loops, num_chains, nd)
fs_np = {}
fs_np['logD0_r2'] = all_logD[None, :, :]       # (1, n_samples, nd)
fs_np['phi'] = all_phi[None, :, :]              # (1, n_samples, nd)

# Also save raw parameters
logD_raw_all = flat_samples[:, :num_domains]
phi_raw_all = flat_samples[:, num_domains:]
fs_np['logD_raw'] = logD_raw_all[None, :, :]    # (1, n_samples, nd)
fs_np['phi_raw'] = phi_raw_all[None, :, :]      # (1, n_samples, nd-1)

os.makedirs('results', exist_ok=True)
out_path = f"results/fit_{args.tag}.pkl"
pickle.dump({
    "samples": fs_np,
    "args": vars(args),
    "map_params": {k: np.array(v) for k, v in best_params.items()},
    "map_logD": np.array(logD_map),
    "map_phi": np.array(phi_map),
    "map_logprob": float(best_lp),
    "hessian": H,
    "covariance": cov,
    "ridge_added": ridge_added,
    "map_time": map_time,
    "hessian_time": hessian_time,
    "arrhenius_coverage": int(n_covered),
    "arrhenius_total": n_data,
}, open(out_path, "wb"))

print(f"\nSaved {out_path}")
print(f"Total time: MAP={map_time:.0f}s, Hessian={hessian_time:.0f}s, Arrhenius={arrhenius_time:.0f}s")
