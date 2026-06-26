"""Nested sampling of MDD thermochronology model using ultranest."""
import subprocess, sys

# Ensure ultranest is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultranest", "-q"])

import os, time, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import ultranest

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Nested sampling MDD fit with ultranest")
parser.add_argument("--tag", type=str, default="NS4")
parser.add_argument("--num_domains", type=int, default=4)
parser.add_argument("--min_num_live_points", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

num_domains = args.num_domains
num_logD = num_domains
num_phi = num_domains - 1
num_params = num_logD + num_phi

param_names = [f"logD_raw_{i}" for i in range(num_logD)] + \
              [f"phi_raw_{i}" for i in range(num_phi)]
assert len(param_names) == num_params

print(f"Nested sampling: {num_domains} domains, {num_params} params")
print(f"Parameters: {param_names}")

# ── Data ────────────────────────────────────────────────────────
df = pd.read_csv('data/nakhla1_parsed_fitted.csv')
total_39Ar = df['39Ar'].sum()
df['dF'] = df['39Ar'] / total_39Ar
df['F'] = df['dF'].cumsum()
df['T_K'] = df['Temp'] + 273.15

filt = df.dropna(subset=['seconds_per_extraction_step'])
filt = filt[filt['Temp'] >= 350].copy()

temps_K = filt['T_K'].values.astype(np.float64)
f_obs = filt['dF'].values.astype(np.float64)
sigma_obs = (filt['std_39Ar'].values / total_39Ar + 1e-12).astype(np.float64)
dt = filt['seconds_per_extraction_step'].values.astype(np.float64)
n_data = len(temps_K)
print(f"Data: {n_data} steps after filtering (Temp >= 350)")

# ── Constants ───────────────────────────────────────────────────
Ea_fixed = 117.0  # kJ/mol
R_gas = 8.314

# ── Model functions (numpy) ────────────────────────────────────
def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def stick_breaking(phi_raw):
    """Convert unconstrained phi_raw to domain fractions summing to 1."""
    probs = 1.0 / (1.0 + np.exp(-phi_raw))  # sigmoid
    remaining = np.concatenate([[1.0], np.cumprod(1.0 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])


def fractional_release(y):
    """Spherical diffusion fractional release."""
    F_short = 6.0 * np.sqrt(y / np.pi) - 3.0 * y
    F_long = 1.0 - (6.0 / np.pi**2) * np.exp(-np.pi**2 * y)
    result = np.where(y < 0.3, F_short, F_long)
    return np.clip(result, 0.0, 1.0)


def forward_model(logD_raw, phi_raw):
    """Run the MDD forward model, return predicted fractional releases."""
    # Ordered logD via cumulative softplus
    logD = np.cumsum(softplus(logD_raw)) + logD_raw[0]

    # Domain fractions via stick-breaking
    phi = stick_breaking(phi_raw)

    # Diffusivity for each domain at each temperature step
    # D_r2[d] = exp(logD[d])
    D_r2 = np.exp(logD)  # (num_domains,)

    # y_inc[d, i] = D_r2[d] * exp(-Ea / (R * T_K[i])) * dt[i]
    exp_term = np.exp(-Ea_fixed * 1e3 / (R_gas * temps_K))  # (n_data,)
    y_inc = D_r2[:, None] * exp_term[None, :] * dt[None, :]  # (nd, n_data)
    y_cum = np.cumsum(y_inc, axis=1)  # (nd, n_data)

    # Fractional release for each domain
    Fcum = fractional_release(y_cum)  # (nd, n_data)
    dF = np.diff(Fcum, axis=1, prepend=np.zeros((num_domains, 1)))  # (nd, n_data)

    # Mix domains
    dF_mix = phi @ dF  # (n_data,)
    f_pred = dF_mix / (np.sum(dF_mix) + 1e-30)

    return f_pred


# ── Prior transform ────────────────────────────────────────────
def prior_transform(cube):
    """Map unit hypercube [0,1]^7 to parameter space.

    logD_raw[i]: uniform [-5, 10]
    phi_raw[i]: uniform [-5, 5]
    """
    params = np.empty(num_params)
    # logD_raw: uniform [-5, 10]
    for i in range(num_logD):
        params[i] = cube[i] * 15.0 - 5.0  # [-5, 10]
    # phi_raw: uniform [-5, 5]
    for i in range(num_phi):
        params[num_logD + i] = cube[num_logD + i] * 10.0 - 5.0  # [-5, 5]
    return params


# ── Log-likelihood ─────────────────────────────────────────────
def log_likelihood(params):
    """Compute log-likelihood for a single parameter vector (shape (7,))."""
    logD_raw = params[:num_logD]
    phi_raw = params[num_logD:]

    f_pred = forward_model(logD_raw, phi_raw)

    # Gaussian log-likelihood (no constant -0.5*log(2*pi) needed for model comparison,
    # but we include the normalisation for completeness)
    ll = np.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2 - np.log(sigma_obs))
    return ll


# ── Run nested sampling ───────────────────────────────────────
print("\nStarting ultranest ReactiveNestedSampler...")
t0 = time.time()

np.random.seed(args.seed)

sampler = ultranest.ReactiveNestedSampler(
    param_names,
    log_likelihood,
    prior_transform,
    log_dir=None,
    resume="overwrite",
    storage_backend="memory",
)

result = sampler.run(
    min_num_live_points=args.min_num_live_points,
    min_ess=1000,
)

run_time = time.time() - t0
print(f"\nNested sampling completed in {run_time:.1f}s")

# ── Extract results ────────────────────────────────────────────
logZ = result['logz']
logZ_err = result['logzerr']
print(f"\nLog evidence: logZ = {logZ:.2f} +/- {logZ_err:.2f}")

# Weighted posterior samples
weighted_samples = result['weighted_samples']['points']  # (N, num_params)
weights = result['weighted_samples']['weights']           # (N,)
N = weighted_samples.shape[0]
print(f"Number of weighted posterior samples: {N}")

# Equal-weight posterior samples (resampled)
equal_samples = result['posterior']['points']  # (M, num_params)
M = equal_samples.shape[0]
print(f"Number of equal-weight posterior samples: {M}")

# ── Post-process: compute physical parameters ─────────────────
def compute_physical_params(raw_params):
    """Convert raw parameters to physical logD0_r2 and phi."""
    logD_raw = raw_params[:num_logD]
    phi_raw = raw_params[num_logD:]
    logD = np.cumsum(softplus(logD_raw)) + logD_raw[0]
    phi = stick_breaking(phi_raw)
    return logD, phi

# Compute for all equal-weight samples
all_logD = np.zeros((M, num_domains))
all_phi = np.zeros((M, num_domains))
for i in range(M):
    logD, phi = compute_physical_params(equal_samples[i])
    all_logD[i] = logD
    all_phi[i] = phi

print("\n=== Parameter summaries (equal-weight posterior) ===")
for i in range(num_logD):
    vals = equal_samples[:, i]
    print(f"  logD_raw_{i}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
          f"95%CI=[{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]")
for i in range(num_phi):
    vals = equal_samples[:, num_logD + i]
    print(f"  phi_raw_{i}:  mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
          f"95%CI=[{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]")

print("\n=== Physical parameters ===")
for i in range(num_domains):
    print(f"  logD0_r2[{i}]: median={np.median(all_logD[:, i]):.3f}, "
          f"95%CI=[{np.percentile(all_logD[:, i], 2.5):.3f}, {np.percentile(all_logD[:, i], 97.5):.3f}]")
for i in range(num_domains):
    print(f"  phi[{i}]:      median={np.median(all_phi[:, i]):.3f}, "
          f"95%CI=[{np.percentile(all_phi[:, i], 2.5):.3f}, {np.percentile(all_phi[:, i], 97.5):.3f}]")

# ── Arrhenius coverage ─────────────────────────────────────────
def yF(F):
    if F < 0 or F > 1:
        return np.nan
    if F < 0.85:
        a = 6 / np.sqrt(np.pi)
        return ((a - np.sqrt(max(a * a - 12 * F, 0))) / 6) ** 2
    return -(1 / np.pi**2) * np.log(max((1 - F) * np.pi**2 / 6, 1e-30))

# Observed Arrhenius data
F_before = df[df['Temp'] < 350]['dF'].sum() if len(df[df['Temp'] < 350]) > 0 else 0.0
fitted_F_cum = F_before + np.cumsum(filt['dF'].values)
y_before = yF(F_before)
y_obs_arr = np.array([yF(f) for f in fitted_F_cum])
dy_obs = np.diff(y_obs_arr, prepend=y_before)
D_obs = dy_obs / dt
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)

def compute_effective_arrhenius(logD_vals, phi_vals):
    """Compute effective ln(D/rho2) for one posterior sample."""
    nd = len(logD_vals)
    Dr2 = np.exp(logD_vals)[:, None]
    yi = Dr2 * np.exp(-Ea_fixed * 1e3 / (R_gas * temps_K[None, :])) * dt
    yc = np.cumsum(yi, axis=1)
    Fc = np.array([[yF(yc[d, j]) if np.isfinite(yc[d, j]) else 0.0
                    for j in range(n_data)] for d in range(nd)])
    dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((nd, 1)))
    dF_mix = phi_vals @ dF_dom
    Fm = np.cumsum(dF_mix)
    if Fm[-1] < 1e-30:
        return None
    frac_fitted = fitted_F_cum[-1] - F_before
    Fs = F_before + Fm / Fm[-1] * frac_fitted
    ye = np.array([yF(f) for f in Fs])
    dye = np.diff(ye, prepend=y_before)
    De = dye / dt
    De[De <= 0] = np.nan
    v = np.log(De)
    if np.all(np.isfinite(v)) and np.all(v > -30):
        return v
    return None

print("\n=== Arrhenius coverage ===")
all_lnD_eff = []
n_check = min(M, 500)
for i in range(n_check):
    v = compute_effective_arrhenius(all_logD[i], all_phi[i])
    if v is not None:
        all_lnD_eff.append(v)

if len(all_lnD_eff) >= 10:
    all_lnD_eff = np.array(all_lnD_eff)
    lo = np.percentile(all_lnD_eff, 2.5, axis=0)
    hi = np.percentile(all_lnD_eff, 97.5, axis=0)
    temps_C = filt['Temp'].values
    covered = np.array([(lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i])
                        if np.isfinite(lnD_obs[i]) else True
                        for i in range(n_data)])
    n_covered = covered.sum()
    print(f"  Covered: {n_covered}/{n_data}")
    if n_covered < n_data:
        missed = [f"{temps_C[i]}C" for i in range(n_data) if not covered[i]]
        print(f"  Missed: {', '.join(missed)}")
    else:
        print("  All data points covered by 95% CI")
else:
    print(f"  Only {len(all_lnD_eff)} valid Arrhenius samples (need >= 10)")

# ── T-t constraints ────────────────────────────────────────────
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
    except Exception:
        return np.nan

# Use HRD (smallest logD = first domain, which is the most retentive)
hrd_logD = all_logD[:, 0]  # ordered, [0] is smallest = HRD
hrd_Ea = np.full(M, Ea_fixed)

print(f"\n=== T-t Constraints ({M} samples) ===")
for dur, label in [(10, "10 My"), (100, "100 My"), (200, "200 My"),
                    (500, "500 My"), (1300, "Isothermal")]:
    temps = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med = np.median(valid)
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med:6.1f} C  95%CI=[{lo:.1f}, {hi:.1f}]  crosses 0C: {hi > 0}")

# ── Save results ───────────────────────────────────────────────
# Format compatible with evaluate_sweep.py:
# samples dict with arrays of shape (num_loops, num_chains, num_dims)
# For nested sampling: treat as (N, 1, dim) — one "chain" with N "loops"
fs_np = {}

# Raw parameters
logD_raw_all = equal_samples[:, :num_logD]  # (M, num_domains)
phi_raw_all = equal_samples[:, num_logD:]   # (M, num_phi)

fs_np["logD_raw"] = logD_raw_all[:, None, :]  # (M, 1, num_domains)
fs_np["phi_raw"] = phi_raw_all[:, None, :]    # (M, 1, num_phi)

# Physical parameters
fs_np["logD0_r2"] = all_logD[:, None, :]      # (M, 1, num_domains)
fs_np["phi"] = all_phi[:, None, :]            # (M, 1, num_domains)

os.makedirs("results", exist_ok=True)
output_path = f"results/fit_{args.tag}.pkl"
output = {
    "samples": fs_np,
    "args": vars(args),
    "run_time": run_time,
    "logZ": logZ,
    "logZ_err": logZ_err,
    "ultranest_result": {
        "logz": result['logz'],
        "logzerr": result['logzerr'],
        "ncall": result['ncall'],
        "niter": result['niter'],
        "paramnames": param_names,
    },
    "weighted_samples": weighted_samples,
    "weights": weights,
}
pickle.dump(output, open(output_path, "wb"))
print(f"\nSaved {output_path}")
print(f"  samples shapes: logD_raw={fs_np['logD_raw'].shape}, phi_raw={fs_np['phi_raw'].shape}, "
      f"logD0_r2={fs_np['logD0_r2'].shape}, phi={fs_np['phi'].shape}")
print(f"  Format: (num_loops={M}, num_chains=1, num_dims=...)")
print(f"\nTotal time: {run_time:.1f}s")
