"""MDD thermochronology fit using emcee affine-invariant ensemble sampler."""
import os, sys
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/jefcheng/.cache/jax'

import subprocess

# Ensure emcee is installed; block h5py if it has numpy incompatibility
try:
    import emcee
except (ImportError, ValueError):
    # ValueError can come from h5py numpy binary incompatibility
    if 'h5py' not in sys.modules:
        sys.modules['h5py'] = None
    try:
        import emcee
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "emcee"])
        import emcee

import time, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
from scipy.optimize import brentq

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="EM4")
parser.add_argument("--num_domains", type=int, default=4)
parser.add_argument("--num_walkers", type=int, default=200)
parser.add_argument("--num_steps", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

num_domains = args.num_domains
n_logD = num_domains          # logD_raw parameters
n_phi  = num_domains - 1      # phi_raw parameters
n_params = n_logD + n_phi     # total flat params

print(f"emcee MDD fit: {num_domains} domains, {args.num_walkers} walkers, "
      f"{args.num_steps} steps, {n_params} params")

# ── Data ────────────────────────────────────────────────────────
df = pd.read_csv('data/nakhla1_parsed_fitted.csv')
total_39Ar = df['39Ar'].sum()
df['dF'] = df['39Ar'] / total_39Ar
df['F'] = df['dF'].cumsum()
df['T_K'] = df['Temp'] + 273.15

def y_from_F(F):
    if F < 0 or F > 1:
        return np.nan
    if F < 0.85:
        a = 6 / np.sqrt(np.pi)
        return ((a - np.sqrt(max(a * a - 12 * F, 0))) / 6) ** 2
    return -(1 / np.pi**2) * np.log(max((1 - F) * np.pi**2 / 6, 1e-30))

df['y'] = df['F'].apply(y_from_F)
filt = df.dropna(subset=['seconds_per_extraction_step'])
filt = filt[filt['Temp'] >= 350]

temps_K = filt['T_K'].values.astype(np.float64)
f_obs   = filt['dF'].values.astype(np.float64)
sigma_obs = (filt['std_39Ar'].values / total_39Ar + 1e-12).astype(np.float64)
dt      = filt['seconds_per_extraction_step'].values.astype(np.float64)
n_data  = len(temps_K)
print(f"Data: {n_data} steps after Temp >= 350 filter")

# ── Constants ───────────────────────────────────────────────────
Ea_fixed = 117.0   # kJ/mol
R_gas = 8.314
logD_prior_mean = 5.7
logD_prior_std  = 2.0

# ── Helper functions (numpy) ───────────────────────────────────
def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def stick_breaking(phi_raw):
    """Convert unconstrained phi_raw to simplex phi."""
    probs = 1.0 / (1.0 + np.exp(-phi_raw))  # sigmoid
    remaining = np.concatenate([[1.0], np.cumprod(1.0 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])

def fractional_release(y):
    """Spherical diffusion fractional release."""
    F_short = 6 * np.sqrt(y / np.pi) - 3 * y
    F_long  = 1 - (6 / np.pi**2) * np.exp(-np.pi**2 * y)
    result = np.where(y < 0.3, F_short, F_long)
    return np.clip(result, 0.0, 1.0)

# ── Log-probability (numpy, flat parameter vector) ─────────────
def log_prob(theta):
    """Log posterior for emcee. theta is flat array of length n_params."""
    logD_raw = theta[:n_logD]
    phi_raw  = theta[n_logD:]

    # Ordering constraint on logD
    logD = np.cumsum(softplus(logD_raw)) + logD_raw[0]

    # Domain fractions via stick-breaking
    phi = stick_breaking(phi_raw)

    # Check for numerical issues
    if np.any(phi <= 0):
        return -np.inf

    # Forward model: D/r^2 values
    D_r2 = np.exp(logD)[:, None]  # (nd, 1)
    T = temps_K[None, :]          # (1, n_data)

    y_inc = D_r2 * np.exp(-Ea_fixed * 1e3 / (R_gas * T)) * dt  # (nd, n_data)
    y_cum = np.cumsum(y_inc, axis=1)                             # (nd, n_data)

    Fcum = fractional_release(y_cum)                             # (nd, n_data)
    dF = np.diff(Fcum, axis=1, prepend=np.zeros((num_domains, 1)))

    dF_mix = phi @ dF                                            # (n_data,)
    total_dF = np.sum(dF_mix)
    f_pred = dF_mix / (total_dF + 1e-30)

    # Log-likelihood
    ll = np.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2
                - np.log(sigma_obs))

    if not np.isfinite(ll):
        return -np.inf

    # Priors
    # logD ~ N(5.7, 2.0)
    prior = np.sum(-0.5 * (logD - logD_prior_mean)**2 / logD_prior_std**2)

    # Dirichlet(1) on phi = uniform on simplex (log-prior = 0 for alpha=1)
    # But we need the Jacobian of stick-breaking? For Dirichlet(1) the
    # (alpha-1)*log(phi) term is zero, so prior contribution is 0.

    return prior + ll

# ── Initialize walkers ─────────────────────────────────────────
rng = np.random.default_rng(args.seed)

init_from_pkl = False
pkl_path = 'results/fit_D3_gpu.pkl'
if os.path.exists(pkl_path):
    print(f"Initializing from {pkl_path}...")
    prev = pickle.load(open(pkl_path, 'rb'))
    prev_samples = prev['samples']
    # Use last loop's samples
    prev_logD_raw = prev_samples['logD_raw'][-1]   # (prev_chains, prev_nd)
    prev_phi_raw  = prev_samples['phi_raw'][-1]     # (prev_chains, prev_nd-1)
    prev_nc = prev_logD_raw.shape[0]
    prev_nd = prev_logD_raw.shape[1]
    print(f"  Previous run: {prev_nc} chains, {prev_nd} domains")

    # Build flat initial positions for each walker
    p0 = np.zeros((args.num_walkers, n_params))
    for w in range(args.num_walkers):
        # Pick a random chain from the previous run
        idx = rng.integers(0, prev_nc)
        # logD_raw: pad or truncate to current num_domains
        lr = np.zeros(n_logD)
        n_copy = min(prev_nd, n_logD)
        lr[:n_copy] = prev_logD_raw[idx, :n_copy]
        # Fill extra domains with small random values
        if n_logD > prev_nd:
            lr[prev_nd:] = rng.normal(0, 0.5, n_logD - prev_nd)

        # phi_raw: pad or truncate
        pr = np.zeros(n_phi)
        n_copy_phi = min(prev_nd - 1, n_phi)
        pr[:n_copy_phi] = prev_phi_raw[idx, :n_copy_phi]
        if n_phi > prev_nd - 1:
            pr[prev_nd - 1:] = rng.normal(0, 0.5, n_phi - (prev_nd - 1))

        p0[w, :n_logD] = lr
        p0[w, n_logD:] = pr

    # Add small perturbation to avoid identical walkers
    p0 += rng.normal(0, 0.02, p0.shape)
    init_from_pkl = True
    print(f"  Initialized {args.num_walkers} walkers from posterior (padded to {num_domains} domains)")
else:
    print("No previous run found; initializing randomly.")
    p0 = np.zeros((args.num_walkers, n_params))
    p0[:, :n_logD] = rng.normal(0, 1.0, (args.num_walkers, n_logD))
    p0[:, n_logD:] = rng.normal(0, 0.5, (args.num_walkers, n_phi))

# Verify all initial positions have finite log_prob
n_bad = 0
for w in range(args.num_walkers):
    lp = log_prob(p0[w])
    if not np.isfinite(lp):
        n_bad += 1
        # Resample until finite
        for _ in range(100):
            p0[w, :n_logD] = rng.normal(0, 1.0, n_logD)
            p0[w, n_logD:] = rng.normal(0, 0.5, n_phi)
            if np.isfinite(log_prob(p0[w])):
                break
if n_bad > 0:
    print(f"  Fixed {n_bad} walkers with non-finite initial log_prob")

# ── Run emcee ──────────────────────────────────────────────────
print(f"\nStarting emcee: {args.num_walkers} walkers x {args.num_steps} steps")
sampler = emcee.EnsembleSampler(args.num_walkers, n_params, log_prob)

t0 = time.time()
step_count = 0
for sample in sampler.sample(p0, iterations=args.num_steps, progress=False):
    step_count += 1
    if step_count % 5000 == 0:
        elapsed = time.time() - t0
        rate = step_count * args.num_walkers / elapsed
        med_lp = np.median(sampler.get_log_prob()[-1])
        print(f"  Step {step_count}/{args.num_steps}: {elapsed:.0f}s, "
              f"{rate:.0f} evals/s, median log_prob={med_lp:.1f}")

sample_time = time.time() - t0
total_evals = args.num_steps * args.num_walkers
print(f"  Done: {sample_time:.1f}s, {total_evals / sample_time:.0f} evals/s")
print(f"  Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f} "
      f"(range [{np.min(sampler.acceptance_fraction):.3f}, "
      f"{np.max(sampler.acceptance_fraction):.3f}])")

# ── Post-process ───────────────────────────────────────────────
# emcee chain shape: (num_steps, num_walkers, n_params)
chain = sampler.get_chain()  # (num_steps, num_walkers, n_params)
print(f"Chain shape: {chain.shape}")

# Reshape to (num_loops, num_walkers, n_params) with loop_size=1000
loop_size = 1000
num_loops = max(1, args.num_steps // loop_size)
# Adjust loop_size if num_steps < 1000
if args.num_steps < loop_size:
    loop_size = args.num_steps
    num_loops = 1
# Take only the last num_loops*loop_size steps (discard remainder at start)
start_step = args.num_steps - num_loops * loop_size
chain_trimmed = chain[start_step:]  # (num_loops*loop_size, num_walkers, n_params)

# Take the LAST position in each loop as the "sample" for that loop
# This matches run_gpu.py where each loop's final state.position is saved
chain_by_loop = chain_trimmed.reshape(num_loops, loop_size, args.num_walkers, n_params)
chain_end_of_loop = chain_by_loop[:, -1, :, :]  # (num_loops, num_walkers, n_params)

# Extract raw parameters
logD_raw_all = chain_end_of_loop[:, :, :n_logD]   # (num_loops, walkers, nd)
phi_raw_all  = chain_end_of_loop[:, :, n_logD:]    # (num_loops, walkers, nd-1)

# Compute logD0_r2 from logD_raw (ordering constraint)
logD_all = np.cumsum(softplus(logD_raw_all), axis=-1) + logD_raw_all[..., :1]

# Compute phi from phi_raw (stick-breaking)
phi_all = np.zeros((num_loops, args.num_walkers, num_domains))
for t in range(num_loops):
    for c in range(args.num_walkers):
        phi_all[t, c] = stick_breaking(phi_raw_all[t, c])

# Build samples dict matching run_gpu.py format
fs_np = {
    'logD_raw': logD_raw_all,    # (num_loops, num_walkers, nd)
    'phi_raw':  phi_raw_all,      # (num_loops, num_walkers, nd-1)
    'logD0_r2': logD_all,         # (num_loops, num_walkers, nd)
    'phi':      phi_all,           # (num_loops, num_walkers, nd)
}

print(f"\nOutput shapes:")
for k, v in fs_np.items():
    print(f"  {k}: {v.shape}")

# ── Convergence (R-hat) ────────────────────────────────────────
def split_rhat(samples, discard_frac=0.5):
    """Compute split R-hat, discarding first discard_frac of samples as burn-in."""
    T, C = samples.shape
    start = int(T * discard_frac)
    post_burnin = samples[start:]
    T2 = post_burnin.shape[0]
    half = T2 // 2
    if half < 2:
        return np.nan
    chains = np.concatenate([post_burnin[:half].T, post_burnin[half:half*2].T], axis=0)
    n, m = half, chains.shape[0]
    chain_means = chains.mean(axis=1)
    B = n / (m - 1) * np.sum((chain_means - chain_means.mean())**2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    return np.sqrt(((n - 1) / n * W + B / n) / W) if W > 0 else np.nan

print("\n=== Convergence (R-hat on second half of samples) ===")
for key in fs_np:
    arr = fs_np[key]
    if arr.ndim == 3:
        for d in range(arr.shape[2]):
            rhat = split_rhat(arr[:, :, d])
            print(f"  {key}[{d}]: R-hat = {rhat:.3f} "
                  f"{'OK' if rhat < 1.1 else 'WARNING'}")

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

# Use HRD (most retentive domain = smallest logD = index 0 after ordering)
final_logD = fs_np['logD0_r2'][-1]   # (walkers, nd)
hrd_logD = final_logD[:, 0]           # smallest logD = most retentive
hrd_Ea = np.full(len(hrd_logD), Ea_fixed)

nc = len(hrd_logD)
print(f"\n=== T-t Constraints ({nc} walkers) ===")
for dur, label in [(10, "10 My"), (100, "100 My"), (200, "200 My"),
                    (500, "500 My"), (1300, "Isothermal")]:
    temps = np.array([max_temp(ea, ld, dur)
                      for ea, ld in zip(hrd_Ea[:nc], hrd_logD[:nc])])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med = np.median(valid)
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med:6.1f} C  95%CI=[{lo:.1f}, {hi:.1f}]  "
              f"crosses 0 C: {hi > 0}")

# ── Arrhenius coverage ──────────────────────────────────────────
# Compute effective Arrhenius for coverage check
filt2 = df.dropna(subset=['seconds_per_extraction_step'])
filt2 = filt2[filt2['Temp'] >= 350]
F_before = df[df['Temp'] < 350]['dF'].sum()
fitted_F_cum = F_before + np.cumsum(filt2['dF'].values)
y_before = y_from_F(F_before)
y_obs = np.array([y_from_F(f) for f in fitted_F_cum])
dy_obs = np.diff(y_obs, prepend=y_before)
D_obs = dy_obs / dt
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)
frac_fitted = fitted_F_cum[-1] - F_before

def fr_np(y):
    return max(6 * np.sqrt(y / np.pi) - 3 * y, 0) if y < 0.3 else \
           min(1 - (6 / np.pi**2) * np.exp(-np.pi**2 * y), 1)

def compute_effective_arrhenius(logD_vals, phi_vals, nd):
    """Compute effective ln(D/rho2) for one posterior sample."""
    Dr2 = np.exp(logD_vals)[:, None]
    tK = temps_K
    yi = Dr2 * np.exp(-Ea_fixed * 1e3 / (R_gas * tK[None, :])) * dt
    yc = np.cumsum(yi, axis=1)
    Fc = np.array([[fr_np(y) for y in yc[d]] for d in range(nd)])
    dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((nd, 1)))
    dF_mix = phi_vals @ dF_dom
    Fm = np.cumsum(dF_mix)
    if Fm[-1] < 1e-30:
        return None
    Fs = F_before + Fm / Fm[-1] * frac_fitted
    ye = np.array([y_from_F(f) for f in Fs])
    dye = np.diff(ye, prepend=y_before)
    De = dye / dt
    De[De <= 0] = np.nan
    v = np.log(De)
    if np.all(np.isfinite(v)) and np.all(v > -30):
        return v
    return None

all_lnD = []
n_check = min(nc, 500)
for c in range(n_check):
    logD = fs_np['logD0_r2'][-1, c]
    phi  = fs_np['phi'][-1, c]
    v = compute_effective_arrhenius(logD, phi, num_domains)
    if v is not None:
        all_lnD.append(v)

if len(all_lnD) >= 10:
    all_lnD_arr = np.array(all_lnD)
    lo = np.percentile(all_lnD_arr, 2.5, axis=0)
    hi = np.percentile(all_lnD_arr, 97.5, axis=0)
    covered = np.array([(lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i])
                         if np.isfinite(lnD_obs[i]) else True
                         for i in range(n_data)])
    n_covered = covered.sum()
    missed = [f"{filt2['Temp'].values[i]} C" for i in range(n_data) if not covered[i]]
    print(f"\n=== Arrhenius Coverage ===")
    print(f"  Covered: {n_covered}/{n_data}")
    if missed:
        print(f"  Missed: {', '.join(missed)}")
    else:
        print(f"  All points covered!")
else:
    print(f"\n=== Arrhenius Coverage ===")
    print(f"  Too few valid samples ({len(all_lnD)}) for coverage check")

# ── Save ────────────────────────────────────────────────────────
out_path = f"results/fit_{args.tag}.pkl"
pickle.dump({
    "samples": fs_np,
    "args": vars(args),
    "warmup_time": 0.0,        # emcee has no separate warmup
    "sample_time": sample_time,
    "compile_time": 0.0,       # no compilation
    "acceptance_fraction": sampler.acceptance_fraction.tolist(),
}, open(out_path, "wb"))

print(f"\nSaved {out_path}")
print(f"  samples shape: (num_loops={num_loops}, num_chains={args.num_walkers}, ...)")
print(f"Total time: {sample_time:.0f}s")
