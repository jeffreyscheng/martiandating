"""Reparameterized MDD fit: whiten parameter space using D3_gpu posterior covariance."""
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/jefcheng/.cache/jax'
import time, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
import blackjax

print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="RP4")
parser.add_argument("--num_chains", type=int, default=100)
parser.add_argument("--num_loops", type=int, default=100)
parser.add_argument("--steps_per_loop", type=int, default=1000)
parser.add_argument("--num_warmup", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

num_domains = 4
n_params = 2 * num_domains - 1  # logD_raw(4) + phi_raw(3) = 7

# ── Data ────────────────────────────────────────────────────────
df = pd.read_csv('data/nakhla1_parsed_fitted.csv')
total_39Ar = df['39Ar'].sum()
df['dF'] = df['39Ar'] / total_39Ar
df['F'] = df['dF'].cumsum()
df['T_K'] = df['Temp'] + 273.15

filt = df.dropna(subset=['seconds_per_extraction_step'])
filt = filt[filt['Temp'] >= 350]

temps_K = jnp.array(filt['T_K'].values)
f_obs = jnp.array(filt['dF'].values)
sigma_obs = jnp.array(filt['std_39Ar'].values / total_39Ar + 1e-12)
dt = jnp.array(filt['seconds_per_extraction_step'].values)
n_data = len(temps_K)
print(f"Data: {n_data} steps, {num_domains} domains, {args.num_chains} chains")

# ── Load D3_gpu posterior and build whitening transform ─────────
print("Loading D3_gpu posterior for whitening transform...")
d3 = pickle.load(open('results/fit_D3_gpu.pkl', 'rb'))
d3_logD_raw = d3['samples']['logD_raw'][-1]  # (100, 3)
d3_phi_raw = d3['samples']['phi_raw'][-1]     # (100, 2)
d3_theta = np.concatenate([d3_logD_raw, d3_phi_raw], axis=1)  # (100, 5)

# Compute mean and covariance of the 3-domain posterior
mu_3 = d3_theta.mean(axis=0)  # (5,)
cov_3 = np.cov(d3_theta, rowvar=False)  # (5, 5)

# Pad to 7 dimensions (4 domains): append prior mean=0, prior var=1
mu_7 = np.zeros(n_params)
mu_7[:5] = mu_3  # first 5 from D3_gpu
mu_7[5] = 0.0    # logD_raw[3] prior mean
mu_7[6] = 0.0    # phi_raw[2] prior mean

cov_7 = np.zeros((n_params, n_params))
cov_7[:5, :5] = cov_3
cov_7[5, 5] = 1.0  # logD_raw[3] prior variance
cov_7[6, 6] = 1.0  # phi_raw[2] prior variance

# Cholesky decomposition: theta = mu + L @ z
L_np = np.linalg.cholesky(cov_7)
log_det_L = np.sum(np.log(np.diag(L_np)))

mu = jnp.array(mu_7, dtype=jnp.float32)
L = jnp.array(L_np, dtype=jnp.float32)
log_det_L_jax = jnp.float32(log_det_L)

print(f"  D3_gpu: {d3_theta.shape[0]} samples, {d3_theta.shape[1]} params")
print(f"  mu (padded to 7): {mu_7}")
print(f"  Cov eigenvalues: {np.linalg.eigvalsh(cov_7)}")
print(f"  log|det(L)| = {log_det_L:.2f}")

# ── Forward model (same as run_gpu.py, 4 domains, Ea=117 fixed) ─
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

def log_prob_theta(logD_raw, phi_raw):
    """Log probability in original theta-space."""
    logD = jnp.cumsum(jax.nn.softplus(logD_raw)) + logD_raw[0]
    phi = stick_breaking(phi_raw)
    D_r2 = jnp.exp(logD)[:, None]

    T = temps_K[None, :]
    Ea_broadcast = jnp.full((num_domains, 1), Ea_fixed_val)

    y_inc = D_r2 * jnp.exp(-Ea_broadcast * 1e3 / (R_gas * T)) * dt
    y_cum = jnp.cumsum(y_inc, axis=1)
    Fcum = fractional_release(y_cum)
    dF = jnp.diff(Fcum, axis=1, prepend=jnp.zeros((num_domains, 1)))

    dF_mix = phi @ dF
    F_final_mix = phi @ Fcum[..., -1]

    f_pred = dF_mix / (F_final_mix + 1e-30)

    # Likelihood
    ll = jnp.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2
                 - jnp.log(sigma_obs) - 0.5 * jnp.log(2 * jnp.pi))

    # Priors
    prior = jnp.sum(-0.5 * (logD - logD_prior_mean)**2 / logD_prior_std**2)
    prior += jnp.sum((1.0 - 1) * jnp.log(phi))  # Dirichlet(1) = uniform, contributes 0

    return prior + ll

def log_prob_whitened(z):
    """Log probability in whitened z-space.

    z is a flat vector of length n_params.
    theta = mu + L @ z
    log_prob_whitened(z) = log_prob_theta(theta) + log|det(L)|
    The Jacobian log|det(L)| is constant, so it doesn't affect MCMC
    sampling (NUTS only needs gradients), but we include it for correctness.
    """
    theta = mu + L @ z
    logD_raw = theta[:num_domains]
    phi_raw = theta[num_domains:]
    return log_prob_theta(logD_raw, phi_raw) + log_det_L_jax

# ── Init ────────────────────────────────────────────────────────
key = random.PRNGKey(args.seed)
chain_keys = random.split(key, args.num_chains)

# Initialize in z-space: z=0 corresponds to the D3_gpu posterior mean
# Add small noise for chain diversity
init_z = jax.vmap(lambda k: random.normal(k, (n_params,)) * 0.1)(chain_keys)
print(f"Initialized {args.num_chains} chains in z-space (near z=0 = D3_gpu mean)")

# Verify log_prob works
test_lp = log_prob_whitened(init_z[0])
print(f"  Test log_prob at z=0-ish: {float(test_lp):.1f}")

# ── Warmup (looped in 1000-step chunks) ────────────────────────
WARMUP_CHUNK = 1000
n_warmup_chunks = max(1, args.num_warmup // WARMUP_CHUNK)
print(f"Warming up: {n_warmup_chunks} chunks of {WARMUP_CHUNK} steps = {n_warmup_chunks * WARMUP_CHUNK} total")

t0 = time.time()
current_positions = init_z
current_step_size = 1e-2  # whitened space should be ~unit scale

for chunk_i in range(n_warmup_chunks):
    chunk_key = random.PRNGKey(args.seed + 1000 + chunk_i)
    chunk_keys = random.split(chunk_key, args.num_chains)

    def warmup_one(key, init_z):
        warm = blackjax.window_adaptation(
            blackjax.nuts, log_prob_whitened, initial_step_size=current_step_size,
            is_mass_matrix_diagonal=False, target_acceptance_rate=0.8,
            max_num_doublings=8)
        (state, tuned), _ = warm.run(key, init_z, num_steps=WARMUP_CHUNK)
        return state, tuned["step_size"], tuned["inverse_mass_matrix"]

    states, step_sizes, inv_mats = jax.vmap(warmup_one)(chunk_keys, current_positions)
    jax.block_until_ready(states)

    ss = np.array(step_sizes)
    current_step_size = float(jnp.median(step_sizes))
    current_positions = states.position

    if (chunk_i + 1) % 1 == 0:
        elapsed = time.time() - t0
        print(f"  Chunk {chunk_i+1}/{n_warmup_chunks}: {elapsed:.0f}s, "
              f"step_sizes: [{ss.min():.2e}, {ss.max():.2e}], "
              f"active: {(ss > 0).sum()}/{args.num_chains}")

warmup_time = time.time() - t0
print(f"  Warmup done: {warmup_time:.1f}s")

# ── Sampling (continuous adaptation) ───────────────────────────
sample_step_size = current_step_size

def sample_one(key, init_z):
    warm = blackjax.window_adaptation(
        blackjax.nuts, log_prob_whitened, initial_step_size=sample_step_size,
        is_mass_matrix_diagonal=False, target_acceptance_rate=0.8,
        max_num_doublings=8)
    (state, tuned), _ = warm.run(key, init_z, num_steps=args.steps_per_loop)
    return state, tuned["step_size"], tuned["inverse_mass_matrix"]

print(f"Compiling {args.steps_per_loop}-step adaptive kernel...")
t0c = time.time()
compile_keys = random.split(random.PRNGKey(args.seed + 9999), args.num_chains)
_states, _ss, _im = jax.vmap(sample_one)(compile_keys, current_positions)
jax.block_until_ready(_states)
compile_time = time.time() - t0c
print(f"  Compiled in {compile_time:.1f}s")

print(f"Sampling {args.num_loops} loops of {args.steps_per_loop} steps (continuous adaptation)...")
all_z_positions = []
t0 = time.time()

for loop in range(args.num_loops):
    loop_key = random.PRNGKey(args.seed + 2000 + loop)
    loop_keys = random.split(loop_key, args.num_chains)

    states, step_sizes, inv_mats = jax.vmap(sample_one)(loop_keys, current_positions)
    jax.block_until_ready(states)

    current_positions = states.position
    all_z_positions.append(jax.device_get(states.position))

    elapsed = time.time() - t0
    if (loop + 1) % 10 == 0:
        ss = np.array(step_sizes)
        print(f"  Loop {loop+1}/{args.num_loops}: {elapsed:.1f}s, "
              f"step_sizes: [{ss.min():.2e}, {ss.max():.2e}]")

sample_time = time.time() - t0
total_steps = args.steps_per_loop * args.num_loops
print(f"  Sampling done: {sample_time:.1f}s for {total_steps} steps x {args.num_chains} chains")
print(f"  = {total_steps * args.num_chains / sample_time:.0f} chain-steps/sec")

# ── Transform back to theta-space ──────────────────────────────
print("\nTransforming z-space samples back to theta-space...")
z_all = np.stack(all_z_positions)  # (num_loops, num_chains, n_params)
mu_np = np.array(mu)
L_np_f = np.array(L)

# theta = mu + L @ z  for each sample
theta_all = np.einsum('ij,tcj->tci', L_np_f, z_all) + mu_np[None, None, :]
# (num_loops, num_chains, n_params)

# Split into logD_raw and phi_raw
logD_raw_all = theta_all[:, :, :num_domains]  # (loops, chains, 4)
phi_raw_all = theta_all[:, :, num_domains:]    # (loops, chains, 3)

# ── Post-process (same as run_gpu.py) ──────────────────────────
fs_np = {}
fs_np["logD_raw"] = logD_raw_all.astype(np.float32)
fs_np["phi_raw"] = phi_raw_all.astype(np.float32)
fs_np["z"] = z_all.astype(np.float32)

# Compute logD from logD_raw (apply ordering constraint)
logD = np.cumsum(np.log(1 + np.exp(logD_raw_all)), axis=-1) + logD_raw_all[..., :1]
fs_np["logD0_r2"] = logD.astype(np.float32)

# Compute phi from phi_raw (stick-breaking)
def np_stick_breaking(pr):
    probs = 1 / (1 + np.exp(-pr))
    remaining = np.concatenate([[1.0], np.cumprod(1 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])

phi = np.array([[np_stick_breaking(phi_raw_all[t, c])
                 for c in range(args.num_chains)]
                for t in range(args.num_loops)])
fs_np["phi"] = phi

# ── Convergence ────────────────────────────────────────────────
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
# Report on z-space parameters
for i in range(n_params):
    rhat = split_rhat(z_all[:, :, i])
    print(f"  z[{i}]: R-hat = {rhat:.3f} {'OK' if rhat < 1.1 else 'WARNING'}")

# Report on theta-space derived quantities
print("\n  -- Derived (theta-space) --")
for key in ["logD_raw", "phi_raw", "logD0_r2", "phi"]:
    arr = fs_np[key]
    if arr.ndim == 3:
        for d in range(arr.shape[2]):
            rhat = split_rhat(arr[:, :, d])
            print(f"  {key}[{d}]: R-hat = {rhat:.3f} {'OK' if rhat < 1.1 else 'WARNING'}")

# ── Parameter summaries ────────────────────────────────────────
print("\n=== Parameter Summaries (last loop) ===")
last_logD = fs_np["logD0_r2"][-1]  # (chains, 4)
last_phi = fs_np["phi"][-1]        # (chains, 4)

for d in range(num_domains):
    med = np.median(last_logD[:, d])
    lo, hi = np.percentile(last_logD[:, d], [2.5, 97.5])
    print(f"  logD0/r2[{d}]: median={med:.2f}, 95%CI=[{lo:.2f}, {hi:.2f}]")

for d in range(num_domains):
    med = np.median(last_phi[:, d])
    lo, hi = np.percentile(last_phi[:, d], [2.5, 97.5])
    print(f"  phi[{d}]: median={med:.4f}, 95%CI=[{lo:.4f}, {hi:.4f}]")

# ── T-t constraints ───────────────────────────────────────────
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

hrd_logD = last_logD[:, 0]  # smallest logD = most retentive
hrd_Ea = np.full(len(hrd_logD), Ea_fixed_val)
nc = len(hrd_logD)

print(f"\n=== T-t Constraints ({nc} chains) ===")
for dur, label in [(10, "10 My"), (100, "100 My"), (200, "200 My"), (500, "500 My"), (1300, "Isothermal")]:
    temps = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea[:nc], hrd_logD[:nc])])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med = np.median(valid)
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med:6.1f} C  95%CI=[{lo:.1f}, {hi:.1f}]  crosses 0 C: {hi > 0}")

# ── Save ───────────────────────────────────────────────────────
result = {
    "samples": fs_np,
    "args": vars(args),
    "warmup_time": warmup_time,
    "sample_time": sample_time,
    "compile_time": compile_time,
    "whitening": {
        "mu": mu_np,
        "L": L_np_f,
        "log_det_L": float(log_det_L),
        "source": "results/fit_D3_gpu.pkl",
    },
}
pickle.dump(result, open(f"results/fit_{args.tag}.pkl", "wb"))
print(f"\nSaved results/fit_{args.tag}.pkl")
print(f"Total time: warmup={warmup_time:.0f}s, compile={compile_time:.0f}s, sample={sample_time:.0f}s")
