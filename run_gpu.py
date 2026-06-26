"""GPU-optimized MDD fit: compile once with small scan, loop in Python."""
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

n_devices = len(jax.devices())
print(f"Backend: {jax.default_backend()}, Devices: {n_devices}x {jax.devices()[0].platform}")

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_domains", type=int, default=3)
parser.add_argument("--num_chains", type=int, default=100)
parser.add_argument("--num_warmup", type=int, default=5000)
parser.add_argument("--steps_per_loop", type=int, default=1000)
parser.add_argument("--num_loops", type=int, default=50)
parser.add_argument("--tag", type=str, default="gpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ea_mode", type=str, default="fixed", choices=["fixed", "shared", "per_domain", "t_dependent"])
parser.add_argument("--learn_sigma", action="store_true")
parser.add_argument("--noise_penalty_rate", type=float, default=5.0, help="Exponential penalty rate on noise scale")
parser.add_argument("--dirichlet_alpha", type=float, default=1.0, help="Dirichlet concentration parameter")
parser.add_argument("--init_from", type=str, default=None, help="Initialize from a previous run's pkl")
parser.add_argument("--init_mode", type=str, default="random", choices=["random", "pkl", "map"], help="Initialization strategy")
parser.add_argument("--warmup_sigma_inflate", type=float, default=1.0, help="Inflate sigma during warmup, then deflate step sizes")
parser.add_argument("--temperature", type=float, default=1.0, help="Posterior temperature: sample p(theta|D)^(1/T)")
parser.add_argument("--diagonal_mass", action="store_true", help="Use diagonal mass matrix instead of dense")
parser.add_argument("--phi_scale", type=float, default=1.0, help="Scale phi_raw before sigmoid (smaller = wider phi_raw posterior)")
parser.add_argument("--noise_floor", action="store_true", help="Learn additive noise floor (usually collapses domains)")
parser.add_argument("--sigma_floor", type=float, default=0.0, help="Fixed additive noise floor: sigma_total^2 = sigma_meas^2 + sigma_floor^2")
parser.add_argument("--hessian_init", action="store_true", help="Compute Hessian at init to set step size")
args = parser.parse_args()

num_domains = args.num_domains
total_steps = args.steps_per_loop * args.num_loops

# ── Data ────────────────────────────────────────────────────────
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
if args.sigma_floor > 0:
    print(f"Noise floor: {args.sigma_floor:.4f} (effective sigma: [{float(sigma_obs.min()):.5f}, {float(sigma_obs.max()):.5f}])")
n_data = len(temps_K)
print(f"Data: {n_data} steps, {num_domains} domains, {args.num_chains} chains")
print(f"Sampling: {args.num_warmup} warmup + {args.steps_per_loop} steps x {args.num_loops} loops = {total_steps} total")

# ── Model ───────────────────────────────────────────────────────
Ea_fixed_val = 117.0
R_gas = 8.314
logD_prior_mean = 5.7
logD_prior_std = 2.0
Ea_prior_mean = 117.0
Ea_prior_std = 20.0

def stick_breaking(phi_raw):
    probs = jax.nn.sigmoid(phi_raw * args.phi_scale)
    remaining = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(1.0 - probs)])
    return jnp.concatenate([probs * remaining[:-1], remaining[-1:]])

def fractional_release(y):
    F_short = 6 * jnp.sqrt(y / jnp.pi) - 3 * y
    F_long = 1 - (6 / jnp.pi**2) * jnp.exp(-jnp.pi**2 * y)
    return jnp.clip(jnp.where(y < 0.3, F_short, F_long), 0.0, 1.0)

def log_prob(params):
    logD_raw = params["logD_raw"]
    # Order constraint on logD
    logD = jnp.cumsum(jax.nn.softplus(logD_raw)) + logD_raw[0]

    phi_raw = params["phi_raw"]
    phi = stick_breaking(phi_raw)
    D_r2 = jnp.exp(logD)[:, None]
    nd = num_domains

    T = temps_K[None, :]

    # Ea handling
    if args.ea_mode == "fixed":
        Ea_broadcast = jnp.full((nd, 1), Ea_fixed_val)
    elif args.ea_mode == "shared":
        Ea_broadcast = jnp.full((nd, 1), params["Ea_shared"])
    elif args.ea_mode == "per_domain":
        Ea_broadcast = params["Ea_per"][:, None]
    else:  # t_dependent
        Ea_vals = params["Ea_base"] + params["Ea_slope"] * (temps_K - 773.15)
        Ea_broadcast = Ea_vals[None, :]

    y_inc = D_r2 * jnp.exp(-Ea_broadcast * 1e3 / (R_gas * T)) * dt
    y_cum = jnp.cumsum(y_inc, axis=1)
    Fcum = fractional_release(y_cum)
    dF = jnp.diff(Fcum, axis=1, prepend=jnp.zeros((nd, 1)))

    dF_mix = phi @ dF
    F_final_mix = phi @ Fcum[..., -1]

    obs_sigma = sigma_obs
    if args.learn_sigma:
        obs_sigma = sigma_obs * jnp.exp(params["log_sigma_scale"])
    if args.noise_floor:
        sigma_floor = jnp.exp(params["log_sigma_floor"])
        obs_sigma = jnp.sqrt(sigma_obs**2 + sigma_floor**2)

    f_pred = dF_mix / (F_final_mix + 1e-30)

    ll = jnp.sum(-0.5 * ((f_obs - f_pred) / obs_sigma)**2
                 - jnp.log(obs_sigma) - 0.5 * jnp.log(2 * jnp.pi))

    # Priors
    prior = jnp.sum(-0.5 * (logD - logD_prior_mean)**2 / logD_prior_std**2)
    if num_domains > 1:
        prior += jnp.sum((args.dirichlet_alpha - 1) * jnp.log(phi))
    if args.ea_mode == "shared":
        prior += -0.5 * (params["Ea_shared"] - Ea_prior_mean)**2 / Ea_prior_std**2
    elif args.ea_mode == "per_domain":
        prior += jnp.sum(-0.5 * (params["Ea_per"] - Ea_prior_mean)**2 / Ea_prior_std**2)
    elif args.ea_mode == "t_dependent":
        prior += -0.5 * (params["Ea_base"] - Ea_prior_mean)**2 / Ea_prior_std**2
        prior += -0.5 * params["Ea_slope"]**2 / 0.5**2
    if args.learn_sigma:
        prior += -args.noise_penalty_rate * jnp.exp(params["log_sigma_scale"])
    if args.noise_floor:
        prior += -0.5 * (params["log_sigma_floor"] + 8.0)**2 / 3.0**2

    return prior + ll / args.temperature

# ── Multi-GPU helpers ──────────────────────────────────────────
chains_per_device = args.num_chains // n_devices
if args.num_chains % n_devices != 0:
    chains_per_device += 1
    args.num_chains = chains_per_device * n_devices
    print(f"  Padded to {args.num_chains} chains ({chains_per_device} per device)")

def shard(x):
    """Reshape (num_chains, ...) -> (n_devices, chains_per_device, ...)"""
    return jax.tree.map(lambda a: a.reshape(n_devices, chains_per_device, *a.shape[1:]), x)

def unshard(x):
    """Reshape (n_devices, chains_per_device, ...) -> (num_chains, ...)"""
    return jax.tree.map(lambda a: a.reshape(args.num_chains, *a.shape[2:]), x)

print(f"  {n_devices} devices x {chains_per_device} chains/device = {args.num_chains} chains")

# ── Init ────────────────────────────────────────────────────────
key = random.PRNGKey(args.seed)
chain_keys = random.split(key, args.num_chains)

def init_params(key):
    k1, k2, k3, k4 = random.split(key, 4)
    params = {"logD_raw": random.normal(k1, (num_domains,)) * 1.0}
    params["phi_raw"] = random.normal(k2, (num_domains - 1,)) * 0.5
    if args.ea_mode == "shared":
        params["Ea_shared"] = Ea_prior_mean + random.normal(k2) * 5.0
    elif args.ea_mode == "per_domain":
        params["Ea_per"] = Ea_prior_mean + random.normal(k2, (num_domains,)) * 5.0
    elif args.ea_mode == "t_dependent":
        params["Ea_base"] = Ea_prior_mean + random.normal(k2) * 5.0
        params["Ea_slope"] = random.normal(k3) * 0.1
    if args.learn_sigma:
        params["log_sigma_scale"] = jnp.array(0.0)
    if args.noise_floor:
        params["log_sigma_floor"] = random.normal(k4) * 1.0 - 8.0
    return params

if args.init_from and args.init_mode == "pkl":
    print(f"Initializing from {args.init_from}...")
    prev = pickle.load(open(args.init_from, 'rb'))
    prev_samples = prev['samples']
    n_prev = prev_samples['logD_raw'].shape[1]
    dummy = jax.vmap(init_params)(chain_keys)
    init_batch = {}
    for k in dummy:
        if k in prev_samples:
            vals = prev_samples[k][-1]  # (prev_chains, prev_dims)
            # Tile chains if needed
            if n_prev >= args.num_chains:
                tiled = vals[:args.num_chains]
            else:
                reps = args.num_chains // n_prev + 1
                tiled = np.tile(vals, (reps,) + (1,) * (vals.ndim - 1))[:args.num_chains]
            # Pad dimensions if model has more domains than pkl
            target_shape = dummy[k].shape
            if tiled.shape != target_shape:
                padded = np.array(dummy[k])  # start with random init
                min_d = min(tiled.shape[-1], target_shape[-1]) if tiled.ndim > 1 else 1
                if tiled.ndim > 1:
                    padded[:, :min_d] = tiled[:, :min_d]
                else:
                    padded[:] = tiled[:]
                init_batch[k] = jnp.array(padded)
            else:
                init_batch[k] = jnp.array(tiled)
        else:
            init_batch[k] = dummy[k]
    print(f"  Loaded {n_prev} chains from previous run (padded to {num_domains} domains)")
elif args.init_mode == "map":
    import optax
    print("Finding MAP estimate via Adam (500 restarts × 5000 steps)...")
    neg_log_prob = jax.jit(lambda p: -log_prob(p))
    grad_fn = jax.jit(jax.grad(neg_log_prob))

    best_lp = -jnp.inf
    best_params = None
    n_map_trials = 500
    for trial in range(n_map_trials):
        trial_key = random.PRNGKey(args.seed + trial)
        params = init_params(trial_key)
        opt = optax.adam(1e-3)
        opt_state = opt.init(params)
        for step in range(5000):
            g = grad_fn(params)
            updates, opt_state = opt.update(g, opt_state, params)
            params = optax.apply_updates(params, updates)
        lp = float(log_prob(params))
        if lp > best_lp:
            best_lp = lp
            best_params = params
        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial+1}/{n_map_trials}: best logprob={best_lp:.1f}")
    print(f"  Best MAP logprob: {best_lp:.1f}")
    logD_map = jnp.cumsum(jax.nn.softplus(best_params['logD_raw'])) + best_params['logD_raw'][0]
    phi_map = stick_breaking(best_params['phi_raw']) if 'phi_raw' in best_params else jnp.ones(num_domains)/num_domains
    print(f"  MAP logD: {logD_map}")
    print(f"  MAP phi: {phi_map}")
    init_batch = {}
    for k in best_params:
        val = best_params[k]
        noise = random.normal(random.PRNGKey(args.seed), (args.num_chains,) + val.shape) * 0.05
        init_batch[k] = val[None, ...] + noise
    print(f"  Initialized {args.num_chains} chains around MAP")
else:
    init_batch = jax.vmap(init_params)(chain_keys)

# ── Warmup: Pathfinder + window adaptation ─────────────────────
t0 = time.time()
current_positions = init_batch
current_step_size = 1e-3

print(f"Warmup: pathfinder_adaptation ({args.num_warmup} steps) per chain...")

def pathfinder_warmup_one(key, init_p):
    adapt = blackjax.pathfinder_adaptation(
        blackjax.nuts, log_prob,
        target_acceptance_rate=0.8)
    (state, tuned), _ = adapt.run(key, init_p, num_steps=args.num_warmup)
    return state, tuned["step_size"], tuned["inverse_mass_matrix"]

warmup_keys = random.split(random.PRNGKey(args.seed + 5000), args.num_chains)
states, step_sizes, inv_mats = jax.pmap(jax.vmap(pathfinder_warmup_one))(
    shard(warmup_keys), shard(current_positions))
jax.block_until_ready(states)

ss = np.array(unshard(step_sizes))
current_step_size = float(jnp.median(ss))
current_positions = unshard(states.position)

warmup_time = time.time() - t0
print(f"  Warmup done: {warmup_time:.1f}s")
print(f"  Step sizes: [{ss.min():.2e}, {ss.max():.2e}], active: {(ss > 0).sum()}/{args.num_chains}")

# Show where pathfinder put the chains
for k in current_positions:
    vals = np.array(current_positions[k])
    if vals.ndim > 1:
        print(f"  {k}: mean={np.mean(vals, axis=0).round(2)}, std={np.std(vals, axis=0).round(3)}")
    else:
        print(f"  {k}: mean={np.mean(vals):.2f}, std={np.std(vals):.3f}")

# ── Sampling (continuous adaptation — every chunk is warmup) ──────
sample_step_size = current_step_size

def sample_one(key, init_p):
    warm = blackjax.window_adaptation(
        blackjax.nuts, log_prob, initial_step_size=sample_step_size,
        is_mass_matrix_diagonal=args.diagonal_mass, target_acceptance_rate=0.8,
        max_num_doublings=8)
    (state, tuned), _ = warm.run(key, init_p, num_steps=args.steps_per_loop)
    return state, tuned["step_size"], tuned["inverse_mass_matrix"]

pmap_sample = jax.pmap(jax.vmap(sample_one))

print(f"Compiling {args.steps_per_loop}-step adaptive kernel ({n_devices} devices)...")
t0 = time.time()
compile_keys = random.split(random.PRNGKey(args.seed + 9999), args.num_chains)
_states, _ss, _im = pmap_sample(shard(compile_keys), shard(current_positions))
jax.block_until_ready(_states)
compile_time = time.time() - t0
print(f"  Compiled in {compile_time:.1f}s")

print(f"Sampling {args.num_loops} loops of {args.steps_per_loop} steps (continuous adaptation)...")
all_positions = []
t0 = time.time()

for loop in range(args.num_loops):
    loop_key = random.PRNGKey(args.seed + 2000 + loop)
    loop_keys = random.split(loop_key, args.num_chains)

    states, step_sizes, inv_mats = pmap_sample(shard(loop_keys), shard(current_positions))
    jax.block_until_ready(states)

    current_positions = unshard(states.position)
    all_positions.append(jax.device_get(unshard(states.position)))

    elapsed = time.time() - t0
    if (loop + 1) % 10 == 0:
        ss = np.array(unshard(step_sizes))
        print(f"  Loop {loop+1}/{args.num_loops}: {elapsed:.1f}s, step_sizes: [{ss.min():.2e}, {ss.max():.2e}]")

sample_time = time.time() - t0
print(f"  Sampling done: {sample_time:.1f}s for {total_steps} steps x {args.num_chains} chains")
print(f"  = {total_steps * args.num_chains / sample_time:.0f} chain-steps/sec")

# ── Post-process ────────────────────────────────────────────────
# Stack: (num_loops, chains, params)
fs_np = {}
for key in all_positions[0]:
    fs_np[key] = np.stack([p[key] for p in all_positions])

# Compute logD from logD_raw (apply ordering constraint)
if "logD_raw" in fs_np and fs_np["logD_raw"].ndim == 3:
    logD_raw = fs_np["logD_raw"]
    logD = np.cumsum(np.log(1 + np.exp(logD_raw)), axis=-1) + logD_raw[..., :1]
    fs_np["logD0_r2"] = logD

# Compute phi if discrete model
def np_stick_breaking(pr):
    probs = 1 / (1 + np.exp(-pr * args.phi_scale))
    remaining = np.concatenate([[1.0], np.cumprod(1 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])

if "phi_raw" in fs_np and fs_np["phi_raw"].ndim == 3:
    phi_raw = fs_np["phi_raw"]
    phi = np.array([[np_stick_breaking(phi_raw[t, c]) for c in range(args.num_chains)]
                    for t in range(args.num_loops)])
    fs_np["phi"] = phi
elif "phi_raw" in fs_np and fs_np["phi_raw"].ndim == 2:
    fs_np["phi"] = np.ones((args.num_loops, args.num_chains, 1))


# Convergence
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
    B = n/(m-1)*np.sum((chain_means - chain_means.mean())**2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    return np.sqrt(((n-1)/n*W + B/n)/W) if W > 0 else np.nan

print("\n=== Convergence (R-hat on second half of samples) ===")
for key in fs_np:
    arr = fs_np[key]
    if arr.ndim == 3:
        for d in range(arr.shape[2]):
            rhat = split_rhat(arr[:, :, d])
            print(f"  {key}[{d}]: R-hat = {rhat:.3f} {'OK' if rhat < 1.1 else 'WARNING'}")
    elif arr.ndim == 2:
        rhat = split_rhat(arr)
        print(f"  {key}: R-hat = {rhat:.3f} {'OK' if rhat < 1.1 else 'WARNING'}")

# T-t constraint
from scipy.optimize import brentq
def max_temp(Ea_kJ, logD0_r2_val, dur_Ma, max_loss=0.01):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD0_r2_val)
    def f(T_C):
        T_K = T_C + 273.15
        y = D0r2 * np.exp(-Ea_kJ*1e3/(R_gas*T_K)) * dur_s
        F = 6*np.sqrt(y/np.pi)-3*y if y<0.3 else 1-(6/np.pi**2)*np.exp(-np.pi**2*y)
        return np.clip(F,0,1) - max_loss
    try: return brentq(f, -273, 1000, xtol=0.1)
    except: return np.nan

# Get HRD parameters for T-t constraint
if "logD0_r2" in fs_np and fs_np["logD0_r2"].ndim == 3:
    final_logD = fs_np["logD0_r2"][-1]
    hrd_logD = final_logD[:, 0]  # ordered, [0] is smallest = HRD
else:
    hrd_logD = fs_np["logD_raw"][-1][:, 0] if fs_np["logD_raw"].ndim == 3 else fs_np["logD_raw"][-1]

# Get Ea for each chain
if args.ea_mode == "fixed":
    hrd_Ea = np.full(len(hrd_logD), Ea_fixed_val)
elif args.ea_mode == "shared":
    hrd_Ea = fs_np["Ea_shared"][-1]
elif args.ea_mode == "per_domain":
    hrd_Ea = fs_np.get("Ea_per", np.full((1, len(hrd_logD), num_domains), Ea_fixed_val))[-1][:, 0]
elif args.ea_mode == "t_dependent":
    hrd_Ea = fs_np.get("Ea_base", np.full((1, len(hrd_logD)), Ea_fixed_val))[-1]
else:
    hrd_Ea = np.full(len(hrd_logD), Ea_fixed_val)

nc = len(hrd_logD)
print(f"\n=== T-t Constraints ({nc} chains) ===")
for dur, label in [(10,"10 My"),(100,"100 My"),(200,"200 My"),(500,"500 My"),(1300,"Isothermal")]:
    temps = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea[:nc], hrd_logD[:nc])])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med = np.median(valid)
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med:6.1f}°C  95%CI=[{lo:.1f}, {hi:.1f}]  crosses 0°C: {hi > 0}")

# Save
pickle.dump({"samples": fs_np, "args": vars(args),
             "warmup_time": warmup_time, "sample_time": sample_time,
             "compile_time": compile_time},
            open(f"results/fit_{args.tag}.pkl", "wb"))
print(f"\nSaved results/fit_{args.tag}.pkl")
print(f"Total time: warmup={warmup_time:.0f}s, compile={compile_time:.0f}s, sample={sample_time:.0f}s")
# This file was already created above - see the main content
