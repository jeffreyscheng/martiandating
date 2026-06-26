"""Sequential Monte Carlo via blackjax.adaptive_tempered_smc."""
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/jefcheng/.cache/jax'
import time, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')

import jax
import jax.numpy as jnp
from jax import random
import blackjax
from blackjax.smc.resampling import systematic
from scipy.optimize import brentq

print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")

parser = argparse.ArgumentParser()
parser.add_argument("--num_domains", type=int, default=3)
parser.add_argument("--num_particles", type=int, default=5000)
parser.add_argument("--num_mcmc_steps", type=int, default=20)
parser.add_argument("--tag", type=str, default="SMC")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ea_mode", type=str, default="shared", choices=["fixed", "shared"])
parser.add_argument("--target_ess_frac", type=float, default=0.5)
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
D_obs_arr = dy_obs / filt['seconds_per_extraction_step'].values
D_obs_arr[D_obs_arr <= 0] = np.nan
lnD_obs = np.log(D_obs_arr)

print(f"Data: {n_data} steps, {num_domains} domains, {args.num_particles} particles")

# ── Model (split into prior and likelihood for SMC) ──
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

def log_prior(params):
    logD_raw = params["logD_raw"]
    logD = jnp.cumsum(jax.nn.softplus(logD_raw)) + logD_raw[0]
    prior = jnp.sum(-0.5 * (logD - 6.0)**2 / 1.5**2)
    if args.ea_mode == "shared":
        prior += -0.5 * (params["Ea_shared"] - 117.0)**2 / 10.0**2
    return prior

def log_likelihood(params):
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

    return jnp.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2
                   - jnp.log(sigma_obs) - 0.5 * jnp.log(2 * jnp.pi))

# ── Initialize particles from prior ──
print("Initializing particles from prior...")
key = random.PRNGKey(args.seed)
keys = random.split(key, args.num_particles)

def sample_prior(key):
    k1, k2, k3 = random.split(key, 3)
    params = {
        "logD_raw": jnp.array([2.0, 0.5, 0.3])[:num_domains] + random.normal(k1, (num_domains,)) * 1.5,
        "phi_raw": random.normal(k2, (num_domains - 1,)) * 1.0,
    }
    if args.ea_mode == "shared":
        params["Ea_shared"] = 117.0 + random.normal(k3) * 10.0
    return params

init_particles = jax.vmap(sample_prior)(keys)

# ── Set up SMC ──
print("Setting up adaptive tempered SMC...")

# NUTS as mutation kernel
n_params = num_domains + (num_domains - 1) + (1 if args.ea_mode == "shared" else 0)
inv_mass_matrix = jnp.eye(n_params) * 0.01

nuts_kernel = blackjax.nuts.build_kernel()
nuts_init = blackjax.nuts.init

mcmc_parameters = blackjax.smc.extend_params(
    args.num_particles,
    {"step_size": 1e-3, "inverse_mass_matrix": inv_mass_matrix}
)

smc = blackjax.adaptive_tempered_smc(
    log_prior,
    log_likelihood,
    nuts_kernel,
    nuts_init,
    mcmc_parameters,
    systematic,
    target_ess=args.target_ess_frac * args.num_particles,
    num_mcmc_steps=args.num_mcmc_steps,
)

# ── Run SMC ──
print("Running SMC (tempering from prior to posterior)...")
t0 = time.time()

smc_key = random.PRNGKey(args.seed + 1000)
state = smc.init(init_particles)

step = 0
while state.lmbda < 1.0:
    step_key, smc_key = random.split(smc_key)
    state, info = smc.step(step_key, state)
    step += 1

    elapsed = time.time() - t0
    ess_val = 1.0 / jnp.sum(jnp.exp(2 * state.weights))
    print(f"  Step {step}: λ={float(state.lmbda):.6f}, ESS={float(ess_val):.0f}, "
          f"elapsed={elapsed:.0f}s")

    if step > 500:
        print("WARNING: exceeded 500 steps, stopping")
        break

smc_time = time.time() - t0
print(f"SMC done: {step} steps in {smc_time:.0f}s, final λ={float(state.lmbda):.6f}")

# ── Extract posterior samples ──
print("\nExtracting posterior samples...")
particles = state.particles
weights = jnp.exp(state.weights)
weights = weights / weights.sum()

n_particles = args.num_particles

# Transform to physical parameters
def np_stick_breaking(pr):
    probs = 1 / (1 + np.exp(-pr))
    remaining = np.concatenate([[1.0], np.cumprod(1 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])

all_logD = []
all_phi = []
all_Ea = []

for i in range(n_particles):
    logD_raw = np.array(particles['logD_raw'][i])
    logD = np.cumsum(np.log(1 + np.exp(logD_raw))) + logD_raw[0]
    phi = np_stick_breaking(np.array(particles['phi_raw'][i]))
    all_logD.append(logD)
    all_phi.append(phi)
    if 'Ea_shared' in particles:
        all_Ea.append(float(particles['Ea_shared'][i]))
    else:
        all_Ea.append(Ea_fixed_val)

all_logD = np.array(all_logD)
all_phi = np.array(all_phi)
all_Ea = np.array(all_Ea)
weights_np = np.array(weights)

# ── Effective Arrhenius ──
print("Computing effective Arrhenius...")

def fr_np(y):
    return max(6*np.sqrt(y/np.pi)-3*y, 0) if y < 0.3 else min(1-(6/np.pi**2)*np.exp(-np.pi**2*y), 1)

tK = filt['T_K'].values
dt_arr = filt['seconds_per_extraction_step'].values
all_lnD_eff = []

for i in range(n_particles):
    logD = all_logD[i]
    phi = all_phi[i]
    Ea = all_Ea[i]
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

if len(all_lnD_eff) > 0:
    lo = np.percentile(all_lnD_eff, 2.5, axis=0)
    hi = np.percentile(all_lnD_eff, 97.5, axis=0)
    covered = np.array([(lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i]) if np.isfinite(lnD_obs[i]) else True
                        for i in range(n_data)])
    n_covered = covered.sum()
    print(f"Arrhenius coverage: {n_covered}/{n_data}")
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

print(f"\n=== T-t Constraints ({n_particles} particles) ===")
for dur, label in [(10,"10 My"),(100,"100 My"),(200,"200 My"),(500,"500 My"),(1300,"Isothermal")]:
    temps = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
    valid = temps[~np.isnan(temps)]
    if len(valid) > 0:
        med = np.median(valid)
        lo_t, hi_t = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med:6.1f}°C  95%CI=[{lo_t:.1f}, {hi_t:.1f}]  crosses 0°C: {hi_t > 0}")

# ── Parameter summaries ──
print(f"\n=== Parameter Summaries ===")
print(f"logD: median={np.median(all_logD, axis=0).round(2)}, std={np.std(all_logD, axis=0).round(3)}")
print(f"phi:  median={np.median(all_phi, axis=0).round(3)}, std={np.std(all_phi, axis=0).round(3)}")
print(f"Ea:   median={np.median(all_Ea):.2f}, std={np.std(all_Ea):.2f}")

# ── Save ──
fs_np = {
    'logD0_r2': all_logD.reshape(1, -1, num_domains),
    'phi': all_phi.reshape(1, -1, num_domains),
}
if args.ea_mode == "shared":
    fs_np['Ea_shared'] = np.array(all_Ea).reshape(1, -1)

result = {
    'samples': fs_np,
    'args': vars(args),
    'smc_time': smc_time,
    'n_smc_steps': step,
    'final_lambda': float(state.lmbda),
    'weights': weights_np,
}

out_path = f"results/fit_{args.tag}.pkl"
pickle.dump(result, open(out_path, 'wb'))
print(f"\nSaved {out_path}")
print(f"Total time: {time.time() - t0:.0f}s")
