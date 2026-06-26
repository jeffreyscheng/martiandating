"""
Standalone MDD fitting script. Run with: python run_fit.py [--num_domains 2] [--tag test]
Saves results to results/ and produces diagnostic plots.
"""
import argparse, os, pickle, time
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/home/jefcheng/.cache/jax")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from scipy.optimize import brentq
from scipy.stats import gaussian_kde

import jax
import jax.numpy as jnp
from jax import random
import blackjax

# ── Parse args ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_domains", type=int, default=2)
parser.add_argument("--num_chains", type=int, default=200)
parser.add_argument("--num_warmup", type=int, default=1000)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--ea_sigma", type=float, default=2.0)
parser.add_argument("--repulsion", type=float, default=100.0)
parser.add_argument("--shared_ea", action="store_true", help="Force shared Ea across domains")
parser.add_argument("--ea_fixed", action="store_true", help="Fix Ea at 117 kJ/mol")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_temp", type=int, default=0, help="Exclude steps below this temp (°C)")
parser.add_argument("--data", type=str, default="data/nakhla1_parsed_fitted.csv", help="Path to data CSV")
parser.add_argument("--continuous", action="store_true", help="Use continuous domain size distribution")
parser.add_argument("--n_bins", type=int, default=20, help="Bins for continuous distribution")
parser.add_argument("--lognormal_dist", action="store_true", help="Use lognormal instead of normal")
parser.add_argument("--learn_sigma", action="store_true", help="Learn observation noise scale")
args = parser.parse_args()

num_domains = args.num_domains
num_chains = args.num_chains
num_warmup = args.num_warmup
num_steps = args.num_steps
tag = args.tag or f"{num_domains}d_{num_chains}c_{num_steps}s"
os.makedirs("results", exist_ok=True)

print(f"=== MDD Fit: {num_domains} domains, {num_chains} chains, "
      f"{num_warmup} warmup + {num_steps} samples, tag={tag} ===")

# ── Load data ───────────────────────────────────────────────────
entire_df = pd.read_csv(args.data)[
    ["Temp", "39Ar", "std_39Ar", "seconds_per_extraction_step"]]
total_39Ar = entire_df['39Ar'].sum()
entire_df['ΔF'] = entire_df['39Ar'] / total_39Ar
entire_df['F'] = entire_df['ΔF'].cumsum()
entire_df['T_K'] = entire_df['Temp'] + 273.15

def y_from_F(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6/np.sqrt(np.pi)
        disc = a*a - 12*F
        s = (a - np.sqrt(max(disc, 0))) / 6
        return s**2
    else:
        return -(1/np.pi**2) * np.log(max((1 - F) * np.pi**2 / 6, 1e-30))

entire_df['y'] = entire_df['F'].apply(y_from_F)
filtered_df = entire_df.dropna()
if args.min_temp > 0:
    filtered_df = filtered_df[filtered_df["Temp"] >= args.min_temp]
    print(f"Excluding steps below {args.min_temp}°C ({len(filtered_df)} steps remain)")

temps_K = filtered_df["T_K"].values
f_obs = filtered_df["ΔF"].values
sigma_obs = filtered_df["std_39Ar"].values / total_39Ar + 1e-12
dt = filtered_df["seconds_per_extraction_step"].values
n_steps_data = len(temps_K)
print(f"Data: {n_steps_data} extraction steps, T range: {temps_K.min()-273:.0f}–{temps_K.max()-273:.0f}°C")

# ── Priors ──────────────────────────────────────────────────────
Ea_prior_mode = 117.0
Ea_prior_sigma = args.ea_sigma
Ea_prior_mu = jnp.log(Ea_prior_mode) + Ea_prior_sigma**2
logD0_r2_prior_mean = 5.7
logD0_r2_prior_std = 2.0
dirichlet_alphas = jnp.array([1.0] * num_domains)
diffusion_repulsion = args.repulsion

# ── Model ───────────────────────────────────────────────────────
def stick_breaking(phi_raw):
    phi_raw = jnp.asarray(phi_raw)
    probs = jax.nn.sigmoid(phi_raw)
    one_minus_probs = 1.0 - probs
    remaining = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(one_minus_probs)])
    phi_pieces = probs * remaining[:-1]
    return jnp.concatenate([phi_pieces, remaining[-1:]])

def fractional_release(y):
    y = jnp.asarray(y)
    F_short = 6 * jnp.sqrt(y / jnp.pi) - 3 * y
    F_long  = 1 - (6 / jnp.pi**2) * jnp.exp(-jnp.pi**2 * y)
    return jnp.clip(jnp.where(y < 0.3, F_short, F_long), 0.0, 1.0)

def forward_model(params):
    """Compute predicted fractional release spectrum. Returns (f_pred, per_step_ll)."""
    R_gas = 8.314
    T = temps_K[None, :]
    obs_sigma = sigma_obs

    if args.learn_sigma:
        log_sigma_scale = params["log_sigma_scale"]
        obs_sigma = sigma_obs * jnp.exp(log_sigma_scale)

    if args.continuous:
        # Continuous domain size distribution
        mu_logD = params["mu_logD"]
        sigma_logD = jnp.exp(params["log_sigma_logD"])
        Ea_val = 117.0 if args.ea_fixed else params["Ea"][0]

        bins = jnp.linspace(mu_logD - 3*sigma_logD, mu_logD + 3*sigma_logD, args.n_bins)
        bin_width = bins[1] - bins[0]
        if args.lognormal_dist:
            weights = jnp.exp(-0.5*((bins - mu_logD)/sigma_logD)**2) * bin_width
        else:
            weights = jnp.exp(-0.5*((bins - mu_logD)/sigma_logD)**2) * bin_width
        weights = weights / (weights.sum() + 1e-30)

        D_r2 = jnp.exp(bins)[:, None]
        y_inc = D_r2 * jnp.exp(-Ea_val * 1e3 / (R_gas * T)) * dt
        y_cum = jnp.cumsum(y_inc, axis=1)
        Fcum = fractional_release(y_cum)
        dF = jnp.diff(Fcum, axis=1, prepend=jnp.zeros((args.n_bins, 1)))

        dF_mix = weights @ dF
        F_final_mix = weights @ Fcum[..., -1]
        f_pred = dF_mix / (F_final_mix + 1e-30)
    else:
        # Discrete domain model
        Ea = params["Ea"]
        logD0_r2 = params["logD0_r2"]
        phi_raw = params["phi_raw"]
        phi = stick_breaking(phi_raw)
        D_r2 = jnp.exp(logD0_r2)[:, None]

        if args.ea_fixed:
            Ea_broadcast = jnp.full((num_domains, 1), 117.0)
        elif args.shared_ea:
            Ea_broadcast = jnp.full((num_domains, 1), Ea[0])
        else:
            Ea_broadcast = Ea[:, None]

        y_inc = D_r2 * jnp.exp(-Ea_broadcast * 1e3 / (R_gas * T)) * dt
        y_cum = jnp.cumsum(y_inc, axis=1)
        Fcum = fractional_release(y_cum)
        dF = jnp.diff(Fcum, axis=1, prepend=jnp.zeros((num_domains, 1)))

        dF_mix = phi @ dF
        F_final_mix = phi @ Fcum[..., -1]
        f_pred = dF_mix / (F_final_mix + 1e-30)

    per_step_ll = -0.5 * ((f_obs - f_pred) / obs_sigma)**2 \
                  - jnp.log(obs_sigma) - 0.5 * jnp.log(2 * jnp.pi)
    return f_pred, per_step_ll


def log_prob(params):
    f_pred, per_step_ll = forward_model(params)
    ll = jnp.sum(per_step_ll)

    # Priors
    prior = 0.0
    if not args.ea_fixed:
        Ea = params["Ea"]
        prior += jnp.sum(-0.5 * (Ea - Ea_prior_mode)**2 / args.ea_sigma**2)
        prior += jnp.sum(jnp.where(Ea < 0, -1e6, 0))

    if args.continuous:
        mu_logD = params["mu_logD"]
        log_sigma = params["log_sigma_logD"]
        prior += -0.5 * (mu_logD - logD0_r2_prior_mean)**2 / logD0_r2_prior_std**2
        prior += -0.5 * log_sigma**2  # gentle prior on log-scale
    else:
        logD0_r2 = params["logD0_r2"]
        prior += jnp.sum(-0.5 * (logD0_r2 - logD0_r2_prior_mean)**2 / logD0_r2_prior_std**2)
        if num_domains > 1:
            phi = stick_breaking(params["phi_raw"])
            prior += jnp.sum((dirichlet_alphas - 1) * jnp.log(phi))
            pairwise_diff = logD0_r2[:, None] - logD0_r2[None, :]
            kernel = diffusion_repulsion * jnp.exp(-0.5 * pairwise_diff**2)
            prior += -jnp.sum(kernel * (1.0 - jnp.eye(num_domains)))

    if args.learn_sigma:
        prior += -0.5 * params["log_sigma_scale"]**2  # regularize noise scale

    return ll + prior

# ── Init + Warmup ───────────────────────────────────────────────
key = random.PRNGKey(args.seed)
chain_keys = random.split(key, num_chains)

def init_params(key):
    k1, k2, k3, k4 = random.split(key, 4)
    params = {}

    if args.continuous:
        params["mu_logD"] = logD0_r2_prior_mean + random.normal(k2) * 1.0
        params["log_sigma_logD"] = random.normal(k3) * 0.5
        if not args.ea_fixed:
            params["Ea"] = jnp.array([Ea_prior_mode + random.normal(k1) * 10.0])
        else:
            params["Ea"] = jnp.array([117.0])
    else:
        Ea = Ea_prior_mode + random.normal(k1, (num_domains,)) * 10.0
        Ea = jnp.clip(Ea, 50.0, 300.0)
        if args.ea_fixed:
            Ea = jnp.full((num_domains,), 117.0)
        params["Ea"] = Ea
        params["logD0_r2"] = random.normal(k2, (num_domains,)) * logD0_r2_prior_std + logD0_r2_prior_mean
        params["phi_raw"] = jnp.zeros((num_domains - 1,))

    if args.learn_sigma:
        params["log_sigma_scale"] = jnp.array(0.0)

    return params

print("Initializing parameters...")
init_params_batch = jax.vmap(init_params)(chain_keys)

def warmup_one(key, init_p):
    warm = blackjax.window_adaptation(
        blackjax.nuts, log_prob, initial_step_size=1e-3,
        is_mass_matrix_diagonal=False, target_acceptance_rate=0.95,
        max_num_doublings=10)
    (state, tuned), _ = warm.run(key, init_p, num_steps=num_warmup)
    return state, tuned["step_size"], tuned["inverse_mass_matrix"]

print(f"Warming up {num_chains} chains ({num_warmup} steps each)...")
t0 = time.time()
states, step_sizes, inv_mats = jax.vmap(warmup_one)(chain_keys, init_params_batch)
warmup_time = time.time() - t0
ss = jax.device_get(step_sizes)
print(f"  Warmup done in {warmup_time:.1f}s. Step sizes: [{ss.min():.2e}, {ss.max():.2e}]")
print(f"  Chains with step_size > 0: {(ss > 0).sum()}/{num_chains}")

# ── Sampling ────────────────────────────────────────────────────
def run_chain(state, keys, step_size, inv_mass):
    kernel = blackjax.nuts(log_prob, step_size, inv_mass, max_num_doublings=10)
    def one_step(carry, key):
        st = carry
        st, _ = kernel.step(key, st)
        return st, st.position
    final_state, positions = jax.lax.scan(one_step, state, keys)
    return final_state, positions

run_all = jax.vmap(run_chain, in_axes=(0, 0, 0, 0))

sample_keys = jax.vmap(lambda k: random.split(k, num_steps))(chain_keys)

print(f"Sampling {num_steps} steps across {num_chains} chains...")
t0 = time.time()
final_states, trace_positions = run_all(states, sample_keys, step_sizes, inv_mats)
sample_time = time.time() - t0
print(f"  Sampling done in {sample_time:.1f}s")

trace_positions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trace_positions)
warmup_pos = states.position

full_samples = {
    p: jnp.concatenate([warmup_pos[p][None, ...], trace_positions[p]], axis=0)
    for p in warmup_pos.keys()
}

# ── Post-process ────────────────────────────────────────────────
fs_np = jax.device_get(full_samples)

if not args.continuous and "logD0_r2" in fs_np:
    T_samp, C_samp, _ = fs_np["logD0_r2"].shape
    if num_domains > 1:
        fs_np["phi"] = jax.device_get(
            jax.vmap(jax.vmap(stick_breaking))(full_samples["phi_raw"]))
        final_logD = fs_np["logD0_r2"][-1]
        perm = np.argsort(-final_logD, axis=1)
        for key in fs_np:
            if key == "phi_raw": continue
            if fs_np[key].ndim == 3 and fs_np[key].shape[2] == num_domains:
                fs_np[key] = np.take_along_axis(fs_np[key], perm[None, :, :], axis=2)
else:
    C_samp = full_samples[list(full_samples.keys())[0]].shape[1]

# Compute log-prob trace and per-step ll for WAIC
def compute_ll_and_perstep(params_dict):
    _, per_step = forward_model(params_dict)
    total = log_prob(params_dict)
    return total, per_step

# Build params dicts from samples for vmapping
def build_params(sample_dict, t_idx, c_idx):
    return {k: sample_dict[k][t_idx, c_idx] for k in sample_dict}

# Compute ll for last time step (post-warmup samples only)
print("Computing WAIC...")
last_samples = {k: full_samples[k][-1] for k in full_samples}  # (chains, ...)
per_step_lls = []  # (chains, n_steps_data)
total_lls = []

for c in range(C_samp):
    p = {k: last_samples[k][c] for k in last_samples}
    tl, psl = compute_ll_and_perstep(p)
    total_lls.append(float(tl))
    per_step_lls.append(np.array(psl))

per_step_lls = np.array(per_step_lls)  # (chains, n_steps)
total_lls = np.array(total_lls)
ll = total_lls  # for compatibility with plots

# WAIC computation
from scipy.special import logsumexp
lppd = np.sum(logsumexp(per_step_lls, axis=0) - np.log(C_samp))
p_waic = np.sum(np.var(per_step_lls, axis=0))
waic = -2 * (lppd - p_waic)
print(f"WAIC = {waic:.1f}  (lppd={lppd:.1f}, p_waic={p_waic:.1f})")

# ── Save ────────────────────────────────────────────────────────
pickle.dump({"samples": fs_np, "ll": ll, "waic": waic, "lppd": lppd,
             "p_waic": p_waic, "args": vars(args)},
            open(f"results/fit_{tag}.pkl", "wb"))
print(f"Saved results/fit_{tag}.pkl")

# ── Convergence diagnostics ─────────────────────────────────────
def split_rhat(samples):
    T, C = samples.shape
    half = T // 2
    chains = np.concatenate([samples[:half].T, samples[half:half*2].T], axis=0)
    n, m = half, chains.shape[0]
    chain_means = chains.mean(axis=1)
    B = n / (m - 1) * np.sum((chain_means - chain_means.mean())**2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    return np.sqrt(((n-1)/n * W + B/n) / W) if W > 0 else np.nan

print("\n=== Convergence (split R-hat) ===")
all_converged = True
for key in ["Ea", "logD0_r2", "phi"]:
    for d in range(num_domains):
        rhat = split_rhat(fs_np[key][:, :, d])
        ok = rhat < 1.1
        if not ok: all_converged = False
        print(f"  {key}[{d}]: R-hat = {rhat:.4f} {'OK' if ok else 'WARNING'}")

# ── Diagnostic plots ────────────────────────────────────────────
print("\nGenerating plots...")

# 1. Trace plots
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
cmap = get_cmap("tab10")
alpha = max(5 / num_chains, 0.01)
for ax, (key, title) in zip(axes, [("Ea","Ea (kJ/mol)"), ("logD0_r2","ln(D₀/ρ²)"), ("phi","φ")]):
    for d in range(num_domains):
        for c in range(C_samp):
            ax.plot(fs_np[key][:, c, d], color=cmap(d), lw=0.3, alpha=alpha)
    ax.set_title(title)
axes[3].plot(-ll, color="gray", alpha=0.1, lw=0.3)
axes[3].set_yscale('log')
axes[3].set_title("−log prob")
fig.tight_layout()
fig.savefig(f"results/traces_{tag}.png", dpi=150)
plt.close(fig)

# 2. Posterior predictive
R_gas = 8.314
last = {k: full_samples[k][-1] for k in full_samples}
phi_last = jax.vmap(stick_breaking)(last["phi_raw"])
EaJ = last["Ea"] * 1e3
Dr2 = jnp.exp(last["logD0_r2"])

if args.shared_ea:
    y_inc = Dr2[..., None] * jnp.exp(-EaJ[:, :1, None] / (R_gas * temps_K)) * dt
else:
    y_inc = Dr2[..., None] * jnp.exp(-EaJ[..., None] / (R_gas * temps_K)) * dt

y_cum = jnp.cumsum(y_inc, axis=-1)
F_cum = fractional_release(y_cum)
dF = jnp.diff(F_cum, axis=-1, prepend=jnp.zeros_like(F_cum[..., :1]))
Ffin = (phi_last * F_cum[..., -1]).sum(axis=1)[:, None]
f_pred = (phi_last[..., None] * dF).sum(axis=1) / (Ffin + 1e-30)
f_pred_np = jax.device_get(f_pred)

fig, ax = plt.subplots(figsize=(8, 5))
final_cum = filtered_df["F"].values[-1]
scaled_f_obs = f_obs / final_cum
ax.fill_between(temps_K - 273.15, scaled_f_obs - 2*sigma_obs,
                scaled_f_obs + 2*sigma_obs, color='grey', alpha=0.3, label='Observed ±2σ')
ax.plot(temps_K - 273.15, scaled_f_obs, 'k-', lw=2, label='Observed')
for fp in f_pred_np:
    ax.plot(temps_K - 273.15, fp, color='tab:blue', alpha=max(10/num_chains, 0.01), lw=0.5)
ax.set_xlabel('Extraction temperature (°C)')
ax.set_ylabel('Fraction ³⁹Ar per step')
ax.set_title(f'Posterior Predictive ({tag})')
ax.legend()
fig.tight_layout()
fig.savefig(f"results/posterior_predictive_{tag}.png", dpi=150)
plt.close(fig)

# 3. Parameter summaries
print("\n=== Parameter Summary (final samples) ===")
for key in ["Ea", "logD0_r2", "phi"]:
    for d in range(num_domains):
        vals = fs_np[key][-1, :, d]
        print(f"  {key}[{d}]: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"median={np.median(vals):.2f}, 95%CI=[{np.percentile(vals,2.5):.2f}, {np.percentile(vals,97.5):.2f}]")

# 4. T-t constraint
def max_temp_for_excursion(Ea_kJ, logD0_r2_val, duration_Ma, max_frac_loss=0.01):
    duration_s = duration_Ma * 1e6 * 3.15e7
    D0_r2 = np.exp(logD0_r2_val)
    def frac_loss_at_T(T_C):
        T_K = T_C + 273.15
        D_r2 = D0_r2 * np.exp(-Ea_kJ * 1e3 / (R_gas * T_K))
        y = D_r2 * duration_s
        F = 6*np.sqrt(y/np.pi) - 3*y if y < 0.3 else 1-(6/np.pi**2)*np.exp(-np.pi**2*y)
        return np.clip(F, 0, 1) - max_frac_loss
    try:
        return brentq(frac_loss_at_T, -273, 1000, xtol=0.1)
    except ValueError:
        return np.nan

final_Ea = fs_np["Ea"][-1]
final_logD = fs_np["logD0_r2"][-1]
hrd_idx = np.argmin(final_logD, axis=1)
hrd_Ea = final_Ea[np.arange(num_chains), hrd_idx]
hrd_logD = final_logD[np.arange(num_chains), hrd_idx]

durations = [10, 100, 200, 500, 1300]
labels = ["10 My", "100 My", "200 My", "500 My", "Isothermal"]
results_Tt = {}

print("\n=== T-t Constraints (max T for ≤1% Ar loss from HRD) ===")
for dur, label in zip(durations, labels):
    temps_arr = np.array([max_temp_for_excursion(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
    valid = temps_arr[~np.isnan(temps_arr)]
    results_Tt[label] = valid
    if len(valid) > 0:
        med = np.median(valid)
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:12s}: {med:6.1f}°C  95%CI=[{lo:.1f}, {hi:.1f}]  crosses 0°C: {hi > 0}")

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([results_Tt[l] for l in labels if len(results_Tt[l]) > 0],
                positions=range(len([l for l in labels if len(results_Tt[l]) > 0])),
                widths=0.5, patch_artist=True, showfliers=False, whis=[2.5, 97.5])
for patch in bp['boxes']:
    patch.set_facecolor('#4a90d9')
    patch.set_alpha(0.5)
ax.axhline(0, color='blue', ls='--', lw=1.5, label='0°C (freezing)')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel('Maximum temperature (°C)')
ax.set_title(f'Mars Temperature Constraint — Bayesian 95% CI ({tag})')
ax.legend()
fig.tight_layout()
fig.savefig(f"results/Tt_constraint_{tag}.png", dpi=150)
plt.close(fig)

# 5. Residuals
median_pred = np.median(f_pred_np, axis=0)
residuals = (scaled_f_obs - median_pred) / sigma_obs
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(temps_K - 273.15, residuals, width=15, color='tab:blue', alpha=0.7)
ax.axhline(0, color='k', lw=0.5)
ax.axhline(2, color='r', ls='--', lw=0.5)
ax.axhline(-2, color='r', ls='--', lw=0.5)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Standardized residual')
ax.set_title(f'Residuals ({tag})')
fig.tight_layout()
fig.savefig(f"results/residuals_{tag}.png", dpi=150)
plt.close(fig)

chi2 = np.sum(residuals**2)
print(f"\nχ² = {chi2:.1f} ({n_steps_data} data points, {num_domains*2+1} params)")
print(f"χ²/dof = {chi2/(n_steps_data - num_domains*2 - 1):.2f}")
print(f"\nAll plots saved to results/")
print(f"Total time: warmup={warmup_time:.0f}s, sampling={sample_time:.0f}s")
