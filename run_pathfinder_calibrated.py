"""Calibrated multi-start pathfinder: sensitivity analysis across noise floors and Ea modes."""
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
import blackjax.vi.pathfinder as pathfinder
from scipy.optimize import brentq
from scipy.stats import multivariate_normal

warnings.filterwarnings('ignore', category=RuntimeWarning)

print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")

parser = argparse.ArgumentParser()
parser.add_argument("--num_domains", type=int, default=3)
parser.add_argument("--num_starts", type=int, default=500)
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

nd = args.num_domains
R_gas = 8.314

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

filt = df.dropna(subset=['seconds_per_extraction_step'])
filt = filt[filt['Temp'] >= 350]
temps_K = jnp.array(filt['T_K'].values)
f_obs = jnp.array(filt['dF'].values)
sigma_meas = jnp.array(filt['std_39Ar'].values / total_39Ar + 1e-12)
dt = jnp.array(filt['seconds_per_extraction_step'].values)
n_data = len(temps_K)
temps_C = filt['Temp'].values

all_filt = df.dropna(subset=['seconds_per_extraction_step'])
F_before = all_filt[all_filt['Temp'] < 350]['dF'].sum()
fitted_F_cum = F_before + np.cumsum(filt['dF'].values)
frac_fitted = fitted_F_cum[-1] - F_before
y_before = y_from_F(F_before)
y_obs_arr = np.array([y_from_F(f) for f in fitted_F_cum])
dy_obs = np.diff(y_obs_arr, prepend=y_before)
D_obs = dy_obs / filt['seconds_per_extraction_step'].values
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)

# ── Model components ──
def stick_breaking(phi_raw):
    probs = jax.nn.sigmoid(phi_raw)
    remaining = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(1.0 - probs)])
    return jnp.concatenate([probs * remaining[:-1], remaining[-1:]])

def fractional_release(y):
    return jnp.clip(jnp.where(y < 0.3, 6*jnp.sqrt(y/jnp.pi)-3*y,
                               1-(6/jnp.pi**2)*jnp.exp(-jnp.pi**2*y)), 0, 1)

def np_stick_breaking(pr):
    probs = 1 / (1 + np.exp(-pr))
    remaining = np.concatenate([[1.0], np.cumprod(1 - probs)])
    return np.concatenate([probs * remaining[:-1], remaining[-1:]])

def fr_np(y):
    return max(6*np.sqrt(y/np.pi)-3*y, 0) if y < 0.3 else min(1-(6/np.pi**2)*np.exp(-np.pi**2*y), 1)

def max_temp(Ea_kJ, logD_val, dur_Ma):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD_val)
    def f(T_C):
        y = D0r2 * np.exp(-Ea_kJ*1e3/(R_gas*(T_C+273.15))) * dur_s
        return np.clip(fr_np(y), 0, 1) - 0.01
    try: return brentq(f, -273, 1000, xtol=0.1)
    except: return np.nan

# ── Run one configuration ──
def run_config(sigma_floor, ea_mode, Ea_bound_lo=100, Ea_bound_hi=135):
    sigma_obs = jnp.sqrt(sigma_meas**2 + sigma_floor**2)

    def log_prob(params):
        logD = jnp.cumsum(jax.nn.softplus(params["logD_raw"])) + params["logD_raw"][0]
        phi = stick_breaking(params["phi_raw"])
        D_r2 = jnp.exp(logD)[:, None]
        Ea_val = params["Ea_shared"] if ea_mode == "shared" else 117.0
        y_inc = D_r2 * jnp.exp(-Ea_val * 1e3 / (R_gas * temps_K[None, :])) * dt
        Fcum = fractional_release(jnp.cumsum(y_inc, axis=1))
        dF = jnp.diff(Fcum, axis=1, prepend=jnp.zeros((nd, 1)))
        dF_mix = phi @ dF
        f_pred = dF_mix / (phi @ Fcum[..., -1] + 1e-30)
        ll = jnp.sum(-0.5 * ((f_obs - f_pred) / sigma_obs)**2 - jnp.log(sigma_obs))
        prior = jnp.sum(-0.5 * (logD - 5.7)**2 / 2.0**2)
        if ea_mode == "shared":
            prior += -0.5 * (Ea_val - 117.0)**2 / 10.0**2
            prior += -1e6 * jnp.maximum(0, Ea_val - Ea_bound_hi)**2
            prior += -1e6 * jnp.maximum(0, Ea_bound_lo - Ea_val)**2
        return ll + prior

    lp_jit = jax.jit(log_prob)

    # Multi-start pathfinder
    modes = []
    key = random.PRNGKey(args.seed)
    for i in range(args.num_starts):
        k1, k2, k3, key = random.split(key, 4)
        init = {
            "logD_raw": jnp.array([2.0, 0.5, 0.3])[:nd] + random.normal(k1, (nd,)) * 1.5,
            "phi_raw": random.normal(k2, (nd - 1,)) * 1.0,
        }
        if ea_mode == "shared":
            init["Ea_shared"] = 117.0 + random.normal(k3) * 5.0
        try:
            state, info = pathfinder.approximate(random.PRNGKey(i), log_prob, init, num_samples=50, maxiter=100)
            pos = state.position
            lp = float(lp_jit(pos))
            flat, unravel = ravel_pytree(pos)
            lr = np.array(pos['logD_raw'])
            logD = np.cumsum(np.log(1+np.exp(lr))) + lr[0]
            phi = np_stick_breaking(np.array(pos['phi_raw']))
            Ea = float(pos['Ea_shared']) if ea_mode == "shared" else 117.0
            alpha = np.array(state.alpha)
            beta = np.array(state.beta)
            gamma = np.array(state.gamma)
            inv_H = np.diag(alpha) + beta @ gamma @ beta.T
            eigvals = np.linalg.eigvalsh(inv_H)
            if not np.all(eigvals > 0):
                inv_H += np.eye(len(alpha)) * (abs(eigvals.min()) + 1e-8)
            modes.append({'flat': np.array(flat), 'logD': logD, 'phi': phi, 'Ea': Ea,
                         'log_prob': lp, 'inv_hessian': inv_H, 'unravel': unravel})
        except:
            pass

    if not modes:
        return None

    # Cluster
    flats = np.array([m['flat'] for m in modes])
    lps = np.array([m['log_prob'] for m in modes])
    clusters = []
    assigned = np.full(len(modes), -1)
    for i in range(len(modes)):
        if assigned[i] >= 0: continue
        members = [i]
        for j in range(i+1, len(modes)):
            if assigned[j] >= 0: continue
            if np.linalg.norm(flats[i] - flats[j]) < 1.0:
                members.append(j); assigned[j] = len(clusters)
        assigned[i] = len(clusters)
        best = max(members, key=lambda k: lps[k])
        clusters.append({'rep': modes[best], 'size': len(members)})
    clusters.sort(key=lambda c: -c['rep']['log_prob'])

    # Mixture
    components = []
    for c in clusters:
        m = c['rep']
        sign, logdet = np.linalg.slogdet(m['inv_hessian'])
        if sign <= 0: continue
        log_w = m['log_prob'] + 0.5 * logdet + np.log(c['size'])
        components.append({'mean': m['flat'], 'cov': m['inv_hessian'], 'log_weight': log_w, 'mode': m, 'size': c['size']})
    if not components:
        return None
    log_ws = np.array([c['log_weight'] for c in components])
    log_ws -= log_ws.max()
    weights = np.exp(log_ws); weights /= weights.sum()

    # Sample
    rng = np.random.RandomState(args.seed)
    n_per = rng.multinomial(args.num_samples, weights)
    all_flat = []
    for c, n in zip(components, n_per):
        if n == 0: continue
        try: s = rng.multivariate_normal(c['mean'], c['cov'], size=n)
        except: s = rng.multivariate_normal(c['mean'], c['cov']+np.eye(len(c['mean']))*1e-8, size=n)
        all_flat.append(s)
    flat_samples = np.concatenate(all_flat)

    # Importance weights
    unravel = modes[0]['unravel']
    log_target = np.array([float(lp_jit(unravel(jnp.array(s)))) for s in flat_samples])
    log_q = np.zeros(len(flat_samples))
    for si in range(len(flat_samples)):
        lcs = []
        for c, w in zip(components, weights):
            if w < 1e-30: continue
            try: lcs.append(np.log(w) + multivariate_normal.logpdf(flat_samples[si], c['mean'], c['cov']))
            except: pass
        if lcs:
            mx = max(lcs); log_q[si] = mx + np.log(sum(np.exp(lc-mx) for lc in lcs))
    log_iw = log_target - log_q; log_iw -= log_iw.max()
    iw = np.exp(log_iw); iw /= iw.sum()
    ess = 1.0 / np.sum(iw**2)

    # Physical params
    all_logD, all_phi, all_Ea = [], [], []
    for s in flat_samples:
        p = unravel(jnp.array(s))
        lr = np.array(p['logD_raw']); logD = np.cumsum(np.log(1+np.exp(lr))) + lr[0]
        phi = np_stick_breaking(np.array(p['phi_raw']))
        Ea = float(p['Ea_shared']) if ea_mode == "shared" else 117.0
        all_logD.append(logD); all_phi.append(phi); all_Ea.append(Ea)
    all_logD = np.array(all_logD); all_phi = np.array(all_phi); all_Ea = np.array(all_Ea)

    # Arrhenius
    tK = filt['T_K'].values; dt_arr = filt['seconds_per_extraction_step'].values
    all_lnD_eff = []
    for i in range(len(flat_samples)):
        Dr2 = np.exp(all_logD[i])[:, None]
        yi = Dr2 * np.exp(-all_Ea[i]*1e3/(R_gas*tK[None,:]))*dt_arr
        yc = np.cumsum(yi, axis=1)
        Fc = np.array([[fr_np(y) for y in yc[d]] for d in range(nd)])
        dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((nd,1)))
        dF_mix = all_phi[i] @ dF_dom; Fm = np.cumsum(dF_mix)
        if Fm[-1] < 1e-30: continue
        Fs = F_before + Fm/Fm[-1]*frac_fitted
        ye = np.array([y_from_F(f) for f in Fs]); dye = np.diff(ye, prepend=y_before)
        De = dye/dt_arr; De[De<=0] = np.nan; v = np.log(De)
        if np.all(np.isfinite(v)) and np.all(v > -30): all_lnD_eff.append(v)
    all_lnD_eff = np.array(all_lnD_eff) if all_lnD_eff else np.zeros((0, n_data))

    # Coverage
    n_covered = 0
    if len(all_lnD_eff) > 10:
        lo = np.percentile(all_lnD_eff, 2.5, axis=0); hi = np.percentile(all_lnD_eff, 97.5, axis=0)
        covered = [(lnD_obs[i] >= lo[i] and lnD_obs[i] <= hi[i]) if np.isfinite(lnD_obs[i]) else True for i in range(n_data)]
        n_covered = sum(covered)

    # T-t
    hrd_logD = all_logD[:, 0]
    iso_temps = np.array([max_temp(ea, ld, 1300) for ea, ld in zip(all_Ea, hrd_logD)])
    iso_valid = iso_temps[~np.isnan(iso_temps)]

    best_mode = clusters[0]['rep']
    return {
        'sigma_floor': sigma_floor, 'ea_mode': ea_mode,
        'n_modes': len(clusters), 'ess': ess,
        'map_logD': best_mode['logD'], 'map_phi': best_mode['phi'], 'map_Ea': best_mode['Ea'],
        'map_logprob': best_mode['log_prob'],
        'arrhenius_coverage': n_covered,
        'iso_median': np.median(iso_valid) if len(iso_valid) > 0 else np.nan,
        'iso_lo': np.percentile(iso_valid, 2.5) if len(iso_valid) > 0 else np.nan,
        'iso_hi': np.percentile(iso_valid, 97.5) if len(iso_valid) > 0 else np.nan,
        'logD_median': np.median(all_logD, axis=0),
        'logD_std': np.std(all_logD, axis=0),
        'phi_median': np.median(all_phi, axis=0),
        'Ea_median': np.median(all_Ea),
        'Ea_std': np.std(all_Ea),
        'all_lnD_eff': all_lnD_eff,
        'all_logD': all_logD, 'all_phi': all_phi, 'all_Ea': all_Ea,
    }

# ── Sweep ──
configs = [
    (0.0,    "fixed"),
    (0.0005, "fixed"),
    (0.001,  "fixed"),
    (0.002,  "fixed"),
    (0.005,  "fixed"),
    (0.0,    "shared"),
    (0.0005, "shared"),
    (0.001,  "shared"),
    (0.002,  "shared"),
]

results = []
t0 = time.time()
for sigma_floor, ea_mode in configs:
    label = f"floor={sigma_floor:.4f}, Ea={ea_mode}"
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    r = run_config(sigma_floor, ea_mode)
    if r is None:
        print("  FAILED")
        continue
    results.append(r)
    print(f"  MAP: logD={r['map_logD'].round(2)}, phi={r['map_phi'].round(3)}, Ea={r['map_Ea']:.1f}")
    print(f"  Modes: {r['n_modes']}, ESS: {r['ess']:.0f}")
    print(f"  Arrhenius: {r['arrhenius_coverage']}/{n_data}")
    print(f"  Isothermal: {r['iso_median']:.1f}°C [{r['iso_lo']:.1f}, {r['iso_hi']:.1f}]")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

# ── Summary table ──
print(f"\n{'='*80}")
print("SENSITIVITY TABLE")
print(f"{'='*80}")
print(f"{'Floor':>8s} {'Ea':>8s} | {'MAP logD':>20s} {'MAP Ea':>7s} | {'Arr':>5s} {'ESS':>6s} | {'T_iso med':>9s} {'95% CI':>15s} {'>0°C?':>6s}")
print("-"*90)
for r in results:
    logD_str = np.array2string(r['map_logD'], precision=1, separator=',')
    ci_str = f"[{r['iso_lo']:.1f}, {r['iso_hi']:.1f}]"
    crosses = "YES" if r['iso_hi'] > 0 else "no"
    print(f"{r['sigma_floor']:>8.4f} {r['ea_mode']:>8s} | {logD_str:>20s} {r['map_Ea']:>7.1f} | "
          f"{r['arrhenius_coverage']:>3d}/18 {r['ess']:>6.0f} | {r['iso_median']:>9.1f} {ci_str:>15s} {crosses:>6s}")

# ── Save ──
pickle.dump(results, open('results/sensitivity_sweep.pkl', 'wb'))

# ── Generate comparison plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
invT = 1e4 / filt['T_K'].values

colors_fixed = ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9', '#BBDEFB']
colors_shared = ['#C62828', '#E53935', '#EF5350', '#EF9A9A']

for ri, r in enumerate(results):
    if len(r['all_lnD_eff']) < 10: continue
    med = np.median(r['all_lnD_eff'], axis=0)
    lo = np.percentile(r['all_lnD_eff'], 2.5, axis=0)
    hi = np.percentile(r['all_lnD_eff'], 97.5, axis=0)
    c = colors_fixed[ri] if r['ea_mode'] == 'fixed' else colors_shared[ri - 5]
    label = f"floor={r['sigma_floor']:.4f}, Ea={r['ea_mode']}"
    axes[0].fill_between(invT, lo, hi, alpha=0.15, color=c)
    axes[0].plot(invT, med, lw=1.5, color=c, label=label, alpha=0.8)

axes[0].scatter(invT, lnD_obs, color='k', s=30, zorder=10, label='Observed')
axes[0].set_xlabel('10⁴/T (K⁻¹)'); axes[0].set_ylabel('ln(D/ρ²)')
axes[0].legend(fontsize=6, loc='lower left'); axes[0].grid(True, alpha=0.1)
axes[0].set_title('A. Arrhenius: sensitivity to noise floor', fontsize=11, fontweight='bold', loc='left')

# Panel B: T-t constraint by config
labels_plot = [f"f={r['sigma_floor']:.4f}\nEa={r['ea_mode']}" for r in results]
iso_data = []
for r in results:
    iso = np.array([max_temp(ea, ld, 1300) for ea, ld in zip(r['all_Ea'], r['all_logD'][:, 0])])
    iso_data.append(iso[~np.isnan(iso)])
bp = axes[1].boxplot(iso_data, positions=range(len(results)), widths=0.6,
                     patch_artist=True, showfliers=False, whis=[2.5, 97.5])
for i, p in enumerate(bp['boxes']):
    c = colors_fixed[i] if results[i]['ea_mode'] == 'fixed' else colors_shared[i-5]
    p.set_facecolor(c); p.set_alpha(0.5)
axes[1].axhline(0, color='blue', ls='--', lw=1.5, alpha=0.6)
axes[1].set_xticks(range(len(results)))
axes[1].set_xticklabels(labels_plot, fontsize=6, rotation=45, ha='right')
axes[1].set_ylabel('Isothermal max T (°C)')
axes[1].set_title('B. Mars temperature constraint', fontsize=11, fontweight='bold', loc='left')

# Panel C: summary table as text
axes[2].axis('off')
table_text = "Floor    Ea      Arr   ESS    T_iso [95% CI]\n" + "-"*50 + "\n"
for r in results:
    crosses = "*" if r['iso_hi'] > 0 else ""
    table_text += f"{r['sigma_floor']:.4f}  {r['ea_mode']:>6s}  {r['arrhenius_coverage']:>2d}/18  {r['ess']:>5.0f}  {r['iso_median']:>5.1f} [{r['iso_lo']:.1f},{r['iso_hi']:.1f}]{crosses}\n"
table_text += "\n* = 95% CI crosses 0°C"
axes[2].text(0.05, 0.95, table_text, transform=axes[2].transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[2].set_title('C. Summary', fontsize=11, fontweight='bold', loc='left')

fig.suptitle('Bayesian MDD Sensitivity Analysis: Noise Floor × Ea Mode', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('results/sensitivity_analysis.png', dpi=200, bbox_inches='tight')
print(f"\nSaved results/sensitivity_analysis.png")
print(f"Total time: {time.time()-t0:.0f}s")
