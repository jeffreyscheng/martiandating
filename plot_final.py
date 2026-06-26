"""Generate 3-panel summary plot for each model: Arrhenius + domain posteriors + T-t constraint."""
import os, sys, pickle
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/jefcheng/.cache/jax'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

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
dt = fitted['seconds_per_extraction_step'].values
invT = 1e4 / tK
y_before = yF(F_before)
y_obs = np.array([yF(f) for f in fitted_F_cum])
dy_obs = np.diff(y_obs, prepend=y_before)
D_obs = dy_obs / dt
D_obs[D_obs <= 0] = np.nan
lnD_obs = np.log(D_obs)
frac_fitted = fitted_F_cum[-1] - F_before

# S&W reference
yi_sw = np.exp(5.7) * np.exp(-117e3 / (R_gas * tK)) * dt
yc_sw = np.cumsum(yi_sw)
def fr_np(y):
    return max(6*np.sqrt(y/np.pi)-3*y, 0) if y < 0.3 else min(1-(6/np.pi**2)*np.exp(-np.pi**2*y), 1)
Fc_sw = np.clip(np.array([fr_np(y) for y in yc_sw]), 0, 1)
F_sw = F_before + Fc_sw / max(Fc_sw[-1], 1e-30) * frac_fitted
lnD_sw = np.log(np.maximum(np.diff(np.array([yF(f) for f in F_sw]), prepend=y_before) / dt, 1e-30))

def max_temp(Ea_kJ, logD_val, dur_Ma):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD_val)
    def f(T_C):
        T_K = T_C + 273.15
        y = D0r2 * np.exp(-Ea_kJ * 1e3 / (R_gas * T_K)) * dur_s
        F = fr_np(y)
        return np.clip(F, 0, 1) - 0.01
    try: return brentq(f, -273, 1000, xtol=0.1)
    except: return np.nan

def split_rhat(samples):
    T, C = samples.shape
    half = T // 2
    if half < 2: return np.nan
    chains = np.concatenate([samples[:half].T, samples[half:half*2].T], axis=0)
    n, m = half, chains.shape[0]
    chain_means = chains.mean(axis=1)
    B = n / (m - 1) * np.sum((chain_means - chain_means.mean()) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    return np.sqrt(((n - 1) / n * W + B / n) / W) if W > 0 else np.nan


for tag in ['PF_NUTS_f003', 'PF_3es_r1', 'PF_3es_r2', 'PF_3es_r3', 'FU_3es_r1', 'FU_3es_r2', 'AW4r', 'AW4', 'AW4ep', 'AW4es', 'AC4e']:
    pkl = f'results/fit_{tag}_combined.pkl'
    if not os.path.exists(pkl):
        print(f'{tag}: no combined results')
        continue

    d = pickle.load(open(pkl, 'rb'))
    fs = d['samples']
    nd = fs['logD0_r2'].shape[2]
    nc = fs['logD0_r2'].shape[1]
    nl = fs['logD0_r2'].shape[0]

    # R-hat
    rhats = {}
    for key in ['logD0_r2', 'phi']:
        if key in fs:
            for dim in range(fs[key].shape[2]):
                rhats[f'{key}[{dim}]'] = split_rhat(fs[key][:, :, dim])

    # Posterior predictive effective Arrhenius: parameter uncertainty + observation noise
    sigma_floor = d.get('args', {}).get('sigma_floor', 0.0)
    sigma_meas_arr = fitted['std_39Ar'].values / df['39Ar'].sum() + 1e-12
    sigma_total = np.sqrt(sigma_meas_arr**2 + sigma_floor**2)
    rng_pp = np.random.RandomState(42)

    all_lnD = []
    for c in range(min(nc, 500)):
        logD = fs['logD0_r2'][-1, c]
        phi = fs['phi'][-1, c] if 'phi' in fs else np.ones(nd) / nd
        Ea_use = fs['Ea_shared'][-1, c] if 'Ea_shared' in fs else Ea_fixed
        Dr2 = np.exp(logD)[:, None]
        yi = Dr2 * np.exp(-Ea_use * 1e3 / (R_gas * tK[None, :])) * dt
        yc = np.cumsum(yi, axis=1)
        Fc = np.array([[fr_np(y) for y in yc[d]] for d in range(nd)])
        dF_dom = np.diff(Fc, axis=1, prepend=np.zeros((nd, 1)))
        dF_mix = phi @ dF_dom
        f_pred = dF_mix / (dF_mix.sum() + 1e-30) * frac_fitted

        # Add observation noise for posterior predictive
        f_sim = f_pred + rng_pp.normal(0, sigma_total)
        f_sim = np.maximum(f_sim, 1e-12)

        # Compute effective Arrhenius from simulated observation
        Fs = F_before + np.cumsum(f_sim)
        ye = np.array([yF(f) for f in Fs])
        dye = np.diff(ye, prepend=y_before)
        De = dye / dt
        De[De <= 0] = np.nan
        v = np.log(De)
        if np.all(np.isfinite(v)) and np.all(v > -30):
            all_lnD.append(v)
    all_lnD = np.array(all_lnD)

    # T-t constraints
    hrd_logD = fs['logD0_r2'][-1, :, 0]  # ordered, [0] is smallest = HRD
    hrd_Ea = fs['Ea_shared'][-1] if 'Ea_shared' in fs else np.full(nc, Ea_fixed)
    durs = [10, 100, 200, 500, 1300]
    labs = ['10 My', '100 My', '200 My', '500 My', 'Isothermal']
    tt = {}
    for dur, lab in zip(durs, labs):
        t = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
        tt[lab] = t[~np.isnan(t)]

    # === 3-PANEL FIGURE ===
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Arrhenius
    if len(all_lnD) > 0:
        med = np.median(all_lnD, 0)
        lo = np.percentile(all_lnD, 2.5, 0)
        hi = np.percentile(all_lnD, 97.5, 0)
        ax1.fill_between(invT, lo, hi, color='#2196F3', alpha=0.2, label='95% CI')
        ax1.plot(invT, med, color='#2196F3', lw=2.5, label='Bayesian median')
    ax1.plot(invT, lnD_sw, 'r--', lw=2, alpha=0.7, label='S&W point est.')
    ax1.scatter(invT, lnD_obs, color='k', s=35, zorder=10, label='Observed')
    all_y = np.concatenate([lnD_obs[np.isfinite(lnD_obs)], lnD_sw])
    ax1.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
    ax1.set_xlim(9.5, 16.5)
    ax1.set_xlabel('10⁴/T (K⁻¹)', fontsize=12)
    ax1.set_ylabel('Effective diffusion rate ln(D/ρ²)', fontsize=12)
    ax1t = ax1.secondary_xaxis('top', functions=(lambda x: 1e4/x-273.15, lambda x: 1e4/(x+273.15)))
    ax1t.set_xlabel('Temperature (°C)')
    ax1t.set_xticks([700, 600, 500, 400, 350])
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(True, alpha=0.1)
    ax1.set_title('A. Effective Arrhenius', fontsize=12, fontweight='bold', loc='left')

    # Panel B: Domain size posteriors
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for dim in range(nd):
        vals = fs['logD0_r2'][-1, :, dim]
        ax2.hist(vals, bins=50, alpha=0.5, color=colors[dim],
                 label=f'Domain {dim+1}: median={np.median(vals):.1f}', density=True)
        ax2.axvline(np.median(vals), color=colors[dim], ls='--', lw=1.5)
    ax2.axvline(5.7, color='red', ls=':', lw=2, label='S&W HRD (5.7)')
    ax2.set_xlabel('Diffusion rate ln(D₀/ρ²)', fontsize=12)
    ax2.set_ylabel('Posterior density', fontsize=12)
    ax2.legend(fontsize=8)
    # Add noise scale info if available
    if 'log_sigma_scale' in fs:
        noise_med = np.exp(np.median(fs['log_sigma_scale'][-1]))
        ax2.text(0.98, 0.98, f'Learned noise: {noise_med:.1f}× input σ',
                 transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.set_title('B. Domain size posteriors', fontsize=12, fontweight='bold', loc='left')

    # Phi posteriors as distributions in inset
    if 'phi' in fs and nd > 1:
        ax2_inset = ax2.inset_axes([0.52, 0.45, 0.45, 0.5])
        phi_samples = fs['phi'][-1]  # (chains, nd)
        vp = ax2_inset.violinplot([phi_samples[:, d] for d in range(nd)],
                                   positions=range(nd), showmedians=True, showextrema=False)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i]); body.set_alpha(0.6)
        vp['cmedians'].set_color('black')
        ax2_inset.set_xticks(range(nd))
        ax2_inset.set_xticklabels([f'D{i+1}' for i in range(nd)], fontsize=8)
        ax2_inset.set_ylabel('Volume fraction φ', fontsize=8)
        ax2_inset.set_ylim(-0.05, 1.0)
        ax2_inset.set_title('Volume fractions', fontsize=8)

    # Panel C: T-t constraint
    bp = ax3.boxplot([tt[l] for l in labs], positions=range(len(labs)),
                     widths=0.5, patch_artist=True, showfliers=False, whis=[2.5, 97.5])
    for p in bp['boxes']:
        p.set_facecolor('#2196F3')
        p.set_alpha(0.4)
    ax3.axhline(0, color='blue', ls='--', lw=1.5, alpha=0.6, label='0°C')
    sw_t = [max_temp(Ea_fixed, 5.7, dur) for dur in durs]
    ax3.scatter(range(len(labs)), sw_t, color='red', s=80, marker='D', zorder=10, label='S&W')
    ax3.set_xticks(range(len(labs)))
    ax3.set_xticklabels(labs, rotation=15, fontsize=9)
    ax3.set_ylabel('Max temperature (°C)', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.1)
    ax3.set_title('C. Mars temperature constraint', fontsize=12, fontweight='bold', loc='left')

    # Suptitle with convergence info
    max_rhat = max(rhats.values()) if rhats else np.nan
    desc = f'{tag}: {nd}-domain, Ea=117 fixed'
    if tag in ('N2', 'N3'):
        desc += ' + learned noise'
    if tag == 'N3':
        desc += ' (exp penalty + Dir(2))'
    if tag in ('D4v2', 'N4'):
        desc += ' [TRUE σ: 0.2-0.5%]'
    if tag == 'N4':
        desc += ' + learned noise (exp+Dir2)'
    fig.suptitle(f'{desc} | {nc} chains × 100K steps | max R-hat: {max_rhat:.3f}',
                 fontsize=12, y=1.02)

    # Print key numbers
    iso = tt['Isothermal']
    iso_str = f'Isothermal: {np.median(iso):.1f}°C [{np.percentile(iso,2.5):.1f}, {np.percentile(iso,97.5):.1f}]'
    crosses = np.percentile(iso, 97.5) > 0
    print(f'{tag}: {iso_str}, crosses 0°C: {crosses}, max R-hat: {max_rhat:.3f}')

    fig.tight_layout()
    fig.savefig(f'results/{tag}_final.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved results/{tag}_final.png')
