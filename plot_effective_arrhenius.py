"""2x3 subplot: top = effective Arrhenius, bottom = domain parameters."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import jax
import jax.numpy as jnp

R_gas = 8.314

# ── Load data ───────────────────────────────────────────────────
entire_df = pd.read_csv('data/nakhla1_parsed_fitted.csv')[
    ["Temp", "39Ar", "std_39Ar", "seconds_per_extraction_step"]]
total_39Ar = entire_df['39Ar'].sum()
entire_df['ΔF'] = entire_df['39Ar'] / total_39Ar
entire_df['F'] = entire_df['ΔF'].cumsum()
entire_df['T_K'] = entire_df['Temp'] + 273.15

def y_from_F_np(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6 / np.sqrt(np.pi)
        return ((a - np.sqrt(max(a * a - 12 * F, 0))) / 6) ** 2
    else:
        return -(1 / np.pi ** 2) * np.log(max((1 - F) * np.pi ** 2 / 6, 1e-30))

entire_df['y'] = entire_df['F'].apply(y_from_F_np)
filtered_df = entire_df.dropna()
temps_K = filtered_df['T_K'].values
dt_obs = filtered_df['seconds_per_extraction_step'].values
inv_T = 1e4 / temps_K

y_obs = filtered_df['y'].values
dy_obs = np.diff(y_obs, prepend=0)
D_eff_obs = dy_obs / dt_obs
D_eff_obs[D_eff_obs <= 0] = np.nan
ln_D_obs = np.log(D_eff_obs)

# ── Helpers ─────────────────────────────────────────────────────
def stick_breaking(phi_raw):
    phi_raw = jnp.asarray(phi_raw)
    probs = jax.nn.sigmoid(phi_raw)
    remaining = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(1.0 - probs)])
    return jnp.concatenate([probs * remaining[:-1], remaining[-1:]])

def fractional_release(y):
    y = jnp.asarray(y)
    F_short = 6 * jnp.sqrt(y / jnp.pi) - 3 * y
    F_long = 1 - (6 / jnp.pi ** 2) * jnp.exp(-jnp.pi ** 2 * y)
    return jnp.clip(jnp.where(y < 0.3, F_short, F_long), 0.0, 1.0)

def y_from_F_jax(F):
    a = 6 / jnp.sqrt(jnp.pi)
    s = (a - jnp.sqrt(jnp.maximum(a * a - 12 * F, 0.0))) / 6
    y_short = s ** 2
    y_long = -(1 / jnp.pi ** 2) * jnp.log(jnp.maximum((1 - F) * jnp.pi ** 2 / 6, 1e-30))
    return jnp.where(F < 0.85, y_short, y_long)

def compute_effective_lnD(Ea_vec, logD0_vec, phi_raw, shared_ea=False):
    phi = stick_breaking(phi_raw)
    nd = len(Ea_vec)
    D_r2 = jnp.exp(logD0_vec)[:, None]
    T = temps_K[None, :]
    Ea_b = jnp.full((nd, 1), Ea_vec[0]) if shared_ea else Ea_vec[:, None]
    y_inc = D_r2 * jnp.exp(-Ea_b * 1e3 / (R_gas * T)) * dt_obs
    y_cum = jnp.cumsum(y_inc, axis=1)
    F_cum = fractional_release(y_cum)
    F_cum_mix = (phi[:, None] * F_cum).sum(axis=0)
    y_eff = y_from_F_jax(F_cum_mix)
    dy_eff = jnp.diff(y_eff, prepend=0.0)
    D_eff = dy_eff / dt_obs
    return jnp.log(jnp.maximum(D_eff, 1e-30))

# ── S&W point estimate ──────────────────────────────────────────
sw_Ea = jnp.array([117.0, 117.0])
sw_logD = jnp.array([9.0, 5.7])
sw_phi_raw = jnp.array([jnp.log(0.03 / 0.97)])
ln_D_sw = np.array(compute_effective_lnD(sw_Ea, sw_logD, sw_phi_raw))

# ── Models ──────────────────────────────────────────────────────
models = {
    "2-domain\n(per-domain Eₐ)": ("results/fit_2d_med.pkl", False),
    "2-domain\n(shared Eₐ)": ("results/fit_2d_shared_ea.pkl", True),
    "3-domain\n(per-domain Eₐ)": ("results/fit_3d_med.pkl", False),
}

domain_colors = ['#2196F3', '#FF9800', '#4CAF50']

fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                         gridspec_kw={'height_ratios': [2, 1.2]})

for col, (name, (path, shared_ea)) in enumerate(models.items()):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    data = pickle.load(open(path, "rb"))
    fs = data["samples"]
    nc = fs["Ea"].shape[1]
    nd = fs["Ea"].shape[2]

    # ── Top: Effective Arrhenius ────────────────────────────────
    all_ln_D = []
    for c in range(nc):
        Ea_c = jnp.array(fs["Ea"][-1, c])
        logD_c = jnp.array(fs["logD0_r2"][-1, c])
        phi_raw_c = jnp.array(fs["phi_raw"][-1, c])
        ln_D_c = np.array(compute_effective_lnD(Ea_c, logD_c, phi_raw_c, shared_ea))
        if np.all(np.isfinite(ln_D_c)) and np.all(ln_D_c > -30):
            all_ln_D.append(ln_D_c)
    all_ln_D = np.array(all_ln_D)

    if len(all_ln_D) > 0:
        med = np.median(all_ln_D, axis=0)
        lo = np.percentile(all_ln_D, 2.5, axis=0)
        hi = np.percentile(all_ln_D, 97.5, axis=0)
        ax_top.fill_between(inv_T, lo, hi, color='#2196F3', alpha=0.2, label='Bayesian 95% CI')
        ax_top.plot(inv_T, med, color='#2196F3', lw=2, label='Bayesian median')

    valid_sw = np.isfinite(ln_D_sw) & (ln_D_sw > -30)
    ax_top.plot(inv_T[valid_sw], ln_D_sw[valid_sw], 'r-', lw=2,
                label='S&W (3% LRD, 97% HRD)')
    ax_top.scatter(inv_T, ln_D_obs, color='k', s=25, zorder=10, label='Observed')

    ax_top.set_title(name, fontsize=12, fontweight='bold')
    ax_top.set_xlim(9, 22)
    ax_top.set_ylim(-24, -6)
    ax_top_sec = ax_top.secondary_xaxis('top',
        functions=(lambda x: 1e4 / x - 273.15, lambda x: 1e4 / (x + 273.15)))
    ax_top_sec.set_xticks([700, 500, 400, 300, 250, 200])
    if col == 1:
        ax_top_sec.set_xlabel('Temperature (°C)', fontsize=11)
    if col == 0:
        ax_top.set_ylabel('ln(D/ρ²)  effective', fontsize=11)
        ax_top.legend(fontsize=8, loc='lower left')

    # ── Bottom: Domain parameters ───────────────────────────────
    # Show individual domain Arrhenius lines + volume fraction bars

    # Posterior medians for each domain
    Ea_med = np.median(fs["Ea"][-1], axis=0)
    logD_med = np.median(fs["logD0_r2"][-1], axis=0)
    phi_med = np.median(fs["phi"][-1], axis=0)
    Ea_lo = np.percentile(fs["Ea"][-1], 16, axis=0)
    Ea_hi = np.percentile(fs["Ea"][-1], 84, axis=0)
    logD_lo = np.percentile(fs["logD0_r2"][-1], 16, axis=0)
    logD_hi = np.percentile(fs["logD0_r2"][-1], 84, axis=0)
    phi_lo = np.percentile(fs["phi"][-1], 16, axis=0)
    phi_hi = np.percentile(fs["phi"][-1], 84, axis=0)

    # Left half of bottom panel: domain Arrhenius lines
    T_line = np.linspace(450, 1100, 100)
    inv_T_line = 1e4 / T_line
    for d in range(nd):
        ln_D_line = logD_med[d] - Ea_med[d] * 1e3 / (R_gas * T_line)
        ax_bot.plot(inv_T_line, ln_D_line, color=domain_colors[d], lw=2.5)

        # Label with parameters
        x_pos = 11 + d * 2.5
        y_pos = logD_med[d] - Ea_med[d] * 1e3 / (R_gas * (1e4 / x_pos))
        label = (f"D{d+1}: φ={phi_med[d]:.0%}\n"
                 f"Eₐ={Ea_med[d]:.0f} kJ/mol\n"
                 f"ln(D₀/ρ²)={logD_med[d]:.1f}")
        ax_bot.annotate(label, xy=(x_pos, y_pos), fontsize=7,
                       color=domain_colors[d], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor=domain_colors[d], alpha=0.9))

    # S&W reference lines
    for d, (ea, ld, lbl) in enumerate([(117, 9.0, 'S&W LRD'), (117, 5.7, 'S&W HRD')]):
        ln_ref = ld - ea * 1e3 / (R_gas * T_line)
        ax_bot.plot(inv_T_line, ln_ref, 'r--', lw=1, alpha=0.5)

    ax_bot.set_xlim(9, 22)
    ax_bot.set_ylim(-24, -6)
    ax_bot.set_xlabel('10⁴/T  (K⁻¹)', fontsize=10)
    if col == 0:
        ax_bot.set_ylabel('ln(D/ρ²)  per domain', fontsize=11)
    ax_bot.set_title('Individual domain Arrhenius lines', fontsize=9, style='italic')

    # Add S&W comparison text
    sw_text = "S&W: LRD 3%, HRD 97%\nEₐ=117, ln(D₀/ρ²)=5.7"
    ax_bot.text(0.98, 0.02, sw_text, transform=ax_bot.transAxes,
               fontsize=7, ha='right', va='bottom', color='red', alpha=0.7,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.3))

fig.suptitle('MDD Model Comparison: Effective Diffusivity & Domain Structure',
             fontsize=14, fontweight='bold', y=1.0)
fig.tight_layout()
fig.savefig('results/effective_arrhenius_comparison.png', dpi=200, bbox_inches='tight')
print('Saved results/effective_arrhenius_comparison.png')
