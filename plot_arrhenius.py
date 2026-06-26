"""Plot Arrhenius diagram: observed vs posterior predictive D/rho^2."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle, numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

tag = sys.argv[1] if len(sys.argv) > 1 else "2d_med"
data = pickle.load(open(f"results/fit_{tag}.pkl", "rb"))
fs = data["samples"]
args_d = data["args"]
num_domains = args_d["num_domains"]
shared_ea = args_d.get("shared_ea", False)

# Load data
import pandas as pd
entire_df = pd.read_csv('data/nakhla1_parsed_fitted.csv')[
    ["Temp", "39Ar", "std_39Ar", "seconds_per_extraction_step"]]
total_39Ar = entire_df['39Ar'].sum()
entire_df['ΔF'] = entire_df['39Ar'] / total_39Ar
entire_df['F'] = entire_df['ΔF'].cumsum()
entire_df['T_K'] = entire_df['Temp'] + 273.15

def y_from_F(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6/np.sqrt(np.pi)
        s = (a - np.sqrt(a*a - 12*F)) / 6
        return s**2
    raise ValueError()

entire_df['y'] = entire_df['F'].apply(y_from_F)
filtered_df = entire_df.dropna()

# Observed Arrhenius: ln(D/rho^2) from cumulative release
y_vals = filtered_df['y'].values
dy = np.diff(y_vals, prepend=0)
dt_obs = filtered_df['seconds_per_extraction_step'].values
temps_K = filtered_df['T_K'].values

# D/rho^2 = dy/dt for each step
D_r2_obs = dy / dt_obs
D_r2_obs[D_r2_obs <= 0] = np.nan
ln_D_obs = np.log(D_r2_obs)
inv_T = 1e4 / temps_K

# Forward model
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

R_gas = 8.314

# For each posterior sample, compute the predicted Arrhenius plot
last_Ea = fs["Ea"][-1]        # (chains, domains)
last_logD = fs["logD0_r2"][-1]
last_phi_raw = fs["phi_raw"][-1] if "phi_raw" in fs else None
nc = last_Ea.shape[0]

fig, ax = plt.subplots(figsize=(7, 5))

# Plot posterior predictive Arrhenius lines
for c in range(nc):
    Ea_c = last_Ea[c]
    logD_c = last_logD[c]

    # For each domain, compute ln(D/rho^2) vs 1/T
    for d in range(num_domains):
        Ea_val = Ea_c[0] if shared_ea else Ea_c[d]
        logD0_val = logD_c[d]
        # ln(D/rho^2) = logD0 - Ea*1e3/(R*T)
        ln_D_pred = logD0_val - Ea_val * 1e3 / (R_gas * temps_K)
        ax.plot(inv_T, ln_D_pred, color=f'C{d}', alpha=max(5/nc, 0.01), lw=0.4)

# Plot observed data
ax.scatter(inv_T, ln_D_obs, color='k', s=30, zorder=10, label='Observed')

# Shuster & Weiss HRD reference line
Ea_ref, logD_ref = 117.0, 5.7
T_range = np.linspace(400, 1100, 100)
ln_D_ref = logD_ref - Ea_ref * 1e3 / (R_gas * T_range)
ax.plot(1e4/T_range, ln_D_ref, 'k--', lw=1.5, alpha=0.5, label='S&W HRD (Ea=117, ln(D₀/ρ²)=5.7)')

ax.set_xlabel('10⁴/T  (K⁻¹)', fontsize=12)
ax.set_ylabel('ln(D/ρ²)  (ln(s⁻¹))', fontsize=12)
ax.set_ylim(-24, -6)
ax.set_xlim(9, 22)

# Secondary x-axis: temperature in °C
ax2 = ax.secondary_xaxis('top', functions=(lambda x: 1e4/x - 273.15, lambda x: 1e4/(x+273.15)))
ax2.set_xlabel('Temperature (°C)')
ax2.set_xticks([700, 600, 500, 400, 350, 300, 250, 200])

# Domain legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color=f'C{d}', lw=2) for d in range(num_domains)]
handles += [Line2D([0],[0], color='k', marker='o', ls='', ms=5),
            Line2D([0],[0], color='k', ls='--', lw=1.5, alpha=0.5)]
labels = [f'Domain {d+1} posterior' for d in range(num_domains)]
labels += ['Observed', 'S&W HRD reference']
ax.legend(handles, labels, fontsize=9, loc='lower left')

ax.set_title(f'Arrhenius Plot: Observed vs Posterior ({tag})', fontsize=13)
fig.tight_layout()
fig.savefig(f'results/arrhenius_{tag}.png', dpi=150)
print(f'Saved results/arrhenius_{tag}.png')
