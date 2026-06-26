"""Headline figure: Arrhenius fit + T-t constraint showing what Bayesian uncertainty adds."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd

R_gas = 8.314

# Load data
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
        return ((a - np.sqrt(a*a - 12*F)) / 6)**2
    raise ValueError()

entire_df['y'] = entire_df['F'].apply(y_from_F)
filtered_df = entire_df.dropna()
temps_K = filtered_df['T_K'].values
dy = np.diff(filtered_df['y'].values, prepend=0)
dt_obs = filtered_df['seconds_per_extraction_step'].values
D_r2_obs = dy / dt_obs
D_r2_obs[D_r2_obs <= 0] = np.nan
ln_D_obs = np.log(D_r2_obs)
inv_T = 1e4 / temps_K

# Load best Bayesian result
data = pickle.load(open("results/fit_2d_med.pkl", "rb"))
fs = data["samples"]
nc = fs["Ea"].shape[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Panel A: Arrhenius plot ────────────────────────────────────
# S&W point estimate (single HRD line)
Ea_sw, logD_sw = 117.0, 5.7
T_line = np.linspace(400, 1200, 200)
ax1.plot(1e4/T_line, logD_sw - Ea_sw*1e3/(R_gas*T_line),
         'k-', lw=2.5, label='Shuster & Weiss point estimate', zorder=5)

# Bayesian posterior domain lines
last_Ea = fs["Ea"][-1]
last_logD = fs["logD0_r2"][-1]
for c in range(nc):
    for d in range(2):
        Ea_val = last_Ea[c, d]
        logD_val = last_logD[c, d]
        ln_pred = logD_val - Ea_val*1e3/(R_gas*temps_K)
        color = '#2196F3' if d == 0 else '#FF9800'
        ax1.plot(inv_T, ln_pred, color=color, alpha=max(8/nc, 0.02), lw=0.5)

# Observed
ax1.scatter(inv_T, ln_D_obs, color='k', s=25, zorder=10, label='Observed data')

ax1.set_xlabel('10⁴/T  (K⁻¹)', fontsize=12)
ax1.set_ylabel('ln(D/ρ²)  (ln(s⁻¹))', fontsize=12)
ax1.set_ylim(-24, -6)
ax1.set_xlim(9, 22)

ax1_top = ax1.secondary_xaxis('top',
    functions=(lambda x: 1e4/x - 273.15, lambda x: 1e4/(x+273.15)))
ax1_top.set_xlabel('Temperature (°C)')
ax1_top.set_xticks([700, 600, 500, 400, 350, 300, 250, 200])

from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color='k', lw=2.5),
    Line2D([0],[0], color='#2196F3', lw=2, alpha=0.7),
    Line2D([0],[0], color='#FF9800', lw=2, alpha=0.7),
    Line2D([0],[0], color='k', marker='o', ls='', ms=5),
]
ax1.legend(handles, ['S&W point estimate (HRD only)',
                      'Bayesian LRD posterior', 'Bayesian HRD posterior',
                      'Observed'], fontsize=9, loc='lower left')
ax1.set_title('A.  Arrhenius Plot', fontsize=13, fontweight='bold', loc='left')

# ── Panel B: T-t constraint comparison ─────────────────────────
def max_temp(Ea_kJ, logD0_r2_val, dur_Ma, max_loss=0.01):
    dur_s = dur_Ma * 1e6 * 3.15e7
    D0r2 = np.exp(logD0_r2_val)
    def f(T_C):
        T_K = T_C + 273.15
        y = D0r2 * np.exp(-Ea_kJ*1e3/(R_gas*T_K)) * dur_s
        F = 6*np.sqrt(y/np.pi)-3*y if y < 0.3 else 1-(6/np.pi**2)*np.exp(-np.pi**2*y)
        return np.clip(F,0,1) - max_loss
    try: return brentq(f, -273, 1000, xtol=0.1)
    except: return np.nan

durations = [10, 100, 200, 500, 1300]
labels = ["10", "100", "200", "500", "1300\n(isothermal)"]

# S&W point estimate
sw_temps = [max_temp(117.0, 5.7, d) for d in durations]

# Bayesian posterior
hrd_idx = np.argmin(last_logD, axis=1)
hrd_Ea = last_Ea[np.arange(nc), hrd_idx]
hrd_logD = last_logD[np.arange(nc), hrd_idx]

x_pos = np.arange(len(durations))
width = 0.35

# Bayesian boxplots
bayes_results = []
for dur in durations:
    temps_arr = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
    bayes_results.append(temps_arr[~np.isnan(temps_arr)])

bp = ax2.boxplot(bayes_results, positions=x_pos + width/2, widths=width,
                 patch_artist=True, showfliers=False, whis=[2.5, 97.5])
for patch in bp['boxes']:
    patch.set_facecolor('#2196F3')
    patch.set_alpha(0.4)
for element in ['whiskers', 'caps']:
    for line in bp[element]:
        line.set_color('#2196F3')

# S&W point estimates as red diamonds
ax2.scatter(x_pos - width/2, sw_temps, color='#D32F2F', s=100, marker='D',
            zorder=10, label='S&W point estimate')

# 0°C line
ax2.axhline(0, color='blue', ls='--', lw=1.5, alpha=0.6, label='0°C (freezing)')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_xlabel('Excursion duration (My)', fontsize=12)
ax2.set_ylabel('Maximum temperature (°C)', fontsize=12)

ax2.legend(fontsize=9, loc='upper right')
ax2.set_title('B.  Mars Temperature Constraint', fontsize=13, fontweight='bold', loc='left')

# Annotate the key finding
ax2.annotate('Bayesian 95% CI\ncrosses 0°C',
             xy=(4 + width/2, np.percentile(bayes_results[4], 97.5)),
             xytext=(3, 70), fontsize=10, color='#D32F2F',
             arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
             ha='center')

fig.suptitle('Bayesian Reanalysis of Martian Nakhlite Thermochronology',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('results/headline_figure.png', dpi=200, bbox_inches='tight')
print('Saved results/headline_figure.png')
