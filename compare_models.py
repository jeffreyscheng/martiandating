"""Compare T-t constraints across model variants."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = {
    "2-domain\n(per-domain Ea)": "results/fit_2d_med.pkl",
    "2-domain\n(shared Ea)": "results/fit_2d_shared_ea.pkl",
    "3-domain\n(per-domain Ea)": "results/fit_3d_med.pkl",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

durations = [10, 100, 200, 500, 1300]
labels = ["10 My", "100 My", "200 My", "500 My", "Isothermal"]

from scipy.optimize import brentq
R_gas = 8.314

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

for ax, (name, path) in zip(axes, models.items()):
    data = pickle.load(open(path, "rb"))
    fs = data["samples"]
    nc = fs["Ea"].shape[1]
    nd = fs["Ea"].shape[2]

    final_Ea = fs["Ea"][-1]
    final_logD = fs["logD0_r2"][-1]
    hrd_idx = np.argmin(final_logD, axis=1)
    hrd_Ea = final_Ea[np.arange(nc), hrd_idx]
    hrd_logD = final_logD[np.arange(nc), hrd_idx]

    results = {}
    for dur, label in zip(durations, labels):
        temps = np.array([max_temp(ea, ld, dur) for ea, ld in zip(hrd_Ea, hrd_logD)])
        results[label] = temps[~np.isnan(temps)]

    bp = ax.boxplot([results[l] for l in labels],
                    positions=range(len(labels)), widths=0.5,
                    patch_artist=True, showfliers=False, whis=[2.5, 97.5])
    for patch in bp['boxes']:
        patch.set_facecolor('#4a90d9')
        patch.set_alpha(0.5)
    ax.axhline(0, color='blue', ls='--', lw=1.5, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_title(name, fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel('Maximum temperature (°C)', fontsize=11)

    # Print summary
    iso = results["Isothermal"]
    if len(iso) > 0:
        med = np.median(iso)
        lo, hi = np.percentile(iso, [2.5, 97.5])
        print(f"{name.replace(chr(10),' ')}: isothermal median={med:.1f}°C, 95%CI=[{lo:.1f}, {hi:.1f}]")

fig.suptitle("Mars Temperature Constraint: Bayesian 95% CI Across Model Variants", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("results/model_comparison.png", dpi=150, bbox_inches='tight')
print("Saved results/model_comparison.png")
