"""Plot comparison of all model variants from the sweep."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import json, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

R_gas = 8.314

# Load sweep results
with open("results/sweep_results.json") as f:
    sweep = json.load(f)

# Load WAIC from each fit file
results = []
for tag, info in sweep.items():
    pkl_path = f"results/fit_{tag}.pkl"
    if os.path.exists(pkl_path):
        data = pickle.load(open(pkl_path, "rb"))
        waic = data.get("waic", None)
        chi2 = info.get("chi2_dof", None)
        results.append({"tag": tag, "waic": waic, "chi2_dof": chi2,
                        "config": info["config"], "elapsed": info["elapsed"]})

results.sort(key=lambda r: r["waic"] if r["waic"] is not None else 1e9)

# ── Plot 1: WAIC comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
tags = [r["tag"] for r in results if r["waic"] is not None]
waics = [r["waic"] for r in results if r["waic"] is not None]

colors = []
for r in results:
    if r["waic"] is None: continue
    t = r["tag"]
    if t.startswith("D"): colors.append("#2196F3")
    elif t.startswith("C"): colors.append("#4CAF50")
    elif t.startswith("N"): colors.append("#FF9800")

ax.barh(range(len(tags)), waics, color=colors, alpha=0.7)
ax.set_yticks(range(len(tags)))
ax.set_yticklabels(tags)
ax.set_xlabel("WAIC (lower = better)")
ax.set_title("Model Comparison by WAIC")
ax.invert_yaxis()

# Add value labels
for i, (w, t) in enumerate(zip(waics, tags)):
    ax.text(w + max(waics)*0.01, i, f"{w:.0f}", va='center', fontsize=9)

# Legend
from matplotlib.patches import Patch
ax.legend([Patch(color="#2196F3"), Patch(color="#4CAF50"), Patch(color="#FF9800")],
          ["Discrete domains", "Continuous distribution", "Learned noise"],
          loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig("results/waic_comparison.png", dpi=150)
print("Saved results/waic_comparison.png")

# ── Print ranking ──────────────────────────────────────────────
print("\n=== Model Ranking by WAIC ===")
print(f"{'Rank':>4s}  {'Tag':>5s}  {'WAIC':>8s}  {'χ²/dof':>8s}  {'Config':>30s}")
for i, r in enumerate(results):
    if r["waic"] is None: continue
    cfg = r["config"]
    desc = f"k={cfg.get('num_domains','?')}, ea={cfg.get('ea_mode','?')}"
    if cfg.get("continuous"): desc += ", continuous"
    if cfg.get("learn_sigma"): desc += ", σ-learned"
    print(f"{i+1:4d}  {r['tag']:>5s}  {r['waic']:8.0f}  {r['chi2_dof']:8.1f}  {desc:>30s}"
          if r['chi2_dof'] else f"{i+1:4d}  {r['tag']:>5s}  {r['waic']:8.0f}  {'N/A':>8s}  {desc:>30s}")

# ── Plot 2: T-t constraints for top 3 models ───────────────────
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

top3 = [r for r in results if r["waic"] is not None][:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
durations = [10, 100, 200, 500, 1300]
labels = ["10 My", "100 My", "200 My", "500 My", "Isothermal"]

for ax, r in zip(axes, top3):
    data = pickle.load(open(f"results/fit_{r['tag']}.pkl", "rb"))
    fs = data["samples"]
    nc = list(fs.values())[0].shape[0] if list(fs.values())[0].ndim == 1 else list(fs.values())[0].shape[1]

    if "logD0_r2" in fs and fs["logD0_r2"].ndim == 3:
        final_Ea = fs["Ea"][-1]
        final_logD = fs["logD0_r2"][-1]
        hrd_idx = np.argmin(final_logD, axis=1)
        nc_actual = final_Ea.shape[0]
        hrd_Ea = final_Ea[np.arange(nc_actual), hrd_idx]
        hrd_logD = final_logD[np.arange(nc_actual), hrd_idx]
    elif "mu_logD" in fs:
        # Continuous model: use mu - sigma as HRD estimate
        hrd_Ea = fs["Ea"][-1] if fs["Ea"].ndim > 1 else np.full(nc, 117.0)
        hrd_logD = fs["mu_logD"][-1] if fs["mu_logD"].ndim > 1 else fs["mu_logD"]
        if hrd_logD.ndim == 0:
            hrd_logD = np.full(nc, float(hrd_logD))
        nc_actual = len(hrd_logD)
    else:
        continue

    tt_results = {}
    for dur, label in zip(durations, labels):
        temps = np.array([max_temp(ea, ld, dur)
                         for ea, ld in zip(hrd_Ea[:nc_actual], hrd_logD[:nc_actual])])
        tt_results[label] = temps[~np.isnan(temps)]

    bp = ax.boxplot([tt_results[l] for l in labels if len(tt_results[l])>0],
                    positions=range(len([l for l in labels if len(tt_results[l])>0])),
                    widths=0.5, patch_artist=True, showfliers=False, whis=[2.5, 97.5])
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.4)
    ax.axhline(0, color='blue', ls='--', lw=1.5, alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_title(f"{r['tag']} (WAIC={r['waic']:.0f})", fontsize=11)
    if ax == axes[0]:
        ax.set_ylabel("Max temperature (°C)")

fig.suptitle("T-t Constraints: Top 3 Models by WAIC", fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig("results/top3_Tt_comparison.png", dpi=150)
print("Saved results/top3_Tt_comparison.png")
