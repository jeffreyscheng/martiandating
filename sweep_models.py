"""Sweep all model variants, compute WAIC, rank them."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle, time, json, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = "data/nakhla1_parsed_fitted.csv"
MIN_TEMP = 350
NUM_CHAINS = 200
NUM_WARMUP = 2000
NUM_STEPS = 500
EA_SIGMA = 20.0

configs = [
    # Discrete, fixed Ea
    {"tag": "D1", "num_domains": 1, "ea_mode": "fixed"},
    {"tag": "D2", "num_domains": 2, "ea_mode": "fixed"},
    {"tag": "D3", "num_domains": 3, "ea_mode": "fixed"},
    {"tag": "D4", "num_domains": 4, "ea_mode": "fixed"},
    # Discrete, shared Ea
    {"tag": "D5", "num_domains": 2, "ea_mode": "shared"},
    {"tag": "D6", "num_domains": 3, "ea_mode": "shared"},
    # Discrete, per-domain Ea
    {"tag": "D7", "num_domains": 2, "ea_mode": "per_domain"},
    {"tag": "D8", "num_domains": 3, "ea_mode": "per_domain"},
    # Continuous, fixed Ea
    {"tag": "C1", "num_domains": 1, "ea_mode": "fixed", "continuous": True, "n_bins": 20},
    {"tag": "C2", "num_domains": 1, "ea_mode": "fixed", "continuous": True, "n_bins": 50},
    # Continuous, sampled Ea
    {"tag": "C3", "num_domains": 1, "ea_mode": "shared", "continuous": True, "n_bins": 20},
    {"tag": "C4", "num_domains": 1, "ea_mode": "shared", "continuous": True, "n_bins": 20, "dist": "lognormal"},
    # With learned noise
    {"tag": "N1", "num_domains": 2, "ea_mode": "per_domain", "learn_sigma": True},
    {"tag": "N2", "num_domains": 1, "ea_mode": "fixed", "continuous": True, "n_bins": 20, "learn_sigma": True},
]

def build_cmd(cfg):
    cmd = [sys.executable, "run_fit.py",
           "--data", DATA,
           "--min_temp", str(MIN_TEMP),
           "--num_chains", str(NUM_CHAINS),
           "--num_warmup", str(NUM_WARMUP),
           "--num_steps", str(NUM_STEPS),
           "--ea_sigma", str(EA_SIGMA),
           "--tag", cfg["tag"],
           "--num_domains", str(cfg["num_domains"]),
           "--seed", "42"]

    if cfg.get("ea_mode") == "fixed":
        cmd.append("--ea_fixed")
    elif cfg.get("ea_mode") == "shared":
        cmd.append("--shared_ea")
    # per_domain is default

    if cfg.get("continuous"):
        cmd.extend(["--continuous", "--n_bins", str(cfg.get("n_bins", 20))])
        if cfg.get("dist") == "lognormal":
            cmd.append("--lognormal_dist")

    if cfg.get("learn_sigma"):
        cmd.append("--learn_sigma")

    return cmd

# Run all configs
results = {}
for i, cfg in enumerate(configs):
    tag = cfg["tag"]
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(configs)}] Running model {tag}...")
    print(f"{'='*60}")

    cmd = build_cmd(cfg)
    print(f"  CMD: {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    # Parse output for key metrics
    stdout = proc.stdout + proc.stderr
    chi2 = None
    isothermal = None
    for line in stdout.split('\n'):
        if 'χ²/dof' in line:
            try:
                chi2 = float(line.split('=')[1].strip())
            except:
                pass
        if 'Isothermal' in line and 'crosses' in line:
            isothermal = line.strip()
        if 'WAIC' in line and '=' in line:
            try:
                waic = float(line.split('=')[1].split()[0])
            except:
                waic = None

    results[tag] = {
        "config": cfg,
        "chi2_dof": chi2,
        "isothermal": isothermal,
        "elapsed": elapsed,
        "returncode": proc.returncode,
    }

    status = "OK" if proc.returncode == 0 else "FAILED"
    print(f"  {status} in {elapsed:.0f}s, χ²/dof={chi2}")
    if isothermal:
        print(f"  {isothermal}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Tag':>5s}  {'χ²/dof':>8s}  {'Time':>6s}  {'Status':>7s}")
for tag, r in sorted(results.items(), key=lambda x: x[1].get('chi2_dof') or 1e9):
    chi2 = f"{r['chi2_dof']:.1f}" if r['chi2_dof'] else "N/A"
    print(f"{tag:>5s}  {chi2:>8s}  {r['elapsed']:5.0f}s  {'OK' if r['returncode']==0 else 'FAIL':>7s}")

# Save results
with open("results/sweep_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results/sweep_results.json")
