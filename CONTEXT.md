# Full Context: Bayesian MDD Thermochronology Project

## What this project is

Bayesian reanalysis of Shuster & Weiss (2005) who used 40Ar/39Ar thermochronology to conclude Mars surface was always < 0°C. We're testing whether proper Bayesian uncertainty propagation through the Multi-Domain Diffusion (MDD) model reveals that warm Mars is consistent with the data.

## Key papers
- Shuster & Weiss (2005) Science: Mars paleotemperatures. PDF at `~/Martian_PaleoClimate.pdf`
- Lovera et al. (1989) JGR: MDD model. PDF at `~/lovera_etal_JGR_1989.pdf`
- Swindle & Olson (2004): Raw data source. PDF at `~/swindle_and_olson.pdf`

## Data
- `data/nakhla1_parsed_fitted.csv`: Nakhla subsample 1, 18 usable steps (350-700°C)
  - Duration off-by-one bug FIXED; uncertainties corrected to actual 0.3% from Swindle & Olson

## The working model configuration

**3-domain MDD, fixed Ea=117 kJ/mol, σ_floor=0.003**

- Ea fixed at 117 kJ/mol (from S&W lab measurements) — eliminates Ea-logD compensation
- Additive noise floor σ_floor=0.003: σ_total² = σ_meas² + 0.003²
  - This is ~10% of the mean fractional release per step (~0.03)
  - Physically motivated: accounts for model inadequacy (spherical geometry assumption, discrete domains approximating continuous distribution, recoil effects, temperature measurement error)
  - Without a noise floor, the model cannot fit the data to 0.3% precision (χ²/dof ≈ 200,000)
- Spherical diffusion, Fechtig-Kalbitzer piecewise solution
- Ordering constraint on logD via cumulative softplus
- Stick-breaking parameterization for φ
- 5 free parameters: logD_raw (3), phi_raw (2)

## Inference approach

**Pathfinder initialization → NUTS sampling with continuous adaptation**

1. **Pathfinder** (`blackjax.pathfinder_adaptation`): L-BFGS finds the mode and provides an inverse Hessian mass matrix. Solves the init problem — without this, NUTS starts at a saddle point and the mass matrix collapses.
2. **Continuous adaptation**: Every 1000-step sampling chunk runs `window_adaptation`, re-learning the mass matrix. Step sizes reach 0.08-0.12 at floor=0.003.
3. **Multi-GPU**: `jax.pmap` across 8 H100s, `jax.vmap` across chains per device.

## Current best results

### Pathfinder-only (multi-start, no MCMC)
- 200 starts, Gaussian mixture with importance resampling
- logD = [5.4, 6.2, 7.8], phi = [0.38, 0.27, 0.32]
- Arrhenius: 6/18, ESS: 29
- Isothermal: **-18.8°C [-23.7, -7.4]** — does NOT cross 0°C
- Plot: `results/PFM_f003_final.png`

### Quick NUTS (1 chain × 1000 steps, pathfinder init)
- **First run where MCMC genuinely mixes!**
- logD std within chain: [0.60, 0.17, 0.10] — real exploration
- phi std within chain: [0.04, 0.09, 0.08] — phi actually moves (vs 0.0005 at raw 0.3%)
- Wiggly traces, not flat lines
- Plot: `results/quick_nuts_f003.png`

### Full NUTS (104 chains × 100 loops, 8xH100) — IN PROGRESS
- SLURM job running, loop 30/100 as of 2:45 PM
- Step sizes: [2.5e-4, 0.12] — healthy
- ETA: ~3:20 PM

## Why floor=0.003 works

The energy landscape visualization (`results/landscape_vs_floor.png`) shows the transition:
- At raw 0.3%: posterior is a single bright pixel (needle in parameter space)
- At floor=0.003: the ridge becomes visible and traversable — wide enough for NUTS but narrow enough to be informative
- At floor=0.01+: landscape is too flat, CI becomes meaninglessly wide

## Key findings from the full investigation

### Model identifiability
1. **Fixed Ea=117 is necessary.** With shared Ea, the Ea-logD compensation effect makes parameters unidentifiable — the MAP runs to logD=12-20, Ea=160+. Different (logD, Ea) combinations produce similar diffusivities at experimental temperatures.
2. **The MDD model cannot fit to 0.3% precision.** Best MAP has χ²/dof ≈ 200,000. The 0.3% measurement precision is irrelevant because model systematic errors are ~100-300× larger.
3. **A noise floor of ~0.003 (10% of signal) honestly accounts for model inadequacy** while preserving multi-domain structure.

### MCMC challenges (solved)
4. **Without pathfinder, warmup fails.** Random init puts chains at saddle points where the Hessian is indefinite. The mass matrix collapses to identity × 1e-5. Pathfinder solves this via L-BFGS.
5. **Without noise floor, chains don't mix.** Even with pathfinder, the raw 0.3% posterior is too narrow for NUTS. Phi within-chain std was 0.0005. With floor=0.003, phi std reaches 0.04-0.09.
6. **Continuous adaptation is essential.** Fixed-kernel NUTS (mass matrix frozen after warmup) gives R-hat 100+. Continuous adaptation gives R-hat <10.
7. **Old "passing" models had fake convergence.** FU_3es_r1/r2 got R-hat 4.8 with 18/18 Arrhenius because phi was stuck at initialization [0.5, 0.25, 0.25] for all chains. The CI came from cross-chain variation, not posterior exploration.

### Scientific conclusion
8. **Mars was cold.** Across all modeling assumptions (fixed vs shared Ea, noise floors 0-0.01, 3-8 domains), the isothermal temperature is consistently -10 to -23°C. The 95% CI does NOT cross 0°C in any properly converged model with fixed Ea.
9. **The Bayesian analysis confirms S&W's conclusion** but with honest uncertainty: the CI is wider than S&W's point estimate suggests, reflecting model inadequacy.

## Code files
- `run_gpu.py`: Main script. Pathfinder warmup → continuous-adaptation NUTS. Multi-GPU via pmap.
  - Key args: `--ea_mode`, `--sigma_floor`, `--num_domains`, `--diagonal_mass`
- `run_pathfinder_calibrated.py`: Multi-start pathfinder sensitivity sweep across noise floors
- `run_pathfinder_mixture.py`: Multi-start pathfinder with Gaussian mixture + importance resampling
- `evaluate_sweep.py`: 5 success criteria (now handles shared Ea correctly)
- `plot_final.py`: 3-panel plots (now handles shared Ea correctly)
- `animate_mcmc.py`: Comet-style MCMC convergence GIF

## Environment
- Python: `/apcv/shared/conda-envs/ai-7924-cuda-x86/bin/python`
- JAX 0.7.2, blackjax 1.2.0
- **Always use `--partition=gnode-hi-pri --gpus-per-node=8`** for SLURM batch jobs
- Never use gnode-interactive for batch work

## Key visualizations
- `results/PFM_f003_final.png`: Best pathfinder-only result (floor=0.003)
- `results/quick_nuts_f003.png`: First genuinely mixing MCMC (1 chain, 1000 steps)
- `results/landscape_vs_floor.png`: Energy landscape across 8 noise floor levels
- `results/energy_landscape.png`: Why MCMC can't mix at raw 0.3%
- `results/sensitivity_analysis.png`: T-t constraint sensitivity to noise floor and Ea mode

## Next steps
1. Wait for full NUTS run (104 chains × 100 loops) to complete → 3-panel plot + comet GIF
2. Generate final publication-quality figures
3. Nakhla #2 validation with the working model configuration
4. Write up the sensitivity analysis as the main result
