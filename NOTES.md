# Bayesian Thermochronology of Martian Nakhlites — Notes

## Scientific question

Shuster & Weiss (2005) used 40Ar/39Ar thermochronology to conclude Mars surface was always < 0°C. This conflicts with geomorphological evidence (channels, deltas). Can a proper Bayesian treatment of the MDD model parameters reveal that warmer Mars is consistent with the data?

## The MDD forward model

Lovera et al. (1989): multiple diffusion domains, all sharing Eₐ and D₀, differing only in size ρⱼ and volume fraction φⱼ. Spherical diffusion geometry assumed. Fechtig & Kalbitzer (1966) piecewise solution:
- Low-loss (y < 0.3): F = (6/√π)√y − 3y
- High-loss (y ≥ 0.3): F = 1 − (6/π²)exp(−π²y)

Where y = ∫ D(T)dt / ρ² is the dimensionless diffusion progress.

## Data

Swindle & Olson Nakhla subsample 1: 28 extraction steps (250°C–1250°C). Step durations imputed from Shuster & Weiss Fig 1 (only valid ≤750°C → 22 usable steps). Observable: ΔF (fractional 39Ar release per step).

## What the existing code does

1. Parses raw data with parenthetical uncertainties
2. Imputes step durations by fitting to S&W Fig 1 Arrhenius plot
3. Implements 2-domain forward model in JAX (vectorized, differentiable)
4. Runs NUTS HMC via Blackjax with vmapped parallel chains
5. Produces trace plots and animated posterior evolution GIFs

## What was missing (fixed May 2026)

1. **Priors were commented out** — now enabled: lognormal on Eₐ (mode=117, σ=5.0), normal on ln(D₀/ρ²) (mean=5.7, σ=2.0), Dirichlet(1,1) on φ
2. **Eₐ was frozen at 117 kJ/mol** — now sampled from the lognormal prior
3. **Domain repulsion was disabled** — now active with weight 1e2 to prevent collapse
4. **Sampling budget was too small** — increased to 2000 warmup + 500 samples

## The key Bayesian output

For each posterior sample (Eₐ, D₀/ρ², φ), compute the max constant temperature producing ≤1% Ar loss from the HRD over a given excursion duration. The distribution of max-T values IS the Bayesian answer. If the 95% CI crosses 0°C for any duration, then "warm Mars" is not excluded by the data.

## Parameters of the 2-domain model

Per-domain (k=2):
- ρⱼ (implicit via D₀/ρⱼ²)
- φⱼ (volume fraction, sum-to-1 via stick-breaking)

Shared:
- Eₐ (activation energy, kJ/mol) — same for all domains per Lovera assumption
- D₀ (frequency factor) — absorbed into D₀/ρⱼ² for each domain

Total free parameters: 2×Eₐ (if allowed to vary per domain) + 2×logD₀/ρ² + 1×φ_raw = 5.
With shared Eₐ: 1×Eₐ + 2×logD₀/ρ² + 1×φ_raw = 4.

Note: current implementation allows per-domain Eₐ. The Lovera assumption (shared Eₐ) could be imposed as a strong prior or hard constraint. Per-domain Eₐ is more flexible but less physically motivated.

## Shuster & Weiss reported uncertainties

HRD: Eₐ = 117 ± 5.4 kJ/mol, ln(D₀/a²) = 5.7 ± 0.9 ln(s⁻¹)
LRD: volume fraction fᵥ = 3%, kinetics predict ~zero 40Ar retention

## Duration imputation bug fix (May 2026)

The original `fit_fig1_to_get_extraction_durations.ipynb` had an off-by-one error: it used `df['y'].diff().shift(-1)` instead of `df['y'].diff()`. This means each step's duration was computed using the NEXT step's y-increment. After fixing, Nakhla #1 χ²/dof improved from 36 to 7.9.

## Nakhla #2 validation (May 2026)

Extracted Arrhenius data points from Shuster & Weiss supplementary Figure S1A using PyMuPDF vector graphics extraction. Used precise y-axis calibration from tick label positions. Imputed step durations using same method as Nakhla #1. Results: T-t constraints consistent with Nakhla #1 (isothermal median -12.6°C, 95% CI crosses 0°C).

The reported measurement uncertainties for Nakhla #2 (0.3% from Swindle & Olson Table A3) are much smaller than the 10% used for Nakhla #1. Using consistent 10% model-uncertainty for both gives comparable χ²/dof.

## Key modeling choices and tradeoffs

1. **Prior width on Eₐ**: Wide (σ=5) lets data speak. If convergence is poor, tighten.
2. **Number of domains**: k=2 matches S&W. k=3 matches Lovera. Can compare via posterior predictive.
3. **Shared vs per-domain Eₐ**: Lovera says shared. Letting it vary per domain is a relaxation of that assumption — the posterior should tell us if the data supports different Eₐ values.
4. **Domain repulsion**: Needed to prevent multimodality from domain label switching. Weight 1e2 is a tuning parameter.
5. **Spherical geometry**: Assumed throughout. Planar geometry would change the Fechtig-Kalbitzer formula.
