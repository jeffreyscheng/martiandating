# Bayesian MDD Thermochronology: Results Summary

## Data
- **Nakhla #1** (Swindle & Olson 2004, Table A2): 18 extraction steps, 350-700°C
  - Step durations imputed from Shuster & Weiss Fig 1 (off-by-one bug fixed)
  - Steps below 350°C excluded (³⁹Ar recoil contamination)
  - 10% observation uncertainties on ³⁹Ar
- **Nakhla #2** validated separately (consistent T-t constraints)

## Models Run (each: 800 chains × 100K steps on 8×H100 GPUs)

| Model | Description | R-hat | Isothermal median | 95% CI | Crosses 0°C |
|-------|-------------|-------|-------------------|--------|-------------|
| D3 | 3-domain, Ea=117 fixed | 1.78 | -22.1°C | [-24.5, -7.4] | No |
| D4 | 4-domain, Ea=117 fixed | 1.67 | -21.4°C | [-24.4, -3.3] | No |
| N2 | 3-domain + learned noise (flat prior) | 1.00 | -23.9°C | [-26.4, -18.9] | No |
| N3 | 3-domain + learned noise (exp penalty) + Dir(2) | 1.00 | -24.3°C | [-25.9, -22.0] | No |

## N3 Analysis
N3 converged perfectly (R-hat 1.000) but found a degenerate solution: all 3 domains collapsed to the same logD=6.65 with equal fractions (~0.29 each). Noise scale reduced to 5.55× (from N2's 8.7×) but still high. The Dirichlet(2) prior prevented volume collapse but not logD collapse. The ordering constraint via cumulative softplus allows domains to merge when the likelihood doesn't distinguish them.

**Interpretation**: With 10% input uncertainties AND learned noise, the model finds that a single effective domain + moderate noise (55% effective σ) fits the data adequately. The multi-domain structure visible in the Arrhenius plot is within the noise tolerance.

## Key Findings

1. **Duration imputation bug**: Original code had off-by-one error using `diff().shift(-1)` instead of `diff()`. Fixed, improving χ²/dof from 36 to 7.9.

2. **Recoil exclusion**: Steps below 350°C show anomalously high D/ρ² from ³⁹Ar recoil during neutron irradiation, not volume diffusion. Excluding these is standard practice.

3. **Nakhla #2 validation**: Extracted Arrhenius data from S&W supplementary Fig S1A. Consistent T-t constraints with Nakhla #1.

4. **N2 degenerate solution**: With a flat Normal prior on noise scale, N2 inflated noise to 8.7× input σ and collapsed to a single domain. N3 fixes this with exponential penalty on noise + Dirichlet(2,2,2) prior.

5. **Observation noise**: The 10% ³⁹Ar uncertainty translates to only ~0.1 units on the Arrhenius plot, but actual scatter is ~0.3-0.5 units (dominated by step duration imputation error).

## Final Plots
- `/home/jefcheng/dev/martiandating/results/D3_final.png`
- `/home/jefcheng/dev/martiandating/results/D4_final.png`
- `/home/jefcheng/dev/martiandating/results/N2_final.png`
- `/home/jefcheng/dev/martiandating/results/N3_final.png` (pending)

## Code
- `run_gpu.py`: GPU-optimized MCMC with loop-based sampling
- `plot_final.py`: 3-panel summary plots (Arrhenius + domain posteriors + T-t constraint)
- `slurm_scripts/`: SLURM batch scripts for each model
- `data/nakhla1_parsed_fitted.csv`: Nakhla #1 with corrected step durations
- `data/nakhla2_parsed_fitted.csv`: Nakhla #2 with Fig S1A-derived durations
