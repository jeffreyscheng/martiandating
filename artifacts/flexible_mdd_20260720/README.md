# Flexible MDD rerun, 2026-07-20

This directory is the compact, tracked record of the eight-A100 rerun. The full
posterior files remain under `results/flexible/` because each production run is
hundreds of megabytes. The exact sampler, analysis code, manifests, diagnostic
summaries, scientific summaries, and the primary/replicate analysis subsets are
retained here.

## What was fit

- Nakhla stepped-heating data from `data/nakhla1_parsed_fitted.csv`.
- All 21 timed steps from 275–700°C are propagated through the diffusion model.
  The likelihood scores the 18 steps from 350–700°C. Fractions keep their
  whole-experiment normalization; the retained window is not renormalized.
- The diffusion-scale distribution is a smooth logistic-normal field on a
  fixed grid over `ln(D0/r²)`, not a two-spike mixture. The primary grid has 28
  bins on [-4, 14]. A single activation energy has prior N(117, 5.4²) kJ/mol.
- The primary observation scale combines reported measurement uncertainty with
  a fixed 10% relative discrepancy term in quadrature. It is not learned from
  the data. Fifteen- and twenty-percent runs diagnose this assumption.
- An affine map from the MAP Hessian preconditions the target. This is only a
  coordinate change. The NUTS kernel is then frozen, 500 settling draws are
  discarded, and all later states are retained.

## Predeclared gates

Sampler publication gates were max rank/folded R-hat <= 1.01, bulk and tail ESS
>= 2,000, zero divergences, max-depth frequency < 1%, every-chain BFMI > 0.30,
and mean mass on the grid edges < 0.5%. Scientific checks were at least 15 of
18 observations inside 90% posterior-predictive intervals, standardized
residual RMS < 1.5, maximum absolute median residual < 3 sigma, temperature
quantile MCSE < 1°C, and agreement of the independent replicate within 1°C at
the median and 2°C at interval endpoints.

Every retained run passed the sampler gates. The primary run failed one
scientific gate: 14/18 rather than 15/18 observations lie in the 90% predictive
interval. The four misses are the contiguous 475, 525, 550, and 562°C steps.
Changing grid resolution or allowing a rougher diffusion distribution does not
remove that pattern. A 15% fixed discrepancy term covers 18/18 observations,
but should not be selected merely because it passes the check.

## Results

The interval column is the central 95% interval and median for the maximum
constant temperature that loses 1% of the present-day radiogenic argon during
a 100 My event ending at the present. `P(limit > 0°C)` is not the probability
that Mars was warm; it is the posterior probability that this particular
argon-loss upper limit crosses freezing.

| run | retained states | max R-hat | min bulk / tail ESS | divergences | min BFMI | 100 My limit: [2.5%, median, 97.5%] °C | P(limit > 0°C) | PPC 90% coverage; RMS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| primary, 28 bins, 10% | 1,024,000 | 1.0037 | 61,670 / 55,607 | 0 | 0.907 | [-13.76, -5.22, 3.31] | 0.130 | 14/18; 1.393 |
| independent replicate | 1,024,000 | 1.0041 | 61,118 / 53,736 | 0 | 0.900 | [-13.66, -5.29, 3.33] | 0.128 | 14/18; 1.392 |
| rougher prior | 512,000 | 1.0094 | 25,870 / 18,339 | 0 | 0.829 | [-14.87, -5.58, 3.64] | 0.119 | 14/18; 1.323 |
| 20-bin grid | 512,000 | 1.0062 | 39,374 / 32,299 | 0 | 0.881 | [-14.11, -4.76, 3.58] | 0.103 | 14/18; 1.385 |
| 36-bin grid | 512,000 | 1.0090 | 24,756 / 21,403 | 0 | 0.879 | [-13.63, -4.90, 3.43] | 0.130 | 14/18; 1.395 |
| 15% discrepancy | 512,000 | 1.0032 | 69,655 / 75,626 | 0 | 0.895 | [-15.64, -5.32, 5.08] | 0.163 | 18/18; 0.967 |
| 20% discrepancy | 512,000 | 1.0027 | 82,934 / 99,834 | 0 | 0.866 | [-17.25, -5.66, 5.59] | 0.163 | 18/18; 0.741 |

The primary and independently seeded replicate differ by 0.07°C at the median
and at most 0.10°C at the central-interval endpoints. The 20-, 28-, and 36-bin
grids and the rougher prior give similar recent-event limits. Monte Carlo error
on all reported primary quantiles is below 0.09°C.

## Thermal interpretation

The thermal transform uses a 1.3 Ga rock age, a 1.248 Ga potassium-40 half-life,
a constant-temperature event, and a 1% total-radiogenic-argon loss threshold.
It integrates argon produced during the event and accounts for post-event
radiogenic regrowth in the present-day inventory. These are conditional upper
limits, not a posterior over the actual Martian temperature history.

Timing matters. Under the primary model, a 100 My event ending now has interval
[-13.76, -5.22, 3.31]°C. If it ended 250, 500, or 1,000 My ago, the intervals
are respectively [-12.18, -3.63, 4.93]°C, [-9.97, -1.38, 7.21]°C, and
[0.04, 8.66, 17.35]°C. The last lower endpoint is not robust to the prior and
observation-error sensitivities, so it should not be presented as a positive
lower bound.

The defensible claim from this rerun is narrow: after replacing the two-spike
point estimate with a flexible posterior and propagating uncertainty, a 100 My
above-freezing event is no longer cleanly excluded by the argon model. The run
does not show that such an event occurred. The observation-error model and the
1% loss threshold still require scientific justification before this becomes a
paper-level conclusion.

## Files

Each run directory contains its exact manifest, final sampler diagnostics, and
analysis summary. `primary/` also contains the publication plots and a compact
25,600-sample posterior subset; `replicate/` contains the independently seeded
subset. The full run can be reproduced with `run_flexible_mdd.py` and analyzed
with `analyze_flexible_mdd.py`.
