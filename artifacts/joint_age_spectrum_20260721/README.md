# Joint Nakhla age-spectrum analysis

This artifact supersedes the threshold-based temperature transform in
`artifacts/flexible_mdd_20260720`. It fits the laboratory 39Ar release and the
natural normalized radiogenic-Ar spectrum in one likelihood. No fixed argon-
loss criterion is used to infer temperature.

One expanded-ensemble NUTS run marginalizes over the mutually exclusive
duration/onset grid. Responsibility-weighted samples recover the exact
posterior conditional on each grid cell without refitting the same laboratory
experiment or counting the observed natural spectrum more than once.

For a 100 My excursion ending at the present, under a uniform -100 to 100 C
temperature prior, the central 95% interval and median are
`[-94.45, -18.57, -4.47] C`; the conditional
posterior probability above 0 C is
`0.00252`.

Sampler diagnostics: max R-hat 1.0010; minimum bulk/tail ESS 83413/96173; minimum cell responsibility ESS 63341; divergences 0; minimum BFMI 0.908.

The hex-comet animation in the preceding flexible-MDD artifact came from a
different laboratory target: it normalized by all extracted 39Ar and used the
legacy spherical-release approximation. Importance reuse fails decisively
(Pareto k `3.44`, ESS
`1.06` of 25,600). A
separate corrected laboratory-only control passed its sampler gates. Its six
shared diffusion/activation-energy marginals overlap the joint conditionals by
at least `93.2%` across all 21 timing cells.
See `marginal_audit_summary.json` and the two marginal-comparison figures.

The cold lower tail is prior-dominated. Event timing changes the upper
constraint and is tabulated in `analysis_summary.json`. This is a Nakhla
analysis covering the last 1.3 Gy; it does not test Noachian denudation or
replace the ALH84001 part of Shuster and Weiss (2005).

`compact_posterior.npz` contains 100 deterministic draws per chain for
inspection. Full checkpoints are intentionally not tracked.
