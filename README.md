# martiandating

## Posterior trace animation

`run_comet_nuts.py` produces the focused, ordered two-domain fit used by the
website's comet animation. It freezes NUTS during retained sampling, retains
every draw, and refuses `--strict` runs unless rank-normalized R-hat is below
1.01, bulk ESS is at least 400, there are no divergences, and fewer than 5% of
draws reach the maximum tree depth.

```bash
# Pilot used only to estimate an affine preconditioner.
python run_comet_nuts.py --chains 8 --warmup 1500 --draws 3000 \
  --target-accept 0.80 --tag comet_nuts

# Fixed post-adaptation kernel used for the published animation.
python run_comet_nuts.py --chains 8 --warmup 0 --draws 3000 \
  --fixed-step-size 0.12 --whiten-from results/fit_comet_nuts.pkl \
  --tag comet_nuts_final --strict
python animate_mcmc.py comet_nuts_final
```

The July 2026 render passed with maximum R-hat 1.003, minimum bulk ESS 2,306,
minimum tail ESS 2,022, zero divergences, and zero maximum-tree-depth hits. This
is a diagnostic two-domain fit, not a replacement for the project-wide
model-comparison sweep.
