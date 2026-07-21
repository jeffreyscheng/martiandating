# Joint diffusion and thermochronology model

## Target

Infer a constant-temperature Martian excursion directly from both observations:

1. the laboratory `39Ar` release schedule, which constrains the diffusion-domain
   distribution and activation energy; and
2. the corrected step-heating radiogenic `40Ar*/39Ar` spectrum, normalized to
   its bulk ratio, which constrains the natural thermal history.

The posterior is

```text
p(theta, T, tau | y39, yage)
  proportional to
p(y39 | theta) p(yage | theta, T, tau) p(theta) p(T) p(tau).
```

`theta` denotes the flexible diffusion distribution and shared activation
energy. `T` is excursion temperature and `tau` describes its timing and
duration. No retained-argon threshold appears in this model.

## Natural argon forward model

Time runs forward from feldspar closure to the present. Potassium decays at
`lambda = log(2) / 1248 My`. The branching ratio to argon is a common factor
and cancels from apparent ages.

For a spherical domain, the uniform concentration has diffusion eigenmode
weights

```text
b_n = 6 / (pi^2 n^2),       q_n = pi^2 n^2.
```

Before an excursion, during it, and after it, radiogenic argon is produced
continuously. Argon present before or produced during the excursion is
diffused mode by mode. Argon produced afterward remains uniform. The resulting
modal profile is then propagated through every laboratory heating step.

For laboratory cumulative diffusion progress `Y_j`, mode `n` releases

```text
exp(-q_n Y_(j-1)) - exp(-q_n Y_j)
```

during step `j`. Neutron-produced `39Ar` begins uniform; its exact spherical
release is used as the denominator. The apparent ages tabulated by Swindle &
Olson are converted to ratios proportional to

```text
R_j proportional to exp(lambda * t_j) - 1.
```

The common irradiation factor is removed by dividing observations and
predictions by their bulk ratio. This is the `R/R_bulk` observable in Shuster
& Weiss Figure 2. The implementation can also map predictions back to apparent
ages for diagnostic checks. A closed system returns `R/R_bulk = 1` exactly.

## Data and likelihood

- Source: Swindle & Olson Nakhla subsample 1, as transcribed in
  `data/nakhla1.csv`.
- Use the published corrected apparent ages to reconstruct normalized ratios
  rather than treating the uncorrected raw isotope columns as radiogenic
  `40Ar*`. Reconstructing irradiation and interference corrections from the
  raw columns is unnecessary and would introduce avoidable assumptions.
- The primary likelihood scores timed 350--700 C steps. Lower-temperature
  releases are excluded from the diffusion calibration because of recoil and
  alteration; higher-temperature releases do not have reconstructed heating
  durations.
- Published age errors are measurement errors. A robust Student-t discrepancy
  term must be fitted or sensitivity-tested because alteration, recoil, excess
  argon, geometry, and duration reconstruction are not represented by those
  errors.

## Superseded analysis

The existing flexible posterior and its `temperature_violin.png` cannot be
reused for this target:

1. it conditions only on the laboratory `39Ar` release schedule and contains
   no natural-history temperature or radiogenic `40Ar` observation; and
2. it normalized release fractions by all gas through 1250 C, whereas Shuster
   & Weiss explicitly excluded the four recoil-dominated highest-temperature
   releases. The retained total through 850 C is `163.6722`, not `244.0542`.

Importance reweighting that posterior would preserve the wrong normalization.
The joint posterior therefore requires a fresh run.

## Priors and reported analyses

There is no automatic prior for temperature. The primary analysis will report
the likelihood-informed posterior under a declared bounded temperature prior,
and repeat it under reasonable alternatives. A result whose upper limit moves
materially under those alternatives is prior-sensitive and must be labeled as
such.

For direct comparison with Shuster & Weiss, duration and event timing can be
conditioned on a grid. A separate marginal analysis may use an explicit timing
prior over the last billion years. These are different questions and must not
be mixed in one unlabeled violin.

## Validation gates

The implementation must pass all of these before scientific sampling:

1. no natural diffusion produces `R/R_bulk = 1` and a flat apparent-age
   spectrum at closure age;
2. modal mass is conserved and agrees with the exact spherical release curve
   (the legacy one-mode branch approximation is not used);
3. hotter or longer excursions never increase retained pre-event argon;
4. synthetic data recover temperature, timing, and diffusion parameters within
   calibrated uncertainty;
5. the Shuster--Weiss HRD parameters reproduce their approximately 1% retained
   argon deficit at the corresponding Figure 3 boundary;
6. posterior predictive checks show both `39Ar` release and apparent ages;
7. rank-normalized split R-hat is at most 1.01, bulk and tail ESS exceed 2,000,
   there are zero divergences, and independent seeds agree;
8. temperature-prior, discrepancy, included-step, modal-resolution, and event-
   timing sensitivities do not reverse the stated conclusion without being
   shown explicitly.
