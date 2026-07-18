"""A small, diagnostic-first NUTS fit for the posterior trace animation.

The original comet animation was generated from endpoint snapshots of a sampler
that continued adapting during every "sampling" block.  This script instead:

1. uses an ordered two-domain parameterization (no label switching),
2. adapts each chain once, then freezes the NUTS kernel,
3. retains every post-warmup draw, and
4. writes rank-normalized R-hat, ESS, acceptance, and divergence diagnostics.

This is intentionally a focused two-domain diagnostic fit.  It is not a
replacement for the model-comparison sweep used elsewhere in the project.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/martiandating-jax-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--draws", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--relative-sigma", type=float, default=0.10)
    parser.add_argument("--tag", default="comet_nuts")
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument(
        "--fixed-step-size",
        type=float,
        help="Skip adaptation and use this step size with an identity mass matrix",
    )
    parser.add_argument(
        "--whiten-from",
        type=Path,
        help="Use a previous fit's retained draws as an affine preconditioner",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

AFFINE_MEAN = np.zeros(4)
AFFINE_CHOLESKY = np.eye(4)
WHITENED_INITIAL = None
if ARGS.whiten_from:
    with ARGS.whiten_from.open("rb") as handle:
        whitening_fit = pickle.load(handle)
    pilot_raw = np.asarray(whitening_fit["samples"]["unconstrained"]).reshape(-1, 4)
    AFFINE_MEAN = pilot_raw.mean(axis=0)
    pilot_covariance = np.cov(pilot_raw, rowvar=False)
    AFFINE_CHOLESKY = np.linalg.cholesky(pilot_covariance + np.eye(4) * 1e-8)
    selected = np.linspace(0, len(pilot_raw) - 1, ARGS.chains, dtype=int)
    WHITENED_INITIAL = np.linalg.solve(
        AFFINE_CHOLESKY, (pilot_raw[selected] - AFFINE_MEAN).T
    ).T

AFFINE_MEAN_JAX = jnp.asarray(AFFINE_MEAN)
AFFINE_CHOLESKY_JAX = jnp.asarray(AFFINE_CHOLESKY)


def load_data(relative_sigma: float):
    df = pd.read_csv("data/nakhla1_parsed_fitted.csv")
    total_ar = df["39Ar"].sum()
    df["dF"] = df["39Ar"] / total_ar
    df["T_K"] = df["Temp"] + 273.15
    fit = df.dropna(subset=["seconds_per_extraction_step"])
    fit = fit[fit["Temp"] >= 350].copy()

    observed = fit["dF"].to_numpy(float)
    measured_sigma = fit["std_39Ar"].to_numpy(float) / total_ar
    # The imputed extraction durations dominate the tabulated measurement
    # errors.  RESULTS_SUMMARY.md specifies the 10% model/observation error
    # used for the analysis; combine it with the measurement error here.
    sigma = np.sqrt(measured_sigma**2 + (relative_sigma * observed) ** 2)
    return (
        jnp.asarray(fit["T_K"].to_numpy(float)),
        jnp.asarray(fit["seconds_per_extraction_step"].to_numpy(float)),
        jnp.asarray(observed),
        jnp.asarray(sigma),
    )


TEMPS_K, STEP_SECONDS, F_OBS, SIGMA_OBS = load_data(ARGS.relative_sigma)
R_GAS = 8.314
LOGD_PRIOR_MEAN = 5.7
LOGD_PRIOR_SD = 2.0
EA_PRIOR_MEAN = 117.0
EA_PRIOR_SD = 5.4
REFERENCE_TEMP_K = float(np.median(np.asarray(TEMPS_K)))
LOG_RATE_REF_PRIOR_MEAN = (
    LOGD_PRIOR_MEAN - EA_PRIOR_MEAN * 1e3 / (R_GAS * REFERENCE_TEMP_K)
)


def fractional_release(y):
    """Spherical diffusion solution used by the existing MDD code."""
    short = 6 * jnp.sqrt(y / jnp.pi) - 3 * y
    long = 1 - (6 / jnp.pi**2) * jnp.exp(-jnp.pi**2 * y)
    return jnp.clip(jnp.where(y < 0.3, short, long), 0.0, 1.0)


def base_position(position):
    """Map sampler coordinates to the model's unconstrained coordinates."""
    return AFFINE_MEAN_JAX + AFFINE_CHOLESKY_JAX @ position


def transform(position):
    """Map four unconstrained coordinates to an ordered physical model."""
    position = base_position(position)
    log_rate_ref, raw_gap, phi_logit, ea_z = position
    ea = EA_PRIOR_MEAN + EA_PRIOR_SD * ea_z
    # Sampling the diffusion rate at the middle of the experiment removes the
    # nearly linear Ea/intercept correlation that makes raw Arrhenius
    # intercepts extremely expensive for HMC to traverse.
    center = log_rate_ref + ea * 1e3 / (R_GAS * REFERENCE_TEMP_K)
    gap = 0.10 + jax.nn.softplus(raw_gap)
    logd = jnp.array([center - gap / 2, center + gap / 2])
    phi_0 = jax.nn.sigmoid(phi_logit)
    phi = jnp.array([phi_0, 1 - phi_0])
    return logd, phi, ea


def forward_model(position):
    logd, phi, ea = transform(position)
    diffusion = jnp.exp(logd)[:, None]
    progress = diffusion * jnp.exp(-ea * 1e3 / (R_GAS * TEMPS_K[None, :])) * STEP_SECONDS
    cumulative = jnp.cumsum(progress, axis=1)
    released = fractional_release(cumulative)
    increments = jnp.diff(released, axis=1, prepend=jnp.zeros((2, 1)))
    # F_OBS is an absolute fraction of the complete step-heating release.
    # Do not renormalize the modeled window to one: doing so forces the model
    # to fit observations that sum to ~0.55 with predictions that sum to 1.
    return phi @ increments


def log_density(position):
    logd, phi, _ = transform(position)
    prediction = forward_model(position)
    unconstrained = base_position(position)
    _, raw_gap, _, ea_z = unconstrained
    residual = (F_OBS - prediction) / SIGMA_OBS
    likelihood = jnp.sum(-0.5 * residual**2 - jnp.log(SIGMA_OBS))

    logd_prior = jnp.sum(-0.5 * ((logd - LOGD_PRIOR_MEAN) / LOGD_PRIOR_SD) ** 2)
    ea_prior = -0.5 * ea_z**2
    # Jacobians retain the intended priors: independent Normal priors on the
    # ordered logD values and a uniform Beta(1, 1) prior on the first fraction.
    gap_jacobian = jax.nn.log_sigmoid(raw_gap)
    phi_jacobian = jnp.log(phi[0]) + jnp.log(phi[1])
    return likelihood + logd_prior + ea_prior + gap_jacobian + phi_jacobian


def initial_positions(key, chains: int):
    if WHITENED_INITIAL is not None:
        return jnp.asarray(WHITENED_INITIAL)
    keys = jax.random.split(key, 4)
    centers = LOG_RATE_REF_PRIOR_MEAN + 0.25 * jax.random.normal(keys[0], (chains,))
    raw_gaps = 1.0 + 0.35 * jax.random.normal(keys[1], (chains,))
    phi_logits = 0.6 * jax.random.normal(keys[2], (chains,))
    ea_z = 0.25 * jax.random.normal(keys[3], (chains,))
    return jnp.stack([centers, raw_gaps, phi_logits, ea_z], axis=1)


def warmup_chains(keys, positions):
    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        log_density,
        is_mass_matrix_diagonal=False,
        initial_step_size=0.05,
        target_acceptance_rate=ARGS.target_accept,
        max_num_doublings=10,
    )

    def warm_one(key, position):
        result, _ = adaptation.run(key, position, num_steps=ARGS.warmup)
        return result.state, result.parameters["step_size"], result.parameters["inverse_mass_matrix"]

    return jax.jit(jax.vmap(warm_one))(keys, positions)


def sample_chains(keys, states, step_sizes, inverse_masses):
    def sample_one(key, state, step_size, inverse_mass):
        kernel = blackjax.nuts(
            log_density,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass,
            max_num_doublings=10,
        )
        draw_keys = jax.random.split(key, ARGS.draws)

        def one_step(carry, draw_key):
            next_state, info = kernel.step(draw_key, carry)
            stats = jnp.array([
                info.acceptance_rate,
                info.is_divergent,
                info.num_integration_steps,
            ])
            return next_state, (next_state.position, stats)

        final_state, (positions, stats) = jax.lax.scan(one_step, state, draw_keys)
        return final_state, positions, stats

    return jax.jit(jax.vmap(sample_one))(keys, states, step_sizes, inverse_masses)


def scalar_diagnostics(values: np.ndarray):
    """Diagnostics for an array shaped (draw, chain)."""
    chain_draw = values.T
    return {
        "rhat": float(az.rhat(chain_draw, method="rank")),
        "ess_bulk": float(az.ess(chain_draw, method="bulk")),
        "ess_tail": float(az.ess(chain_draw, method="tail", prob=(0.05, 0.95))),
    }


def main():
    print(f"JAX backend: {jax.default_backend()} ({jax.devices()})")
    print(f"Data: {len(F_OBS)} extraction steps; observed fraction sums to {float(F_OBS.sum()):.3f}")
    print(f"Run: {ARGS.chains} chains, {ARGS.warmup} warmup, {ARGS.draws} retained draws")
    if ARGS.whiten_from:
        print(f"Affine preconditioner: {ARGS.whiten_from}")

    root_key = jax.random.key(ARGS.seed)
    init_key, warm_key, sample_key = jax.random.split(root_key, 3)
    initial = initial_positions(init_key, ARGS.chains)

    if ARGS.fixed_step_size is not None:
        if ARGS.whiten_from is None:
            raise SystemExit("--fixed-step-size requires --whiten-from")
        started = time.time()
        states = jax.jit(jax.vmap(lambda position: blackjax.nuts.init(position, log_density)))(initial)
        jax.block_until_ready(states.position)
        warmup_seconds = time.time() - started
        step_sizes = jnp.full((ARGS.chains,), ARGS.fixed_step_size)
        inverse_masses = jnp.repeat(jnp.eye(4)[None, ...], ARGS.chains, axis=0)
        print(
            f"Using fixed whitened kernel: step size {ARGS.fixed_step_size:g}, "
            "identity mass matrix"
        )
    else:
        started = time.time()
        warm_keys = jax.random.split(warm_key, ARGS.chains)
        states, step_sizes, inverse_masses = warmup_chains(warm_keys, initial)
        jax.block_until_ready(states.position)
        warmup_seconds = time.time() - started
        step_sizes_np = np.asarray(step_sizes)
        # A rare pathological adaptation can produce a near-zero step size and
        # a frozen chain even when the other chains agree. Kernel tuning affects
        # efficiency, not the stationary distribution, so share a representative
        # complete dense tuning across chains after adaptation has stopped.
        median_log_step = np.median(np.log(step_sizes_np))
        tuning_chain = int(np.argmin(np.abs(np.log(step_sizes_np) - median_log_step)))
        shared_step_size = step_sizes[tuning_chain]
        shared_inverse_mass = inverse_masses[tuning_chain]
        step_sizes = jnp.repeat(shared_step_size[None], ARGS.chains, axis=0)
        inverse_masses = jnp.repeat(shared_inverse_mass[None, ...], ARGS.chains, axis=0)
        print(
            f"Warmup finished in {warmup_seconds:.1f}s; individual step sizes "
            f"{step_sizes_np.min():.3g}–{step_sizes_np.max():.3g}; "
            f"using chain {tuning_chain + 1}'s {float(shared_step_size):.3g} shared tuning"
        )

    sample_started = time.time()
    sample_keys = jax.random.split(sample_key, ARGS.chains)
    _, raw_chain_draw, stats_chain_draw = sample_chains(
        sample_keys, states, step_sizes, inverse_masses
    )
    jax.block_until_ready(raw_chain_draw)
    sample_seconds = time.time() - sample_started
    print(f"Sampling finished in {sample_seconds:.1f}s")

    sampler_coordinates = np.asarray(raw_chain_draw).transpose(1, 0, 2)
    flat = jnp.asarray(sampler_coordinates.reshape(-1, sampler_coordinates.shape[-1]))
    raw_flat = jax.vmap(base_position)(flat)
    raw = np.asarray(raw_flat).reshape(ARGS.draws, ARGS.chains, 4)
    logd_flat, phi_flat, ea_flat = jax.vmap(transform)(flat)
    logd = np.asarray(logd_flat).reshape(ARGS.draws, ARGS.chains, 2)
    phi = np.asarray(phi_flat).reshape(ARGS.draws, ARGS.chains, 2)
    ea_shared = np.asarray(ea_flat).reshape(ARGS.draws, ARGS.chains)
    ea = np.repeat(ea_shared[..., None], 2, axis=2)

    diagnostic_values = {
        "logD_HRD": logd[:, :, 0],
        "logD_LRD": logd[:, :, 1],
        "phi_HRD": phi[:, :, 0],
        "Ea_shared": ea_shared,
    }
    diagnostics = {name: scalar_diagnostics(values) for name, values in diagnostic_values.items()}
    max_rhat = max(item["rhat"] for item in diagnostics.values())
    min_bulk_ess = min(item["ess_bulk"] for item in diagnostics.values())
    min_tail_ess = min(item["ess_tail"] for item in diagnostics.values())

    stats = np.asarray(stats_chain_draw).transpose(1, 0, 2)
    acceptance = stats[:, :, 0]
    divergences = int(stats[:, :, 1].sum())
    integration_steps = stats[:, :, 2]
    max_depth_fraction = float(np.mean(integration_steps >= (2**10 - 1)))
    summary = {
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "mean_acceptance": float(acceptance.mean()),
        "divergences": divergences,
        "median_integration_steps": float(np.median(integration_steps)),
        "max_tree_depth_fraction": max_depth_fraction,
        "warmup_seconds": warmup_seconds,
        "sample_seconds": sample_seconds,
        "parameters": diagnostics,
    }

    print("\nDiagnostics")
    for name, item in diagnostics.items():
        print(
            f"  {name:12s} R-hat={item['rhat']:.4f}  "
            f"bulk ESS={item['ess_bulk']:.0f}  tail ESS={item['ess_tail']:.0f}"
        )
    print(
        f"  acceptance={summary['mean_acceptance']:.3f}  "
        f"divergences={divergences}  max-depth={max_depth_fraction:.1%}  "
        f"median leapfrog steps={summary['median_integration_steps']:.0f}"
    )

    output = {
        "samples": {
            "unconstrained": raw,
            "logD0_r2": logd,
            "phi": phi,
            "Ea": ea,
        },
        "sampler_stats": {
            "acceptance_rate": acceptance,
            "is_divergent": stats[:, :, 1].astype(bool),
            "num_integration_steps": integration_steps.astype(int),
        },
        "diagnostics": summary,
        "args": vars(ARGS),
        "model_note": "Ordered two-domain model with shared Ea and 10% relative observation/model error.",
    }
    pkl_path = RESULTS / f"fit_{ARGS.tag}.pkl"
    json_path = RESULTS / f"fit_{ARGS.tag}_diagnostics.json"
    with pkl_path.open("wb") as handle:
        pickle.dump(output, handle)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {pkl_path} and {json_path}")

    passed = (
        max_rhat < 1.01
        and min_bulk_ess >= 400
        and divergences == 0
        and max_depth_fraction < 0.05
    )
    print(f"Publication diagnostic gate: {'PASS' if passed else 'FAIL'}")
    if ARGS.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
