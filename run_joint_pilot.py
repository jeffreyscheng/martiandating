"""MAP-preconditioned NUTS pilot for the joint thermochronology target.

This runner is intentionally small and is used for timing, model criticism,
and synthetic-recovery work. Publication runs should retain the checkpointed
kill gates from ``run_flexible_mdd.py`` after this target passes its pilots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/martiandating-jax-cache")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/martiandating-cache")

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

from joint_thermochronology_jax import build_joint_target

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-my", type=float, default=100.0)
    parser.add_argument("--end-lookback-my", type=float, default=0.0)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument(
        "--fixed-step-size",
        type=float,
        help="Skip window adaptation and use this step size in affine coordinates",
    )
    parser.add_argument("--max-doublings", type=int, default=10)
    parser.add_argument("--map-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bins", type=int, default=28)
    parser.add_argument("--mode-count", type=int, default=96)
    parser.add_argument("--quadrature-count", type=int, default=32)
    parser.add_argument("--tag", default="joint_pilot")
    parser.add_argument("--results-dir", type=Path, default=Path("results/joint"))
    return parser.parse_args()


def scalar_diagnostics(values: np.ndarray) -> dict[str, float]:
    return {
        "rhat": float(az.rhat(values.T, method="rank")),
        "ess_bulk": float(az.ess(values.T, method="bulk")),
        "ess_tail": float(az.ess(values.T, method="tail", prob=(0.05, 0.95))),
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    output = args.results_dir / args.tag
    output.mkdir(parents=True, exist_ok=True)
    data, metadata, transform, forward, native_log_density = build_joint_target(
        bins=args.bins,
        event_duration_my=args.duration_my,
        event_end_lookback_my=args.end_lookback_my,
        mode_count=args.mode_count,
        quadrature_count=args.quadrature_count,
    )
    dimension = metadata["dimension"]
    objective = lambda position: -native_log_density(position)
    started = time.time()
    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=args.map_iterations,
        tol=1e-8,
        linesearch="zoom",
        maxls=50,
    )
    map_result = solver.run(jnp.zeros(dimension))
    print(
        f"MAP: iteration={int(map_result.state.iter_num)} "
        f"objective={float(map_result.state.value):.6f} "
        f"error={float(map_result.state.error):.6g}",
        flush=True,
    )
    mode = map_result.params
    hessian = jax.hessian(objective)(mode)
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(hessian))
    floor = max(float(eigenvalues.max()) * 1e-7, 1e-7)
    regularized = np.maximum(eigenvalues, floor)
    root = jnp.asarray(eigenvectors @ np.diag(1.0 / np.sqrt(regularized)))
    print(
        f"Hessian: min={float(eigenvalues.min()):.6g} "
        f"max={float(eigenvalues.max()):.6g} floor={floor:.6g}",
        flush=True,
    )

    def to_native(position):
        return mode + root @ position

    def log_density(position):
        return native_log_density(to_native(position))

    key = jax.random.key(args.seed)
    init_key, warm_key, draw_key = jax.random.split(key, 3)
    initial = 0.1 * jax.random.normal(init_key, (args.chains, dimension))
    if args.fixed_step_size is not None:
        states = jax.vmap(lambda position: blackjax.nuts.init(position, log_density))(
            initial
        )
        step_sizes = jnp.full((args.chains,), args.fixed_step_size)
        inverse_masses = jnp.broadcast_to(
            jnp.eye(dimension), (args.chains, dimension, dimension)
        )
    else:
        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            log_density,
            is_mass_matrix_diagonal=False,
            initial_step_size=0.05,
            target_acceptance_rate=args.target_accept,
            max_num_doublings=args.max_doublings,
            progress_bar=False,
        )

        def warm_one(chain_key, position):
            result, _ = adaptation.run(chain_key, position, num_steps=args.warmup)
            return (
                result.state,
                result.parameters["step_size"],
                result.parameters["inverse_mass_matrix"],
            )

        warm_keys = jax.random.split(warm_key, args.chains)
        states, step_sizes, inverse_masses = jax.vmap(warm_one)(warm_keys, initial)
    jax.block_until_ready(states.position)
    warmup_seconds = time.time() - started
    print(
        f"Warmup: {warmup_seconds:.1f}s; step size median "
        f"{float(jnp.median(step_sizes)):.6g}",
        flush=True,
    )

    def sample_one(chain_key, state, step_size, inverse_mass):
        kernel = blackjax.nuts(
            log_density,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass,
            max_num_doublings=args.max_doublings,
        )
        keys = jax.random.split(chain_key, args.draws)

        def one_step(carry, step_key):
            next_state, info = kernel.step(step_key, carry)
            stats = jnp.asarray(
                [
                    info.acceptance_rate,
                    info.is_divergent,
                    info.num_integration_steps,
                    info.energy,
                ]
            )
            return next_state, (next_state.position, stats)

        return jax.lax.scan(one_step, state, keys)[1]

    draw_keys = jax.random.split(draw_key, args.chains)
    sample_started = time.time()
    affine_positions, stats = jax.vmap(sample_one)(
        draw_keys, states, step_sizes, inverse_masses
    )
    jax.block_until_ready(affine_positions)
    sampling_seconds = time.time() - sample_started
    print(f"Sampling: {sampling_seconds:.1f}s", flush=True)
    affine_positions = np.asarray(affine_positions).transpose(1, 0, 2)
    positions = np.einsum("tcd,ed->tce", affine_positions, np.asarray(root))
    positions += np.asarray(mode)
    stats = np.asarray(stats).transpose(1, 0, 2)

    flat_positions = positions.reshape(-1, dimension)
    weights, ea, temperature, discrepancy = jax.vmap(transform)(
        jnp.asarray(flat_positions)
    )
    shape = positions.shape[:2]
    physical = {
        "temperature_c": np.asarray(temperature).reshape(shape),
        "ea_kj_mol": np.asarray(ea).reshape(shape),
        "discrepancy": np.asarray(discrepancy).reshape(shape),
        "weights": np.asarray(weights).reshape(*shape, args.bins),
    }
    diagnostic_values = {
        "temperature_c": physical["temperature_c"],
        "ea_kj_mol": physical["ea_kj_mol"],
        "discrepancy": physical["discrepancy"],
        **{
            f"latent_{index:02d}": positions[:, :, index]
            for index in range(dimension)
        },
    }
    per_parameter = {
        name: scalar_diagnostics(value)
        for name, value in diagnostic_values.items()
    }
    summary = {
        "metadata": metadata,
        "args": vars(args) | {"results_dir": str(args.results_dir)},
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "map": {
            "iterations": int(map_result.state.iter_num),
            "objective": float(map_result.state.value),
            "gradient_norm": float(map_result.state.error),
            "hessian_min_eigenvalue": float(eigenvalues.min()),
            "hessian_floor": floor,
        },
        "warmup_seconds": warmup_seconds,
        "sampling_seconds": sampling_seconds,
        "mean_acceptance": float(np.mean(stats[:, :, 0])),
        "divergences": int(np.sum(stats[:, :, 1])),
        "max_depth_fraction": float(
            np.mean(stats[:, :, 2] >= 2**args.max_doublings - 1)
        ),
        "max_rhat": max(x["rhat"] for x in per_parameter.values()),
        "minimum_bulk_ess": min(x["ess_bulk"] for x in per_parameter.values()),
        "minimum_tail_ess": min(x["ess_tail"] for x in per_parameter.values()),
        "temperature_c": {
            "central_95_and_median": np.quantile(
                physical["temperature_c"], [0.025, 0.5, 0.975]
            ).tolist(),
            "probability_above_zero": float(
                np.mean(physical["temperature_c"] > 0)
            ),
        },
        "parameters": per_parameter,
    }
    np.savez_compressed(
        output / "posterior.npz",
        position=positions,
        sampler_stats=stats,
        **physical,
    )
    serialized = json.dumps(summary, indent=2, default=json_default)
    (output / "summary.json").write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
