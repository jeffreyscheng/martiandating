"""MAP-preconditioned NUTS pilot for the joint thermochronology target.

This runner is intentionally small and is used for timing, model criticism,
and synthetic-recovery work. Publication runs should retain the checkpointed
kill gates from ``run_flexible_mdd.py`` after this target passes its pilots.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from joint_thermochronology_jax import build_expanded_target, build_joint_target

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-my", type=float, default=100.0)
    parser.add_argument("--end-lookback-my", type=float, default=0.0)
    parser.add_argument(
        "--scenario-grid-json",
        type=Path,
        help=(
            "Marginalize one run over the mutually exclusive duration/start "
            "scenarios in this JSON file"
        ),
    )
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=250)
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
    parser.add_argument("--temperature-min-c", type=float, default=-100.0)
    parser.add_argument("--temperature-max-c", type=float, default=100.0)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--discrepancy-median", type=float, default=0.03)
    parser.add_argument("--discrepancy-log-sd", type=float, default=0.7)
    parser.add_argument("--relative-ar39-sigma", type=float, default=0.10)
    parser.add_argument("--minimum-temperature-c", type=float, default=350.0)
    parser.add_argument("--synthetic-temperature-c", type=float)
    parser.add_argument("--tag", default="joint_pilot")
    parser.add_argument("--results-dir", type=Path, default=Path("results/joint"))
    parser.add_argument("--kill-gates", action="store_true")
    return parser.parse_args()


def scalar_diagnostics(values: np.ndarray) -> dict[str, float]:
    return {
        "rhat": float(az.rhat(values.T, method="rank")),
        "ess_bulk": float(az.ess(values.T, method="bulk")),
        "ess_tail": float(az.ess(values.T, method="tail", prob=(0.05, 0.95))),
    }


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, ordered_values)


def main() -> None:
    args = parse_args()
    output = args.results_dir / args.tag
    output.mkdir(parents=True, exist_ok=True)
    common_target_args = dict(
        bins=args.bins,
        mode_count=args.mode_count,
        quadrature_count=args.quadrature_count,
        temperature_min_c=args.temperature_min_c,
        temperature_max_c=args.temperature_max_c,
        discrepancy_median=args.discrepancy_median,
        discrepancy_log_sd=args.discrepancy_log_sd,
        relative_ar39_sigma=args.relative_ar39_sigma,
        minimum_temperature_c=args.minimum_temperature_c,
    )
    scenario_responsibility = None
    scenarios = None
    if args.scenario_grid_json is None:
        target_args = common_target_args | {
            "event_duration_my": args.duration_my,
            "event_end_lookback_my": args.end_lookback_my,
        }
        data, metadata, transform, forward, native_log_density = (
            build_joint_target(**target_args)
        )
    else:
        scenarios = json.loads(args.scenario_grid_json.read_text())
        if not isinstance(scenarios, list):
            raise SystemExit("scenario grid JSON must contain a list")
        target_args = common_target_args | {"scenarios": scenarios}
        (
            data,
            metadata,
            transform,
            forward,
            scenario_responsibility,
            native_log_density,
        ) = build_expanded_target(**target_args)
    synthetic_truth = None
    if args.synthetic_temperature_c is not None:
        if scenario_responsibility is not None:
            raise SystemExit(
                "synthetic recovery is not yet implemented for an expanded grid"
            )
        unit_temperature = (
            args.synthetic_temperature_c - args.temperature_min_c
        ) / (args.temperature_max_c - args.temperature_min_c)
        if not 0 < unit_temperature < 1:
            raise SystemExit("synthetic temperature must lie strictly inside bounds")
        truth_position = np.zeros(metadata["dimension"])
        truth_position[args.bins] = np.log(
            unit_temperature / (1.0 - unit_temperature)
        )
        true_ar39, true_ratio, true_discrepancy = forward(jnp.asarray(truth_position))
        rng = np.random.default_rng(args.seed + 900_000)
        synthetic_ar39 = np.asarray(true_ar39) + rng.normal(
            size=len(data.temperatures_c)
        ) * data.ar39_sigma
        ratio_scale = np.sqrt(
            data.normalized_ratio_sigma**2 + float(true_discrepancy) ** 2
        )
        synthetic_ratio = np.asarray(true_ratio) + rng.standard_t(
            4, size=len(data.temperatures_c)
        ) * ratio_scale
        data = replace(
            data,
            observed_ar39_fraction=synthetic_ar39,
            normalized_ratio=synthetic_ratio,
        )
        data, metadata, transform, forward, native_log_density = (
            build_joint_target(**target_args, data_override=data)
        )
        true_weights, true_ea, true_temperature, _ = transform(
            jnp.asarray(truth_position)
        )
        synthetic_truth = {
            "temperature_c": float(true_temperature),
            "ea_kj_mol": float(true_ea),
            "discrepancy": float(true_discrepancy),
            "weights": np.asarray(true_weights).tolist(),
        }
    dimension = metadata["dimension"]
    responsibility_batch_fn = (
        None
        if scenario_responsibility is None
        else jax.jit(jax.vmap(scenario_responsibility))
    )
    devices = jax.devices()
    device_count = len(devices)
    if args.chains % device_count:
        raise SystemExit(
            f"--chains ({args.chains}) must be divisible by the JAX device "
            f"count ({device_count})"
        )
    if args.draws % args.chunk_size:
        raise SystemExit("--draws must be divisible by --chunk-size")
    chains_per_device = args.chains // device_count
    print(
        f"JAX backend: {jax.default_backend()} on {device_count} devices; "
        f"{chains_per_device} chains/device",
        flush=True,
    )
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
    np.savez_compressed(
        output / "preconditioner.npz",
        mode=np.asarray(mode),
        covariance_root=np.asarray(root),
        hessian=np.asarray(hessian),
        hessian_eigenvalues=eigenvalues,
    )
    (output / "run_args.json").write_text(
        json.dumps(
            vars(args) | {"results_dir": str(args.results_dir)},
            indent=2,
            default=json_default,
        )
        + "\n"
    )

    def to_native(position):
        return mode + root @ position

    def log_density(position):
        return native_log_density(to_native(position))

    key = jax.random.key(args.seed)
    init_key, warm_key, draw_key = jax.random.split(key, 3)
    initial = args.init_scale * jax.random.normal(
        init_key, (device_count, chains_per_device, dimension)
    )
    if args.fixed_step_size is not None:
        init_fn = jax.pmap(
            jax.vmap(lambda position: blackjax.nuts.init(position, log_density))
        )
        states = init_fn(initial)
        step_sizes = jnp.full(
            (device_count, chains_per_device), args.fixed_step_size
        )
        inverse_masses = jnp.broadcast_to(
            jnp.eye(dimension),
            (device_count, chains_per_device, dimension, dimension),
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

        warm_keys = jax.random.split(warm_key, args.chains).reshape(
            device_count, chains_per_device
        )
        states, step_sizes, inverse_masses = jax.pmap(jax.vmap(warm_one))(
            warm_keys, initial
        )
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
        keys = jax.random.split(chain_key, args.chunk_size)

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

        final_state, (positions, stats) = jax.lax.scan(one_step, state, keys)
        return final_state, positions, stats

    sample_fn = jax.pmap(jax.vmap(sample_one))
    sample_started = time.time()
    affine_parts = []
    stats_parts = []
    responsibility_parts = []
    chunk_dir = output / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    for chunk_index in range(args.draws // args.chunk_size):
        chunk_key = jax.random.fold_in(draw_key, chunk_index)
        draw_keys = jax.random.split(chunk_key, args.chains).reshape(
            device_count, chains_per_device
        )
        states, affine_device, stats_device = sample_fn(
            draw_keys, states, step_sizes, inverse_masses
        )
        jax.block_until_ready(states.position)
        affine_chunk = (
            np.asarray(affine_device)
            .transpose(2, 0, 1, 3)
            .reshape(args.chunk_size, args.chains, dimension)
        )
        stats_chunk = (
            np.asarray(stats_device)
            .transpose(2, 0, 1, 3)
            .reshape(args.chunk_size, args.chains, 4)
        )
        affine_parts.append(affine_chunk)
        stats_parts.append(stats_chunk)
        if responsibility_batch_fn is not None:
            native_chunk = np.einsum(
                "tcd,ed->tce", affine_chunk, np.asarray(root)
            ) + np.asarray(mode)
            flat_native_chunk = native_chunk.reshape(-1, dimension)
            responsibility_chunk = np.asarray(
                responsibility_batch_fn(jnp.asarray(flat_native_chunk))
            ).reshape(args.chunk_size, args.chains, len(scenarios))
            responsibility_parts.append(responsibility_chunk)
        np.savez_compressed(
            chunk_dir / f"chunk_{chunk_index + 1:04d}.npz",
            affine_position=affine_chunk,
            sampler_stats=stats_chunk,
        )
        elapsed = time.time() - sample_started
        completed = (chunk_index + 1) * args.chunk_size
        cumulative_affine = np.concatenate(affine_parts, axis=0)
        cumulative_stats = np.concatenate(stats_parts, axis=0)
        cumulative_position = np.einsum(
            "tcd,ed->tce", cumulative_affine, np.asarray(root)
        ) + np.asarray(mode)
        temperature = args.temperature_min_c + (
            args.temperature_max_c - args.temperature_min_c
        ) / (1.0 + np.exp(-cumulative_position[:, :, args.bins]))
        diagnostics = {
            "draws": completed,
            "seconds": elapsed,
            "chain_draws_per_second": completed * args.chains / elapsed,
            "mean_acceptance": float(np.mean(cumulative_stats[:, :, 0])),
            "divergences": int(np.sum(cumulative_stats[:, :, 1])),
            "divergence_fraction": float(np.mean(cumulative_stats[:, :, 1])),
            "max_depth_fraction": float(
                np.mean(
                    cumulative_stats[:, :, 2]
                    >= 2**args.max_doublings - 1
                )
            ),
            "temperature": scalar_diagnostics(temperature),
        }
        if responsibility_parts:
            cumulative_responsibility = np.concatenate(
                responsibility_parts, axis=0
            )
            flat_responsibility = cumulative_responsibility.reshape(
                -1, len(scenarios)
            )
            responsibility_ess = (
                np.sum(flat_responsibility, axis=0) ** 2
                / np.sum(flat_responsibility**2, axis=0)
            )
            chain_mass = np.mean(cumulative_responsibility, axis=0)
            mean_mass = np.mean(chain_mass, axis=0)
            relative_chain_spread = np.max(
                np.abs(chain_mass - mean_mass[None, :])
                / np.maximum(mean_mass[None, :], 1e-12),
                axis=0,
            )
            diagnostics["expanded_ensemble"] = {
                "minimum_responsibility_ess": float(
                    np.min(responsibility_ess)
                ),
                "minimum_scenario_mass": float(np.min(mean_mass)),
                "maximum_relative_chain_mass_deviation": float(
                    np.max(relative_chain_spread)
                ),
            }
        with (output / "diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(diagnostics) + "\n")
        (output / "latest_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
        print(
            f"Checkpoint {completed}/{args.draws}: {elapsed:.1f}s; "
            f"{completed * args.chains / elapsed:.1f} chain-draws/s; "
            f"T Rhat={diagnostics['temperature']['rhat']:.4f}; "
            f"div={diagnostics['divergences']}; "
            f"depth={diagnostics['max_depth_fraction']:.2%}",
            flush=True,
        )
        egregious = (
            not np.isfinite(cumulative_position).all()
            or diagnostics["divergence_fraction"] > 0.01
            or diagnostics["max_depth_fraction"] > 0.10
        )
        if responsibility_parts and completed >= 500:
            expanded = diagnostics["expanded_ensemble"]
            egregious = egregious or (
                expanded["minimum_scenario_mass"] < 1e-5
                or expanded["maximum_relative_chain_mass_deviation"] > 2.0
            )
        if args.kill_gates and completed >= 500 and egregious:
            (output / "killed.json").write_text(
                json.dumps(diagnostics, indent=2) + "\n"
            )
            raise SystemExit("egregious sampling diagnostic triggered kill gate")
    sampling_seconds = time.time() - sample_started
    print(f"Sampling: {sampling_seconds:.1f}s", flush=True)
    affine_positions = np.concatenate(affine_parts, axis=0)
    positions = np.einsum("tcd,ed->tce", affine_positions, np.asarray(root))
    positions += np.asarray(mode)
    stats = np.concatenate(stats_parts, axis=0)

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
    responsibilities = None
    scenario_summary = None
    if responsibility_parts:
        responsibilities = np.concatenate(responsibility_parts, axis=0)
        flat_temperature = physical["temperature_c"].reshape(-1)
        flat_responsibility = responsibilities.reshape(-1, len(scenarios))
        scenario_summary = []
        for scenario, scenario_weights in zip(
            metadata["scenarios"], flat_responsibility.T
        ):
            weight_sum = float(np.sum(scenario_weights))
            responsibility_ess = float(
                weight_sum**2 / np.sum(scenario_weights**2)
            )
            quantiles = weighted_quantile(
                flat_temperature,
                scenario_weights,
                np.asarray([0.025, 0.5, 0.975]),
            )
            scenario_summary.append(
                scenario
                | {
                    "posterior_mixture_mass": float(
                        np.mean(scenario_weights)
                    ),
                    "responsibility_ess": responsibility_ess,
                    "temperature_c": {
                        "central_95_and_median": quantiles.tolist(),
                        "probability_above_zero": float(
                            np.sum(
                                scenario_weights
                                * (flat_temperature > 0)
                            )
                            / weight_sum
                        ),
                    },
                }
            )
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
        "devices": [str(device) for device in devices],
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
    if scenario_summary is not None:
        chain_mass = np.mean(responsibilities, axis=0)
        mean_mass = np.mean(chain_mass, axis=0)
        summary["expanded_ensemble"] = {
            "method": (
                "NUTS over a discrete-scenario-marginalized mixture; "
                "cell conditionals use Rao-Blackwell responsibilities"
            ),
            "success_criteria": {
                "maximum_rhat": 1.01,
                "maximum_divergence_fraction": 0.001,
                "maximum_depth_fraction": 0.01,
                "minimum_responsibility_ess_per_cell": 1000,
                "maximum_relative_chain_mass_deviation": 0.25,
            },
            "minimum_responsibility_ess": min(
                item["responsibility_ess"] for item in scenario_summary
            ),
            "maximum_relative_chain_mass_deviation": float(
                np.max(
                    np.abs(chain_mass - mean_mass[None, :])
                    / np.maximum(mean_mass[None, :], 1e-12)
                )
            ),
            "scenarios": scenario_summary,
        }
    if synthetic_truth is not None:
        summary["synthetic_truth"] = synthetic_truth
        summary["synthetic_recovery"] = {
            key: {
                "central_95_and_median": np.quantile(
                    physical[key], [0.025, 0.5, 0.975]
                ).tolist(),
                "truth_inside_central_95": bool(
                    np.quantile(physical[key], 0.025)
                    <= synthetic_truth[key]
                    <= np.quantile(physical[key], 0.975)
                ),
            }
            for key in ("temperature_c", "ea_kj_mol", "discrepancy")
        }
        weight_low, weight_median, weight_high = np.quantile(
            physical["weights"], [0.025, 0.5, 0.975], axis=(0, 1)
        )
        true_weights = np.asarray(synthetic_truth["weights"])
        summary["synthetic_recovery"]["weights"] = {
            "bins_with_truth_inside_central_95": int(
                np.sum((true_weights >= weight_low) & (true_weights <= weight_high))
            ),
            "bins": int(args.bins),
            "median_l1_error": float(np.sum(np.abs(weight_median - true_weights))),
        }
    posterior_payload = {
        "position": positions,
        "sampler_stats": stats,
        **physical,
    }
    if responsibilities is not None:
        posterior_payload["scenario_responsibility"] = responsibilities
    np.savez_compressed(output / "posterior.npz", **posterior_payload)
    serialized = json.dumps(summary, indent=2, default=json_default)
    (output / "summary.json").write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
