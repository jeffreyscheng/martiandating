"""Diagnostic-first multi-GPU NUTS for a flexible MDD diffusion distribution.

The diffusion-scale distribution is represented on a fixed grid.  A smooth,
whitened logistic-normal field supplies positive weights without mixture label
switching.  Every timed laboratory step is propagated through the diffusion
model, while the likelihood can mask recoil-contaminated low-temperature
observations without renormalizing the retained observation window.

Warmup happens once.  The resulting NUTS kernels are frozen, every subsequent
draw is retained, sampler statistics are checkpointed, and pre-registered kill
gates are evaluated after each chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/martiandating-jax-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)


R_GAS = 8.314
EA_MEAN = 117.0
EA_SD = 5.4
LOGD_BASE_MEAN = 5.7
LOGD_BASE_SD = 2.0


@dataclass(frozen=True)
class DataBundle:
    temperatures_c: np.ndarray
    temperatures_k: np.ndarray
    durations_s: np.ndarray
    observed_fraction: np.ndarray
    sigma: np.ndarray
    likelihood_mask: np.ndarray
    total_ar: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument(
        "--settle-draws",
        type=int,
        default=0,
        help="Discard this many frozen-kernel draws before retaining samples",
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--grid-min", type=float, default=0.0)
    parser.add_argument("--grid-max", type=float, default=12.0)
    parser.add_argument("--kernel-length", type=float, default=1.0)
    parser.add_argument("--flex-scale", type=float, default=1.5)
    parser.add_argument("--relative-sigma", type=float, default=0.10)
    parser.add_argument("--minimum-temperature", type=float, default=350.0)
    parser.add_argument(
        "--normalization-maximum-temperature",
        type=float,
        help="Normalize 39Ar by releases at or below this temperature",
    )
    parser.add_argument(
        "--exact-spherical-release",
        action="store_true",
        help="Use the converged spherical eigenmode release function",
    )
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--max-doublings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--tag", default="flex20_primary")
    parser.add_argument("--results-dir", type=Path, default=Path("results/flexible"))
    parser.add_argument("--diagonal-mass", action="store_true")
    parser.add_argument("--affine-precondition", action="store_true")
    parser.add_argument("--map-maxiter", type=int, default=1000)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument(
        "--fixed-step-size",
        type=float,
        help="Skip window adaptation and use this step size in affine coordinates",
    )
    parser.add_argument("--kill-gates", action="store_true")
    parser.add_argument("--strict-final", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_data(
    relative_sigma: float,
    minimum_temperature: float,
    normalization_maximum_temperature: float | None = None,
) -> DataBundle:
    path = Path("data/nakhla1_parsed_fitted.csv")
    frame = pd.read_csv(path)
    normalization_mask = (
        np.ones(len(frame), dtype=bool)
        if normalization_maximum_temperature is None
        else frame["Temp"].to_numpy(float)
        <= normalization_maximum_temperature
    )
    total_ar = float(frame.loc[normalization_mask, "39Ar"].sum())
    timed = frame.dropna(subset=["seconds_per_extraction_step"]).copy()
    observed = timed["39Ar"].to_numpy(float) / total_ar
    measured_sigma = timed["std_39Ar"].to_numpy(float) / total_ar
    sigma = np.sqrt(measured_sigma**2 + (relative_sigma * observed) ** 2)
    temperatures_c = timed["Temp"].to_numpy(float)
    return DataBundle(
        temperatures_c=temperatures_c,
        temperatures_k=temperatures_c + 273.15,
        durations_s=timed["seconds_per_extraction_step"].to_numpy(float),
        observed_fraction=observed,
        sigma=sigma,
        likelihood_mask=temperatures_c >= minimum_temperature,
        total_ar=total_ar,
    )


def smooth_basis(grid: np.ndarray, kernel_length: float) -> np.ndarray:
    """Return a centered, normalized RBF basis with no constant direction."""
    distances = grid[:, None] - grid[None, :]
    kernel = np.exp(-0.5 * (distances / kernel_length) ** 2)
    projector = np.eye(len(grid)) - np.ones((len(grid), len(grid))) / len(grid)
    centered = projector @ kernel @ projector
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    keep = eigenvalues > max(eigenvalues.max() * 1e-10, 1e-12)
    basis = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    marginal_sd = np.sqrt(np.mean(np.sum(basis**2, axis=1)))
    return basis / marginal_sd


def fractional_release(progress: jax.Array) -> jax.Array:
    short = 6 * jnp.sqrt(jnp.maximum(progress, 0) / jnp.pi) - 3 * progress
    long = 1 - (6 / jnp.pi**2) * jnp.exp(-jnp.pi**2 * progress)
    return jnp.clip(jnp.where(progress < 0.3, short, long), 0.0, 1.0)


def exact_fractional_release(
    progress: jax.Array, mode_count: int = 64
) -> jax.Array:
    progress = jnp.asarray(progress)
    safe_progress = jnp.maximum(progress, 1e-300)
    short = 6.0 * jnp.sqrt(safe_progress / jnp.pi) - 3.0 * progress
    n = jnp.arange(1, mode_count + 1, dtype=progress.dtype)
    mode_weights = 6.0 / (jnp.pi**2 * n**2)
    eigenvalues = jnp.pi**2 * n**2
    long = 1.0 - jnp.sum(
        mode_weights * jnp.exp(-progress[..., None] * eigenvalues), axis=-1
    )
    return jnp.clip(jnp.where(progress < 0.05, short, long), 0.0, 1.0)


def build_target(args: argparse.Namespace, data: DataBundle):
    grid_np = np.linspace(args.grid_min, args.grid_max, args.bins)
    basis_np = smooth_basis(grid_np, args.kernel_length)
    base_logits_np = -0.5 * ((grid_np - LOGD_BASE_MEAN) / LOGD_BASE_SD) ** 2

    grid = jnp.asarray(grid_np)
    basis = jnp.asarray(basis_np)
    base_logits = jnp.asarray(base_logits_np)
    temperatures_k = jnp.asarray(data.temperatures_k)
    durations_s = jnp.asarray(data.durations_s)
    observed = jnp.asarray(data.observed_fraction[data.likelihood_mask])
    sigma = jnp.asarray(data.sigma[data.likelihood_mask])
    mask = jnp.asarray(data.likelihood_mask)

    def transform(position: jax.Array):
        latent = position[:-1]
        ea = EA_MEAN + EA_SD * position[-1]
        logits = base_logits + args.flex_scale * (basis @ latent)
        weights = jax.nn.softmax(logits)
        return weights, ea

    def forward_all(position: jax.Array):
        weights, ea = transform(position)
        rates = jnp.exp(grid[:, None] - ea * 1e3 / (R_GAS * temperatures_k[None, :]))
        progress = rates * durations_s[None, :]
        cumulative = jnp.cumsum(progress, axis=1)
        release_function = (
            exact_fractional_release
            if getattr(args, "exact_spherical_release", False)
            else fractional_release
        )
        release = release_function(cumulative)
        increments = jnp.diff(release, axis=1, prepend=jnp.zeros((args.bins, 1)))
        return weights @ increments

    def log_density(position: jax.Array):
        prediction = forward_all(position)[mask]
        residual = (observed - prediction) / sigma
        likelihood = jnp.sum(
            -0.5 * residual**2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)
        )
        prior = -0.5 * jnp.sum(position**2)
        return prior + likelihood

    return grid_np, basis_np, base_logits_np, transform, forward_all, log_density


def numpy_transform(
    positions: np.ndarray,
    grid: np.ndarray,
    basis: np.ndarray,
    base_logits: np.ndarray,
    flex_scale: float,
) -> dict[str, np.ndarray]:
    flat = positions.reshape(-1, positions.shape[-1])
    logits = base_logits[None, :] + flex_scale * (flat[:, :-1] @ basis.T)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    ea = EA_MEAN + EA_SD * flat[:, -1]
    mean = weights @ grid
    variance = np.sum(weights * (grid[None, :] - mean[:, None]) ** 2, axis=1)
    cdf = np.cumsum(weights, axis=1)
    quantiles = {
        f"logD_q{int(prob * 100):02d}": grid[np.argmax(cdf >= prob, axis=1)]
        for prob in (0.10, 0.50, 0.90)
    }
    shape = positions.shape[:2]
    return {
        "weights": weights.reshape(*shape, len(grid)),
        "Ea_shared": ea.reshape(shape),
        "logD_mean": mean.reshape(shape),
        "logD_sd": np.sqrt(variance).reshape(shape),
        "edge_mass": (weights[:, 0] + weights[:, -1]).reshape(shape),
        **{name: values.reshape(shape) for name, values in quantiles.items()},
    }


def scalar_diagnostics(values: np.ndarray) -> dict[str, float]:
    chain_draw = np.asarray(values).T
    return {
        "rhat": float(az.rhat(chain_draw, method="rank")),
        "ess_bulk": float(az.ess(chain_draw, method="bulk")),
        "ess_tail": float(az.ess(chain_draw, method="tail", prob=(0.05, 0.95))),
    }


def diagnostics(
    positions: np.ndarray,
    stats: np.ndarray,
    grid: np.ndarray,
    basis: np.ndarray,
    base_logits: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    physical = numpy_transform(
        positions, grid, basis, base_logits, args.flex_scale
    )
    parameter_values: dict[str, np.ndarray] = {
        **{f"latent_{index:02d}": positions[:, :, index] for index in range(positions.shape[-1] - 1)},
        "Ea_shared": physical["Ea_shared"],
        "logD_mean": physical["logD_mean"],
        "logD_sd": physical["logD_sd"],
        "edge_mass": physical["edge_mass"],
    }
    per_parameter = {
        name: scalar_diagnostics(values) for name, values in parameter_values.items()
    }
    max_rhat = max(item["rhat"] for item in per_parameter.values())
    min_bulk = min(item["ess_bulk"] for item in per_parameter.values())
    min_tail = min(item["ess_tail"] for item in per_parameter.values())

    acceptance = stats[:, :, 0]
    divergent = stats[:, :, 1].astype(bool)
    integration_steps = stats[:, :, 2]
    energy = stats[:, :, 3]
    energy_variance = np.var(energy, axis=0, ddof=1)
    bfmi = np.mean(np.diff(energy, axis=0) ** 2, axis=0) / np.maximum(
        energy_variance, 1e-30
    )
    chain_std = np.std(positions, axis=0)
    stuck = np.all(chain_std < 1e-12, axis=1)
    max_steps = 2**args.max_doublings - 1

    return {
        "draws": int(positions.shape[0]),
        "chains": int(positions.shape[1]),
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk,
        "min_tail_ess": min_tail,
        "mean_acceptance": float(np.mean(acceptance)),
        "divergences": int(np.sum(divergent)),
        "divergence_fraction": float(np.mean(divergent)),
        "max_depth_fraction": float(np.mean(integration_steps >= max_steps)),
        "median_integration_steps": float(np.median(integration_steps)),
        "min_bfmi": float(np.min(bfmi)),
        "bfmi_below_0_3_fraction": float(np.mean(bfmi < 0.3)),
        "stuck_chains": int(np.sum(stuck)),
        "mean_edge_mass": float(np.mean(physical["edge_mass"])),
        "max_chain_mean_edge_mass": float(
            np.max(np.mean(physical["edge_mass"], axis=0))
        ),
        "parameters": per_parameter,
    }


def kill_reasons(current: dict[str, Any], previous: dict[str, Any] | None) -> list[str]:
    reasons = []
    draws = current["draws"]
    if current["stuck_chains"] > 0.01 * current["chains"]:
        reasons.append("more than 1% of chains are stuck")
    if draws >= 250 and current["divergence_fraction"] > 0.02:
        reasons.append("divergences exceed 2% after 250 draws")
    if draws >= 250 and current["max_depth_fraction"] > 0.10:
        reasons.append("maximum-depth transitions exceed 10% after 250 draws")
    if draws >= 500 and current["max_rhat"] > 1.20:
        reasons.append("max rank R-hat exceeds 1.20 after 500 draws")
    if (
        draws == 500
        and previous is not None
        and previous["max_rhat"] > 1.05
        and previous["max_rhat"] - current["max_rhat"] < 0.02
    ):
        reasons.append("R-hat failed to improve materially from 250 to 500 draws")
    if draws >= 1000 and current["max_rhat"] > 1.05:
        reasons.append("max rank R-hat exceeds 1.05 after 1000 draws")
    if draws >= 1000 and current["divergence_fraction"] > 0.005:
        reasons.append("divergences exceed 0.5% after 1000 draws")
    if draws >= 1000 and current["min_bulk_ess"] / draws < 0.02:
        reasons.append("minimum bulk ESS per draw is below 0.02 after 1000 draws")
    if current["max_chain_mean_edge_mass"] > 0.10:
        reasons.append("at least one chain puts over 10% mean mass on grid edges")
    return reasons


def final_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if summary["max_rhat"] > 1.01:
        failures.append("max R-hat > 1.01")
    if summary["min_bulk_ess"] < 2000:
        failures.append("minimum bulk ESS < 2000")
    if summary["min_tail_ess"] < 2000:
        failures.append("minimum tail ESS < 2000")
    if summary["divergences"] != 0:
        failures.append("nonzero divergences")
    if summary["max_depth_fraction"] >= 0.01:
        failures.append("maximum-depth fraction >= 1%")
    if summary["min_bfmi"] <= 0.30:
        failures.append("minimum chain E-BFMI <= 0.30")
    if summary["mean_edge_mass"] >= 0.005:
        failures.append("mean edge mass >= 0.5%")
    return not failures, failures


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    data: DataBundle,
    grid: np.ndarray,
    basis: np.ndarray,
) -> dict[str, Any]:
    data_path = Path("data/nakhla1_parsed_fitted.csv")
    devices = [str(device) for device in jax.devices()]
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "script_sha256": sha256(Path(__file__)),
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "blackjax": blackjax.__version__,
        "arviz": az.__version__,
        "backend": jax.default_backend(),
        "devices": devices,
        "timed_temperatures_c": data.temperatures_c.tolist(),
        "likelihood_temperatures_c": data.temperatures_c[data.likelihood_mask].tolist(),
        "observed_full_timed_fraction_sum": float(data.observed_fraction.sum()),
        "observed_likelihood_fraction_sum": float(data.observed_fraction[data.likelihood_mask].sum()),
        "grid": grid.tolist(),
        "basis_shape": list(basis.shape),
        "success_criteria": {
            "max_rank_folded_rhat": 1.01,
            "minimum_bulk_ess": 2000,
            "minimum_tail_ess": 2000,
            "divergences": 0,
            "maximum_depth_fraction": 0.01,
            "minimum_chain_bfmi": 0.30,
            "mean_grid_edge_mass": 0.005,
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def correctness_checks(
    args: argparse.Namespace,
    data: DataBundle,
    grid: np.ndarray,
    basis: np.ndarray,
    transform,
    forward_all,
    log_density,
) -> None:
    assert len(jax.devices()) == 8, f"expected 8 JAX devices, got {jax.devices()}"
    assert all(device.platform == "gpu" for device in jax.devices())
    assert basis.shape == (args.bins, args.bins - 1), basis.shape
    assert data.likelihood_mask.sum() == 18
    assert np.all(np.diff(data.temperatures_c) > 0)
    zero = jnp.zeros(args.bins)
    weights, ea = transform(zero)
    prediction = forward_all(zero)
    value, gradient = jax.value_and_grad(log_density)(zero)
    np_weights = np.asarray(weights)
    np_prediction = np.asarray(prediction)
    assert np.isclose(np_weights.sum(), 1.0)
    assert np.all(np_weights > 0)
    assert np.all(np_prediction >= -1e-12)
    assert np_prediction.sum() <= 1 + 1e-12
    assert np.isfinite(float(ea))
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    prior_positions = jax.random.normal(jax.random.key(args.seed + 91), (100, args.bins))
    prior_values, prior_gradients = jax.vmap(jax.value_and_grad(log_density))(prior_positions)
    assert np.all(np.isfinite(np.asarray(prior_values)))
    assert np.all(np.isfinite(np.asarray(prior_gradients)))
    assert grid[0] < LOGD_BASE_MEAN < grid[-1]


def affine_preconditioner(log_density, dimension: int, maxiter: int):
    """Find a posterior mode and a regularized inverse-Hessian coordinate map."""
    objective = lambda position: -log_density(position)
    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=maxiter,
        tol=1e-9,
        linesearch="zoom",
        maxls=50,
    )
    result = solver.run(jnp.zeros(dimension))
    mode = result.params
    value = objective(mode)
    gradient = jax.grad(objective)(mode)
    hessian = jax.hessian(objective)(mode)
    mode_np = np.asarray(mode)
    hessian_np = np.asarray(hessian)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_np)
    largest = float(eigenvalues.max())
    floor = max(largest * 1e-8, 1e-8)
    regularized = np.maximum(eigenvalues, floor)
    covariance_root = eigenvectors @ np.diag(1 / np.sqrt(regularized))
    summary = {
        "objective": float(value),
        "gradient_norm": float(np.linalg.norm(np.asarray(gradient))),
        "iterations": int(result.state.iter_num),
        "error": float(result.state.error),
        "hessian_eigenvalue_min": float(eigenvalues.min()),
        "hessian_eigenvalue_max": largest,
        "hessian_floor": floor,
        "regularized_condition": float(regularized.max() / regularized.min()),
    }
    return mode_np, covariance_root, hessian_np, summary


def main() -> None:
    args = parse_args()
    if args.chains % len(jax.devices()) != 0:
        raise SystemExit("--chains must be divisible by the number of JAX devices")
    if args.draws % args.chunk_size != 0:
        raise SystemExit("--draws must be divisible by --chunk-size")
    if args.settle_draws % args.chunk_size != 0:
        raise SystemExit("--settle-draws must be divisible by --chunk-size")
    if args.bins < 4:
        raise SystemExit("--bins must be at least four")

    output_dir = args.results_dir / args.tag
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(
        args.relative_sigma,
        args.minimum_temperature,
        args.normalization_maximum_temperature,
    )
    grid, basis, base_logits, transform, forward_all, native_log_density = build_target(args, data)
    correctness_checks(
        args, data, grid, basis, transform, forward_all, native_log_density
    )
    manifest = write_manifest(output_dir, args, data, grid, basis)

    affine_mean = np.zeros(args.bins)
    affine_root = np.eye(args.bins)
    if args.affine_precondition:
        precondition_started = time.time()
        affine_mean, affine_root, hessian, affine_summary = affine_preconditioner(
            native_log_density, args.bins, args.map_maxiter
        )
        affine_summary["seconds"] = time.time() - precondition_started
        np.savez(
            output_dir / "preconditioner.npz",
            mean=affine_mean,
            covariance_root=affine_root,
            hessian=hessian,
        )
        atomic_json(output_dir / "preconditioner.json", affine_summary)
        manifest["affine_preconditioner"] = affine_summary
        atomic_json(output_dir / "manifest.json", manifest)
        print(f"Affine preconditioner: {json.dumps(affine_summary)}")

    affine_mean_jax = jnp.asarray(affine_mean)
    affine_root_jax = jnp.asarray(affine_root)

    def to_native(sampler_position):
        return affine_mean_jax + affine_root_jax @ sampler_position

    def log_density(sampler_position):
        return native_log_density(to_native(sampler_position))

    print(f"JAX backend: {jax.default_backend()} on {len(jax.devices())} devices")
    print(f"Devices: {jax.devices()}")
    print(
        f"Data: {len(data.temperatures_c)} timed steps; "
        f"{int(data.likelihood_mask.sum())} likelihood steps; likelihood fractions sum "
        f"to {data.observed_fraction[data.likelihood_mask].sum():.6f}"
    )
    print(
        f"Run: {args.chains} chains; {args.warmup} warmup; "
        f"{args.draws} retained draws in {args.chunk_size}-draw chunks"
    )
    print(f"Manifest: {output_dir / 'manifest.json'}")

    device_count = len(jax.devices())
    chains_per_device = args.chains // device_count
    dimension = args.bins
    root_key = jax.random.key(args.seed)
    init_key, warm_key, sample_root = jax.random.split(root_key, 3)
    initial = args.init_scale * jax.random.normal(
        init_key, (device_count, chains_per_device, dimension)
    )

    warmup_started = time.time()
    if args.fixed_step_size is not None:
        if not args.affine_precondition:
            raise SystemExit("--fixed-step-size requires --affine-precondition")
        init_fn = jax.pmap(
            jax.vmap(lambda position: blackjax.nuts.init(position, log_density))
        )
        states = init_fn(initial)
        step_sizes = jnp.full(
            (device_count, chains_per_device), args.fixed_step_size
        )
        if args.diagonal_mass:
            inverse_masses = jnp.ones(
                (device_count, chains_per_device, dimension)
            )
        else:
            inverse_masses = jnp.broadcast_to(
                jnp.eye(dimension),
                (device_count, chains_per_device, dimension, dimension),
            )
        jax.block_until_ready(states.position)
    else:
        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            log_density,
            is_mass_matrix_diagonal=args.diagonal_mass,
            initial_step_size=0.05,
            target_acceptance_rate=args.target_accept,
            max_num_doublings=args.max_doublings,
            progress_bar=False,
        )

        def warm_one(key, position):
            result, _ = adaptation.run(key, position, num_steps=args.warmup)
            return (
                result.state,
                result.parameters["step_size"],
                result.parameters["inverse_mass_matrix"],
            )

        warm_keys = jax.random.split(warm_key, args.chains).reshape(
            device_count, chains_per_device
        )
        warmup_fn = jax.pmap(jax.vmap(warm_one))
        states, step_sizes, inverse_masses = warmup_fn(warm_keys, initial)
        jax.block_until_ready(states.position)
    warmup_seconds = time.time() - warmup_started
    step_sizes_np = np.asarray(step_sizes).reshape(-1)
    state_positions_np = np.asarray(states.position).reshape(args.chains, dimension)
    adaptation_bad = ~np.isfinite(step_sizes_np) | (step_sizes_np <= 0)
    adaptation_bad |= ~np.all(np.isfinite(state_positions_np), axis=1)
    adaptation_bad_fraction = float(np.mean(adaptation_bad))
    finite_steps = step_sizes_np[~adaptation_bad]
    step_ratio = float(finite_steps.max() / finite_steps.min()) if len(finite_steps) else float("inf")
    adaptation_summary = {
        "warmup_seconds": warmup_seconds,
        "bad_chain_fraction": adaptation_bad_fraction,
        "step_size_min": float(finite_steps.min()) if len(finite_steps) else None,
        "step_size_median": float(np.median(finite_steps)) if len(finite_steps) else None,
        "step_size_max": float(finite_steps.max()) if len(finite_steps) else None,
        "step_size_ratio": step_ratio,
    }
    atomic_json(output_dir / "adaptation.json", adaptation_summary)
    print(f"Warmup: {json.dumps(adaptation_summary)}")
    if adaptation_bad_fraction > 0.01 or step_ratio > 1e6:
        raise SystemExit("adaptation kill gate failed")

    def sample_one(key, state, step_size, inverse_mass):
        kernel = blackjax.nuts(
            log_density,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass,
            max_num_doublings=args.max_doublings,
        )
        draw_keys = jax.random.split(key, args.chunk_size)

        def one_step(carry, draw_key):
            next_state, info = kernel.step(draw_key, carry)
            stats = jnp.array(
                [
                    info.acceptance_rate,
                    info.is_divergent,
                    info.num_integration_steps,
                    info.energy,
                ]
            )
            return next_state, (next_state.position, stats)

        final_state, (positions, stats) = jax.lax.scan(one_step, state, draw_keys)
        return final_state, positions, stats

    sample_fn = jax.pmap(jax.vmap(sample_one))

    settle_stats_parts = []
    settle_started = time.time()
    for settle_index in range(args.settle_draws // args.chunk_size):
        settle_key = jax.random.fold_in(sample_root, 100_000 + settle_index)
        settle_keys = jax.random.split(settle_key, args.chains).reshape(
            device_count, chains_per_device
        )
        states, _, settle_stats_device = sample_fn(
            settle_keys, states, step_sizes, inverse_masses
        )
        jax.block_until_ready(states.position)
        settle_stats_parts.append(
            np.asarray(settle_stats_device)
            .transpose(2, 0, 1, 3)
            .reshape(args.chunk_size, args.chains, 4)
        )
    if settle_stats_parts:
        settle_stats = np.concatenate(settle_stats_parts, axis=0)
        max_steps = 2**args.max_doublings - 1
        settle_summary = {
            "draws": args.settle_draws,
            "seconds": time.time() - settle_started,
            "mean_acceptance": float(np.mean(settle_stats[:, :, 0])),
            "divergences": int(np.sum(settle_stats[:, :, 1])),
            "divergence_fraction": float(np.mean(settle_stats[:, :, 1])),
            "max_depth_fraction": float(
                np.mean(settle_stats[:, :, 2] >= max_steps)
            ),
        }
        atomic_json(output_dir / "settling.json", settle_summary)
        print(f"Frozen-kernel settling: {json.dumps(settle_summary)}")
        if (
            settle_summary["divergence_fraction"] > 0.02
            or settle_summary["max_depth_fraction"] > 0.10
        ):
            raise SystemExit("settling kill gate failed")

    all_positions: list[np.ndarray] = []
    all_stats: list[np.ndarray] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    sampling_started = time.time()

    for chunk_index in range(args.draws // args.chunk_size):
        sample_key = jax.random.fold_in(sample_root, chunk_index)
        keys = jax.random.split(sample_key, args.chains).reshape(
            device_count, chains_per_device
        )
        states, positions_device, stats_device = sample_fn(
            keys, states, step_sizes, inverse_masses
        )
        jax.block_until_ready(states.position)
        sampler_positions = np.asarray(positions_device).transpose(2, 0, 1, 3).reshape(
            args.chunk_size, args.chains, dimension
        )
        positions = sampler_positions @ affine_root.T + affine_mean
        stats = np.asarray(stats_device).transpose(2, 0, 1, 3).reshape(
            args.chunk_size, args.chains, 4
        )
        all_positions.append(positions)
        all_stats.append(stats)
        chunk_path = chunks_dir / f"chunk_{chunk_index + 1:04d}.npz"
        np.savez(chunk_path, position=positions, sampler_stats=stats)

        cumulative_positions = np.concatenate(all_positions, axis=0)
        cumulative_stats = np.concatenate(all_stats, axis=0)
        summary = diagnostics(
            cumulative_positions,
            cumulative_stats,
            grid,
            basis,
            base_logits,
            args,
        )
        elapsed = time.time() - sampling_started
        summary.update(
            {
                "chunk": chunk_index + 1,
                "sampling_seconds": elapsed,
                "chain_draws_per_second": float(
                    cumulative_positions.shape[0] * args.chains / elapsed
                ),
                "eta_seconds": float(
                    elapsed
                    / cumulative_positions.shape[0]
                    * (args.draws - cumulative_positions.shape[0])
                ),
            }
        )
        previous = checkpoint_summaries[-1] if checkpoint_summaries else None
        reasons = kill_reasons(summary, previous)
        summary["kill_reasons"] = reasons
        checkpoint_summaries.append(summary)
        atomic_json(output_dir / "latest_diagnostics.json", summary)
        with (output_dir / "diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        state_checkpoint = output_dir / "state_checkpoint.pkl.tmp"
        with state_checkpoint.open("wb") as handle:
            pickle.dump(
                {
                    "state": jax.device_get(states),
                    "step_sizes": jax.device_get(step_sizes),
                    "inverse_masses": jax.device_get(inverse_masses),
                    "completed_draws": int(cumulative_positions.shape[0]),
                    "manifest": manifest,
                },
                handle,
            )
        state_checkpoint.replace(output_dir / "state_checkpoint.pkl")
        print(
            f"Checkpoint {cumulative_positions.shape[0]:5d}/{args.draws}: "
            f"Rhat={summary['max_rhat']:.4f}, bulk={summary['min_bulk_ess']:.0f}, "
            f"tail={summary['min_tail_ess']:.0f}, div={summary['divergences']}, "
            f"depth={summary['max_depth_fraction']:.2%}, "
            f"BFMI={summary['min_bfmi']:.3f}, ETA={summary['eta_seconds'] / 60:.1f}m"
        )
        if reasons:
            print(f"Kill-gate findings: {reasons}")
            if args.kill_gates:
                atomic_json(output_dir / "killed.json", summary)
                raise SystemExit(3)

    final_positions = np.concatenate(all_positions, axis=0)
    final_stats = np.concatenate(all_stats, axis=0)
    final_summary = checkpoint_summaries[-1]
    passed, failures = final_pass(final_summary)
    final_summary["publication_gate_pass"] = passed
    final_summary["publication_gate_failures"] = failures
    final_summary["total_seconds"] = warmup_seconds + time.time() - sampling_started
    atomic_json(output_dir / "final_diagnostics.json", final_summary)
    np.savez(
        output_dir / "posterior.npz",
        position=final_positions,
        sampler_stats=final_stats,
        grid=grid,
        basis=basis,
        base_logits=base_logits,
    )
    print(f"Publication gate: {'PASS' if passed else 'FAIL'} {failures}")
    print(f"Saved {output_dir / 'posterior.npz'}")
    if args.strict_final and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
