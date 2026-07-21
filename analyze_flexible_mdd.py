"""Posterior predictive and legacy threshold analysis for flexible MDD runs.

The temperature transform in this file is superseded by the normalized
40Ar*/39Ar joint likelihood in ``joint_thermochronology_jax.py``. It conditions
on an externally supplied loss threshold and must not be presented as a direct
temperature posterior from the measured age spectrum.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

from run_flexible_mdd import EA_MEAN, EA_SD, R_GAS, numpy_transform


SECONDS_PER_MY = 1e6 * 365.25 * 24 * 3600
K40_HALF_LIFE_MY = 1248.0
SW_HRD_LOG_DIFFUSION_SCALE = 5.7
SW_LRD_LOG_DIFFUSION_SCALE = 9.0
SW_HRD_FRACTION = 0.97
SW_LRD_FRACTION = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--draws-per-chain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--rock-age-my", type=float, default=1300.0)
    parser.add_argument("--loss-threshold", type=float, default=0.01)
    return parser.parse_args()


def fractional_release(progress: np.ndarray) -> np.ndarray:
    short = 6 * np.sqrt(np.maximum(progress, 0) / np.pi) - 3 * progress
    long = 1 - (6 / np.pi**2) * np.exp(-np.pi**2 * progress)
    return np.clip(np.where(progress < 0.3, short, long), 0, 1)


def selected_posterior(
    result_dir: Path, draws_per_chain: int, seed: int
) -> dict[str, np.ndarray]:
    posterior = np.load(result_dir / "posterior.npz")
    positions = posterior["position"]
    draws = positions.shape[0]
    if draws_per_chain > draws:
        raise ValueError("--draws-per-chain exceeds retained draws")
    rng = np.random.default_rng(seed)
    selected_draws = np.sort(rng.choice(draws, draws_per_chain, replace=False))
    selected_positions = positions[selected_draws]
    manifest = json.loads((result_dir / "manifest.json").read_text())
    flex_scale = float(manifest["args"]["flex_scale"])
    physical = numpy_transform(
        selected_positions,
        posterior["grid"],
        posterior["basis"],
        posterior["base_logits"],
        flex_scale,
    )
    return {
        "positions": selected_positions,
        "grid": posterior["grid"],
        "weights": physical["weights"],
        "ea": physical["Ea_shared"],
        "draw_indices": selected_draws,
        "relative_sigma": float(manifest["args"]["relative_sigma"]),
    }


def posterior_predictions(
    grid: np.ndarray,
    weights: np.ndarray,
    ea: np.ndarray,
    temperatures_k: np.ndarray,
    durations_s: np.ndarray,
) -> np.ndarray:
    shape = weights.shape[:2]
    flat_weights = weights.reshape(-1, weights.shape[-1])
    flat_ea = ea.reshape(-1)
    outputs = []
    for start in range(0, len(flat_ea), 4096):
        stop = min(start + 4096, len(flat_ea))
        rates = np.exp(
            grid[None, :, None]
            - flat_ea[start:stop, None, None]
            * 1e3
            / (R_GAS * temperatures_k[None, None, :])
        )
        cumulative = np.cumsum(rates * durations_s[None, None, :], axis=2)
        release = fractional_release(cumulative)
        increments = np.diff(
            release,
            axis=2,
            prepend=np.zeros((stop - start, len(grid), 1)),
        )
        outputs.append(np.einsum("sb,sbt->st", flat_weights[start:stop], increments))
    return np.concatenate(outputs).reshape(*shape, len(temperatures_k))


def cohort_loss(
    temperature_c: np.ndarray,
    duration_my: float,
    event_end_lookback_my: float,
    rock_age_my: float,
    grid: np.ndarray,
    weights: np.ndarray,
    ea: np.ndarray,
    quadrature_nodes: int = 12,
) -> np.ndarray:
    """Present-day argon loss, including production during/after the event."""
    event_start = rock_age_my - event_end_lookback_my - duration_my
    if event_start < 0:
        raise ValueError("thermal event begins before rock closure")
    decay = np.log(2) / K40_HALF_LIFE_MY
    total_produced = 1 - np.exp(-decay * rock_age_my)
    pre_event_produced = 1 - np.exp(-decay * event_start)

    temperature_k = temperature_c[:, None] + 273.15
    rates = np.exp(
        grid[None, :] - ea[:, None] * 1e3 / (R_GAS * temperature_k)
    )
    full_progress = rates * duration_my * SECONDS_PER_MY
    lost = pre_event_produced * fractional_release(full_progress)

    nodes, quadrature_weights = leggauss(quadrature_nodes)
    time_in_event = 0.5 * duration_my * (nodes + 1)
    integration_weights = 0.5 * duration_my * quadrature_weights
    production = decay * np.exp(
        -decay * (event_start + time_in_event)
    )
    remaining_seconds = (duration_my - time_in_event) * SECONDS_PER_MY
    during_release = fractional_release(
        rates[:, :, None] * remaining_seconds[None, None, :]
    )
    lost += np.sum(
        during_release
        * production[None, None, :]
        * integration_weights[None, None, :],
        axis=2,
    )
    domain_loss = lost / total_produced
    return np.sum(weights * domain_loss, axis=1)


def temperature_limit(
    duration_my: float,
    event_end_lookback_my: float,
    rock_age_my: float,
    loss_threshold: float,
    grid: np.ndarray,
    weights: np.ndarray,
    ea: np.ndarray,
) -> np.ndarray:
    flat_weights = weights.reshape(-1, weights.shape[-1])
    flat_ea = ea.reshape(-1)
    limits = []
    for start in range(0, len(flat_ea), 2048):
        stop = min(start + 2048, len(flat_ea))
        chunk_weights = flat_weights[start:stop]
        chunk_ea = flat_ea[start:stop]
        low = np.full(stop - start, -200.0)
        high = np.full(stop - start, 500.0)
        for _ in range(42):
            middle = (low + high) / 2
            loss = cohort_loss(
                middle,
                duration_my,
                event_end_lookback_my,
                rock_age_my,
                grid,
                chunk_weights,
                chunk_ea,
            )
            hotter = loss >= loss_threshold
            high = np.where(hotter, middle, high)
            low = np.where(hotter, low, middle)
        limits.append((low + high) / 2)
    return np.concatenate(limits).reshape(weights.shape[:2])


def chain_bootstrap_quantile_mcse(
    values: np.ndarray, probabilities=(0.025, 0.5, 0.975), seed=20260720
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chain_count = values.shape[1]
    estimates = []
    for _ in range(200):
        chains = rng.integers(0, chain_count, chain_count)
        estimates.append(np.quantile(values[:, chains].ravel(), probabilities))
    return np.std(np.asarray(estimates), axis=0, ddof=1)


def summarize_distribution(values: np.ndarray, seed: int) -> dict[str, object]:
    probabilities = (0.025, 0.5, 0.975)
    quantiles = np.quantile(values, probabilities)
    mcse = chain_bootstrap_quantile_mcse(values, probabilities, seed)
    return {
        "central_95_and_median": quantiles.tolist(),
        "quantile_mcse_c": mcse.tolist(),
        "probability_above_freezing": float(np.mean(values > 0)),
    }


def draw_distribution_plot(
    result_dir: Path, grid: np.ndarray, weights: np.ndarray
) -> None:
    flat = weights.reshape(-1, weights.shape[-1])
    low, median, high = np.quantile(flat, [0.025, 0.5, 0.975], axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.fill_between(grid, low, high, color="#8fb8df", alpha=0.5, label="posterior central 95%")
    ax.plot(grid, median, color="#163b59", lw=2, label="posterior median")
    ax.set_xlabel(r"$\ln(D_0/r^2)$")
    ax.set_ylabel("probability mass per grid bin")
    ax.set_title("diffusion-scale posterior and Shuster–Weiss domains")
    ax.set_ylim(0, 0.31)

    reference_ax = ax.twinx()
    reference_ax.vlines(
        [SW_HRD_LOG_DIFFUSION_SCALE, SW_LRD_LOG_DIFFUSION_SCALE],
        0,
        [SW_HRD_FRACTION, SW_LRD_FRACTION],
        color="#d95f36",
        lw=2,
        linestyles="--",
        label="Shuster–Weiss two-domain fit",
    )
    reference_ax.scatter(
        [SW_HRD_LOG_DIFFUSION_SCALE, SW_LRD_LOG_DIFFUSION_SCALE],
        [SW_HRD_FRACTION, SW_LRD_FRACTION],
        color="#d95f36",
        s=32,
        zorder=4,
    )
    reference_ax.annotate(
        "HRD 97%",
        (SW_HRD_LOG_DIFFUSION_SCALE, SW_HRD_FRACTION),
        xytext=(-8, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#a23f22",
        fontsize=8,
    )
    reference_ax.annotate(
        "LRD 3%",
        (SW_LRD_LOG_DIFFUSION_SCALE, SW_LRD_FRACTION),
        xytext=(7, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#a23f22",
        fontsize=8,
    )
    reference_ax.set_ylim(0, 1.04)
    reference_ax.set_ylabel("Shuster–Weiss domain fraction", color="#a23f22")
    reference_ax.tick_params(axis="y", colors="#a23f22")
    reference_ax.spines["right"].set_color("#a23f22")

    handles, labels = ax.get_legend_handles_labels()
    reference_handles, reference_labels = reference_ax.get_legend_handles_labels()
    ax.legend(handles + reference_handles, labels + reference_labels, frameon=False)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(result_dir / "diffusion_distribution.png", dpi=180)
    plt.close(fig)


def draw_ppc_plot(
    result_dir: Path,
    temperatures_c: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    predictions: np.ndarray,
    replicated: np.ndarray,
    likelihood_mask: np.ndarray,
) -> None:
    flat = predictions.reshape(-1, predictions.shape[-1])
    median = np.quantile(flat, 0.5, axis=0)
    replicated_flat = replicated.reshape(-1, replicated.shape[-1])
    low, high = np.quantile(replicated_flat, [0.05, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.fill_between(
        temperatures_c,
        low,
        high,
        color="#8fb8df",
        alpha=0.5,
        label="90% posterior-predictive interval",
    )
    ax.plot(temperatures_c, median, color="#163b59", lw=2, label="posterior prediction")
    ax.errorbar(
        temperatures_c[likelihood_mask],
        observed[likelihood_mask],
        yerr=sigma[likelihood_mask],
        fmt="o",
        color="#e36a2e",
        ms=4,
        label="likelihood data",
    )
    ax.scatter(
        temperatures_c[~likelihood_mask],
        observed[~likelihood_mask],
        facecolors="none",
        edgecolors="#777777",
        s=28,
        label="propagated, not scored",
    )
    ax.set_xlabel("laboratory extraction temperature (°C)")
    ax.set_ylabel("fraction of total ³⁹Ar released")
    ax.set_title("posterior predictive release schedule")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(result_dir / "posterior_predictive.png", dpi=180)
    plt.close(fig)


def draw_temperature_plot(
    result_dir: Path, duration_limits: dict[str, np.ndarray]
) -> None:
    labels = list(duration_limits)
    values = [duration_limits[label].ravel() for label in labels]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    violin = ax.violinplot(
        values,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#8fb8df")
        body.set_edgecolor("#315f86")
        body.set_alpha(0.82)
    for position, samples in zip(positions, values):
        low, median, high = np.quantile(samples, [0.025, 0.5, 0.975])
        ax.vlines(position, low, high, color="#163b59", lw=1.4)
        ax.scatter(position, median, color="#e36a2e", s=32, zorder=3)
    ax.axhline(0, color="#315bce", ls="--", lw=1.2)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("maximum constant temperature (°C)")
    ax.set_title("temperature limits for an event ending at present")
    ax.grid(axis="y", alpha=0.12)
    fig.tight_layout()
    fig.savefig(result_dir / "temperature_violin.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selected = selected_posterior(
        args.result_dir, args.draws_per_chain, args.seed
    )
    frame = pd.read_csv("data/nakhla1_parsed_fitted.csv")
    total_ar = frame["39Ar"].sum()
    timed = frame.dropna(subset=["seconds_per_extraction_step"]).copy()
    temperatures_c = timed["Temp"].to_numpy(float)
    temperatures_k = temperatures_c + 273.15
    durations_s = timed["seconds_per_extraction_step"].to_numpy(float)
    observed = timed["39Ar"].to_numpy(float) / total_ar
    measured_sigma = timed["std_39Ar"].to_numpy(float) / total_ar
    sigma = np.sqrt(
        measured_sigma**2 + (selected["relative_sigma"] * observed) ** 2
    )
    likelihood_mask = temperatures_c >= 350

    predictions = posterior_predictions(
        selected["grid"],
        selected["weights"],
        selected["ea"],
        temperatures_k,
        durations_s,
    )
    prediction_median = np.median(predictions, axis=(0, 1))
    standardized = (
        observed[likelihood_mask] - prediction_median[likelihood_mask]
    ) / sigma[likelihood_mask]
    rng = np.random.default_rng(args.seed)
    replicated = predictions + rng.normal(
        size=predictions.shape
    ) * sigma[None, None, :]
    predictive_low, predictive_high = np.quantile(
        replicated, [0.05, 0.95], axis=(0, 1)
    )
    covered = (
        (observed >= predictive_low) & (observed <= predictive_high) & likelihood_mask
    )

    recent_limits = {}
    temperature_summary = {}
    for duration in (10.0, 100.0, 200.0, 500.0):
        label = f"{int(duration)} My"
        limits = temperature_limit(
            duration,
            0.0,
            args.rock_age_my,
            args.loss_threshold,
            selected["grid"],
            selected["weights"],
            selected["ea"],
        )
        recent_limits[label] = limits
        temperature_summary[f"duration_{int(duration)}my_end_0my"] = summarize_distribution(
            limits, args.seed + int(duration)
        )

    timing_limits = {}
    for lookback in (0.0, 250.0, 500.0, 1000.0):
        limits = temperature_limit(
            100.0,
            lookback,
            args.rock_age_my,
            args.loss_threshold,
            selected["grid"],
            selected["weights"],
            selected["ea"],
        )
        timing_limits[f"100 My duration; ends {int(lookback)} My ago"] = limits
        temperature_summary[f"duration_100my_end_{int(lookback)}my"] = summarize_distribution(
            limits, args.seed + int(lookback) + 1000
        )

    summary = {
        "posterior_subset": {
            "draws_per_chain": args.draws_per_chain,
            "chains": int(selected["weights"].shape[1]),
            "samples": int(np.prod(selected["weights"].shape[:2])),
            "relative_sigma": selected["relative_sigma"],
        },
        "posterior_predictive": {
            "standardized_residual_rms": float(np.sqrt(np.mean(standardized**2))),
            "maximum_absolute_standardized_median_residual": float(
                np.max(np.abs(standardized))
            ),
            "observations_in_90pct_predictive_interval": int(covered.sum()),
            "likelihood_observations": int(likelihood_mask.sum()),
        },
        "temperature": temperature_summary,
        "thermal_model": {
            "rock_age_my": args.rock_age_my,
            "k40_half_life_my": K40_HALF_LIFE_MY,
            "loss_threshold": args.loss_threshold,
            "notes": "Constant-temperature event; integrates radiogenic production during the event and post-event regrowth through the present-day inventory denominator.",
        },
    }
    (args.result_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    draw_distribution_plot(args.result_dir, selected["grid"], selected["weights"])
    draw_ppc_plot(
        args.result_dir,
        temperatures_c,
        observed,
        sigma,
        predictions,
        replicated,
        likelihood_mask,
    )
    draw_temperature_plot(args.result_dir, recent_limits)
    np.savez(
        args.result_dir / "temperature_limits.npz",
        **{f"recent_{label.replace(' ', '_')}": value for label, value in recent_limits.items()},
        **{f"timing_{index}": value for index, value in enumerate(timing_limits.values())},
    )
    np.savez_compressed(
        args.result_dir / "analysis_subset.npz",
        position=selected["positions"],
        weights=selected["weights"],
        ea=selected["ea"],
        grid=selected["grid"],
        retained_draw_indices=selected["draw_indices"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
