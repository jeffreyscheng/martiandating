"""Audit shared marginals across the hex, corrected-lab, and joint targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/martiandating-cache")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


R_GAS = 8.314
EA_MEAN = 117.0
EA_SD = 5.4
BLUE = "#315f86"
ORANGE = "#e36a2e"
GRAY = "#777b7d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-subset",
        type=Path,
        default=Path(
            "artifacts/flexible_mdd_20260720/primary/analysis_subset.npz"
        ),
    )
    parser.add_argument(
        "--corrected-run",
        type=Path,
        default=Path("results/flexible/corrected_lab_control_ta98"),
    )
    parser.add_argument(
        "--joint-run",
        type=Path,
        default=Path("results/joint/ensemble_grid_production"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/joint/marginal_audit"),
    )
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def load_old(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    posterior = np.load(path)
    return (
        posterior["grid"],
        posterior["weights"].reshape(-1, posterior["weights"].shape[-1]),
        posterior["ea"].reshape(-1),
    )


def load_corrected(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    posterior = np.load(path / "posterior.npz")
    draw_indices = np.linspace(
        0, posterior["position"].shape[0] - 1, 400, dtype=int
    )
    position = posterior["position"][draw_indices].reshape(
        -1, posterior["position"].shape[-1]
    )
    grid = posterior["grid"]
    logits = posterior["base_logits"][None, :] + 1.5 * (
        position[:, :-1] @ posterior["basis"].T
    )
    return grid, softmax(logits), EA_MEAN + EA_SD * position[:, -1]


def load_joint(
    path: Path, scenario_index: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    posterior = np.load(path / "posterior.npz")
    draw_indices = np.linspace(
        0, posterior["weights"].shape[0] - 1, 1000, dtype=int
    )
    weights = posterior["weights"][draw_indices].reshape(
        -1, posterior["weights"].shape[-1]
    )
    ea = posterior["ea_kj_mol"][draw_indices].reshape(-1)
    all_responsibilities = posterior["scenario_responsibility"][
        draw_indices
    ].reshape(-1, posterior["scenario_responsibility"].shape[-1])
    all_responsibilities /= all_responsibilities.sum(axis=0, keepdims=True)
    responsibility = all_responsibilities[:, scenario_index]
    responsibility /= responsibility.sum()
    return (
        np.linspace(-4.0, 14.0, weights.shape[-1]),
        weights,
        ea,
        responsibility,
        all_responsibilities,
    )


def weighted_quantile(
    values: np.ndarray, probabilities: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    if weights is None:
        return np.quantile(values, probabilities)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values[order])


def feature_values(
    grid: np.ndarray, weights: np.ndarray, ea: np.ndarray
) -> dict[str, np.ndarray]:
    mean = weights @ grid
    variance = np.sum(weights * (grid[None, :] - mean[:, None]) ** 2, axis=1)
    slow = grid < 5.5
    slow_mass = weights[:, slow].sum(axis=1)
    fast_mass = 1.0 - slow_mass
    return {
        "activation energy": ea,
        "mean diffusion scale": mean,
        "diffusion-scale SD": np.sqrt(variance),
        "slow-region mass": slow_mass,
        "slow-region centroid": (weights[:, slow] @ grid[slow]) / slow_mass,
        "fast-region centroid": (weights[:, ~slow] @ grid[~slow]) / fast_mass,
    }


def density(
    values: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    edges = np.linspace(grid[0], grid[-1], len(grid) + 1)
    result, _ = np.histogram(values, bins=edges, weights=weights, density=True)
    return gaussian_filter1d(result, sigma=1.3, mode="nearest")


def overlap(
    first: np.ndarray,
    second: np.ndarray,
    first_weights: np.ndarray | None = None,
    second_weights: np.ndarray | None = None,
) -> float:
    low = min(float(first.min()), float(second.min()))
    high = max(float(first.max()), float(second.max()))
    edges = np.linspace(low, high, 401)
    a, _ = np.histogram(first, bins=edges, weights=first_weights, density=True)
    b, _ = np.histogram(second, bins=edges, weights=second_weights, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(a, b).sum() * width)


def legacy_release(progress: np.ndarray) -> np.ndarray:
    short = 6.0 * np.sqrt(np.maximum(progress, 0.0) / np.pi) - 3.0 * progress
    long = 1.0 - (6.0 / np.pi**2) * np.exp(-np.pi**2 * progress)
    return np.clip(np.where(progress < 0.3, short, long), 0.0, 1.0)


def exact_release(progress: np.ndarray) -> np.ndarray:
    safe = np.maximum(progress, 1e-300)
    short = 6.0 * np.sqrt(safe / np.pi) - 3.0 * progress
    long = np.ones_like(progress)
    # Sixteen modes agree with the 64-mode implementation to machine precision
    # over the long-time branch, which starts at progress 0.05.
    for n in range(1, 17):
        long -= (6.0 / (np.pi**2 * n**2)) * np.exp(
            -progress * np.pi**2 * n**2
        )
    return np.clip(np.where(progress < 0.05, short, long), 0.0, 1.0)


def gaussian_log_likelihood(
    observed: np.ndarray,
    sigma: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    residual = (observed[mask] - prediction[:, mask]) / sigma[mask]
    return np.sum(
        -0.5 * residual**2
        - np.log(sigma[mask])
        - 0.5 * np.log(2.0 * np.pi),
        axis=1,
    )


def importance_audit(
    grid: np.ndarray, weights: np.ndarray, ea: np.ndarray
) -> dict[str, float | list[float]]:
    frame = pd.read_csv("data/nakhla1_parsed_fitted.csv")
    timed = frame.dropna(subset=["seconds_per_extraction_step"])
    temperatures_k = timed["Temp"].to_numpy(float) + 273.15
    durations_s = timed["seconds_per_extraction_step"].to_numpy(float)
    mask = timed["Temp"].to_numpy(float) >= 350.0
    all_total = float(frame["39Ar"].sum())
    retained_total = float(frame.loc[frame["Temp"] <= 850.0, "39Ar"].sum())
    observed_old = timed["39Ar"].to_numpy(float) / all_total
    observed_new = timed["39Ar"].to_numpy(float) / retained_total
    sigma_old = np.sqrt(
        (timed["std_39Ar"].to_numpy(float) / all_total) ** 2
        + (0.10 * observed_old) ** 2
    )
    sigma_new = np.sqrt(
        (timed["std_39Ar"].to_numpy(float) / retained_total) ** 2
        + (0.10 * observed_new) ** 2
    )
    log_ratio_parts = []
    for start in range(0, len(weights), 256):
        batch_weights = weights[start : start + 256]
        batch_ea = ea[start : start + 256]
        rates = np.exp(
            grid[None, :, None]
            - batch_ea[:, None, None]
            * 1e3
            / (R_GAS * temperatures_k[None, None, :])
        )
        cumulative = np.cumsum(rates * durations_s[None, None, :], axis=2)
        previous = np.concatenate(
            [
                np.zeros((len(batch_weights), len(grid), 1)),
                cumulative[:, :, :-1],
            ],
            axis=2,
        )
        old_prediction = np.einsum(
            "bd,bdj->bj",
            batch_weights,
            legacy_release(cumulative) - legacy_release(previous),
        )
        new_prediction = np.einsum(
            "bd,bdj->bj",
            batch_weights,
            exact_release(cumulative) - exact_release(previous),
        )
        log_ratio_parts.append(
            gaussian_log_likelihood(
                observed_new, sigma_new, new_prediction, mask
            )
            - gaussian_log_likelihood(
                observed_old, sigma_old, old_prediction, mask
            )
        )
    log_ratio = np.concatenate(log_ratio_parts)
    smoothed_log_weights, pareto_k = az.psislw(log_ratio)
    importance_weights = np.exp(smoothed_log_weights)
    importance_weights /= importance_weights.sum()
    return {
        "samples": int(len(log_ratio)),
        "pareto_k": float(pareto_k),
        "importance_ess": float(1.0 / np.sum(importance_weights**2)),
        "maximum_weight": float(importance_weights.max()),
        "log_likelihood_ratio_quantiles": np.quantile(
            log_ratio, [0.0, 0.01, 0.5, 0.99, 1.0]
        ).tolist(),
    }


def draw_weight_comparison(
    output: Path,
    grid: np.ndarray,
    groups: list[tuple[str, np.ndarray, np.ndarray | None, str]],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.4), sharex=True, sharey=True)
    for axis, (label, values, sample_weights, color) in zip(axes, groups):
        quantiles = np.asarray(
            [
                weighted_quantile(
                    values[:, column],
                    np.asarray([0.025, 0.5, 0.975]),
                    sample_weights,
                )
                for column in range(values.shape[1])
            ]
        ).T
        axis.fill_between(
            grid, quantiles[0], quantiles[2], color=color, alpha=0.20
        )
        axis.plot(grid, quantiles[1], color=color, lw=1.8)
        axis.text(0.01, 0.92, label, transform=axis.transAxes, va="top")
        axis.set_ylabel("mass / bin")
        axis.grid(alpha=0.10)
    axes[-1].set_xlabel(r"diffusion scale $\ln(D_0/r^2)$")
    fig.tight_layout()
    fig.savefig(output / "diffusion_marginal_comparison.png", dpi=220)
    plt.close(fig)


def draw_feature_comparison(
    output: Path,
    feature_groups: list[
        tuple[str, dict[str, np.ndarray], np.ndarray | None, str]
    ],
) -> None:
    names = list(feature_groups[0][1])
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6))
    for axis, name in zip(axes.ravel(), names):
        all_values = np.concatenate([group[1][name] for group in feature_groups])
        grid = np.linspace(
            np.quantile(all_values, 0.001), np.quantile(all_values, 0.999), 300
        )
        for label, features, sample_weights, color in feature_groups:
            axis.plot(
                grid,
                density(features[name], grid, sample_weights),
                color=color,
                lw=1.7,
                label=label,
            )
        axis.set_xlabel(name)
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "shared_parameter_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_grid, old_weights, old_ea = load_old(args.old_subset)
    corrected_grid, corrected_weights, corrected_ea = load_corrected(
        args.corrected_run
    )
    (
        joint_grid,
        joint_weights,
        joint_ea,
        joint_responsibility,
        all_joint_responsibilities,
    ) = load_joint(args.joint_run)
    np.testing.assert_allclose(old_grid, corrected_grid)
    np.testing.assert_allclose(old_grid, joint_grid)

    old_features = feature_values(old_grid, old_weights, old_ea)
    corrected_features = feature_values(
        old_grid, corrected_weights, corrected_ea
    )
    joint_features = feature_values(old_grid, joint_weights, joint_ea)
    comparison = {}
    for name in old_features:
        scenario_overlaps = np.asarray(
            [
                overlap(
                    corrected_features[name],
                    joint_features[name],
                    second_weights=all_joint_responsibilities[:, index],
                )
                for index in range(all_joint_responsibilities.shape[1])
            ]
        )
        comparison[name] = {
            "old_hex_central_95_and_median": weighted_quantile(
                old_features[name], np.asarray([0.025, 0.5, 0.975]), None
            ).tolist(),
            "corrected_lab_central_95_and_median": weighted_quantile(
                corrected_features[name],
                np.asarray([0.025, 0.5, 0.975]),
                None,
            ).tolist(),
            "joint_100my_recent_central_95_and_median": weighted_quantile(
                joint_features[name],
                np.asarray([0.025, 0.5, 0.975]),
                joint_responsibility,
            ).tolist(),
            "overlap_old_vs_corrected": overlap(
                old_features[name], corrected_features[name]
            ),
            "overlap_corrected_vs_joint": overlap(
                corrected_features[name],
                joint_features[name],
                second_weights=joint_responsibility,
            ),
            "overlap_corrected_vs_joint_across_21_scenarios": {
                "minimum": float(scenario_overlaps.min()),
                "median": float(np.median(scenario_overlaps)),
                "maximum": float(scenario_overlaps.max()),
            },
        }
    summary = {
        "targets": {
            "old_hex": (
                "all-extraction 39Ar normalization; legacy spherical release; "
                "laboratory likelihood only"
            ),
            "corrected_lab": (
                "39Ar retained through 850 C; exact spherical release; "
                "laboratory likelihood only"
            ),
            "joint_100my_recent": (
                "corrected laboratory likelihood plus natural radiogenic-Ar "
                "spectrum; conditioned on a recent 100 My excursion"
            ),
        },
        "importance_reweighting_old_to_corrected": importance_audit(
            old_grid, old_weights, old_ea
        ),
        "shared_marginals": comparison,
    }
    draw_weight_comparison(
        args.output_dir,
        old_grid,
        [
            ("old hex run", old_weights, None, GRAY),
            ("corrected laboratory control", corrected_weights, None, BLUE),
            (
                "joint: 100 My ending now",
                joint_weights,
                joint_responsibility,
                ORANGE,
            ),
        ],
    )
    draw_feature_comparison(
        args.output_dir,
        [
            ("old hex run", old_features, None, GRAY),
            ("corrected lab", corrected_features, None, BLUE),
            (
                "joint 100 My recent",
                joint_features,
                joint_responsibility,
                ORANGE,
            ),
        ],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
