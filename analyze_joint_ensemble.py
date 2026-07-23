"""Diagnostics and figures for the expanded excursion ensemble."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/martiandating-cache")

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from joint_thermochronology_jax import build_joint_target


BLUE = "#315f86"
LIGHT_BLUE = "#8fb8df"
ORANGE = "#e36a2e"
DARK = "#172d3d"
SW_LOGD = (5.7, 9.0)
SW_FRACTION = (0.97, 0.03)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/joint/ensemble_grid_production"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/joint/ensemble_analysis"),
    )
    return parser.parse_args()


def diagnostics(values: np.ndarray) -> dict[str, float]:
    chains_by_draws = np.swapaxes(values, 0, 1)
    return {
        "rhat": float(az.rhat(chains_by_draws, method="rank")),
        "ess_bulk": float(az.ess(chains_by_draws, method="bulk")),
        "ess_tail": float(
            az.ess(chains_by_draws, method="tail", prob=(0.05, 0.95))
        ),
    }


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, ordered_values)


def scenario_index(
    scenarios: list[dict], duration: int, start_age: int
) -> int:
    for scenario in scenarios:
        if (
            int(scenario["duration_my"]) == duration
            and int(scenario["start_age_before_present_my"]) == start_age
        ):
            return int(scenario["index"])
    raise KeyError((duration, start_age))


def conditional_indices(
    responsibilities: np.ndarray,
    index: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = responsibilities[..., index].reshape(-1)
    probabilities = weights / weights.sum()
    return rng.choice(len(weights), size=count, replace=True, p=probabilities)


def draw_mixing(
    posterior: dict[str, np.ndarray],
    summary: dict,
    output: Path,
) -> None:
    variables = [
        ("temperature_c", "temperature (°C)"),
        ("ea_kj_mol", r"$E_a$ (kJ mol$^{-1}$)"),
        ("discrepancy", r"age-spectrum discrepancy $\sigma$"),
    ]
    chain_indices = np.linspace(
        0, posterior["temperature_c"].shape[1] - 1, 12, dtype=int
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.8), sharex=True)
    for axis, (key, label) in zip(axes, variables):
        for chain in chain_indices:
            axis.plot(posterior[key][:, chain], lw=0.5, alpha=0.55)
        stat = summary["diagnostics"][key]
        axis.set_ylabel(label)
        axis.text(
            0.995,
            0.96,
            f"R̂ {stat['rhat']:.3f}   ESS {stat['ess_bulk']:,.0f}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=8,
            color="#4c5962",
        )
        axis.grid(alpha=0.12)
    axes[-1].set_xlabel("retained draw")
    fig.tight_layout()
    fig.savefig(output / "joint_mixing.png", dpi=200)
    plt.close(fig)


def draw_distribution(
    posterior: dict[str, np.ndarray],
    responsibilities: np.ndarray,
    index: int,
    output: Path,
) -> None:
    grid = np.linspace(-4.0, 14.0, posterior["weights"].shape[-1])
    values = posterior["weights"].reshape(-1, posterior["weights"].shape[-1])
    weights = responsibilities[..., index].reshape(-1)
    quantiles = np.asarray(
        [
            weighted_quantile(
                values[:, column], weights, np.asarray([0.025, 0.5, 0.975])
            )
            for column in range(values.shape[1])
        ]
    ).T
    low, median, high = quantiles
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.fill_between(grid, low, high, color=LIGHT_BLUE, alpha=0.55)
    axis.plot(grid, median, color=BLUE, lw=2)
    axis.set_xlabel(r"diffusion scale $\ln(D_0/r^2)$")
    axis.set_ylabel("posterior mass per grid bin")
    axis.grid(alpha=0.12)
    reference = axis.twinx()
    reference.vlines(
        SW_LOGD, 0, SW_FRACTION, color=ORANGE, lw=2, linestyles="--"
    )
    reference.scatter(SW_LOGD, SW_FRACTION, color=ORANGE, s=28, zorder=4)
    reference.annotate(
        "HRD 97%",
        (SW_LOGD[0], SW_FRACTION[0]),
        xytext=(-7, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#a23f22",
        fontsize=8,
    )
    reference.annotate(
        "LRD 3%",
        (SW_LOGD[1], SW_FRACTION[1]),
        xytext=(7, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#a23f22",
        fontsize=8,
    )
    reference.set_ylabel("Shuster–Weiss domain fraction", color="#a23f22")
    reference.set_ylim(0, 1.04)
    reference.tick_params(axis="y", colors="#a23f22")
    reference.spines["right"].set_color("#a23f22")
    fig.tight_layout()
    fig.savefig(output / "joint_diffusion_distribution.png", dpi=200)
    plt.close(fig)


def draw_temperature_violins(
    posterior: dict[str, np.ndarray],
    responsibilities: np.ndarray,
    scenarios: list[dict],
    output: Path,
    rng: np.random.Generator,
) -> None:
    durations = (10, 100, 200, 500, 1300)
    flat_temperature = posterior["temperature_c"].reshape(-1)
    values = []
    exact_quantiles = []
    for duration in durations:
        index = scenario_index(scenarios, duration, duration)
        weights = responsibilities[..., index].reshape(-1)
        values.append(
            flat_temperature[
                conditional_indices(responsibilities, index, 30000, rng)
            ]
        )
        exact_quantiles.append(
            weighted_quantile(
                flat_temperature, weights, np.asarray([0.025, 0.5, 0.975])
            )
        )
    positions = np.arange(len(values))
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    violins = axis.violinplot(
        values,
        positions=positions,
        widths=0.76,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.22,
    )
    for body in violins["bodies"]:
        body.set_facecolor(LIGHT_BLUE)
        body.set_edgecolor(BLUE)
        body.set_alpha(0.82)
    for position, (low, median, high) in zip(positions, exact_quantiles):
        axis.vlines(position, low, high, color=DARK, lw=1.4)
        axis.scatter(position, median, color=ORANGE, s=30, zorder=3)
    axis.axhline(0, color="#5b67b7", ls="--", lw=1.1)
    axis.set_xticks(
        positions, ["10 My", "100 My", "200 My", "500 My", "1.3 Gy"]
    )
    axis.set_ylabel("constant excursion temperature (°C)")
    axis.set_ylim(bottom=-40.0)
    axis.grid(axis="y", alpha=0.12)
    fig.tight_layout()
    fig.savefig(output / "joint_temperature_violin.png", dpi=200)
    plt.close(fig)


def draw_timing(
    summary_scenarios: list[dict], output: Path
) -> dict[str, dict[str, list[float]]]:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = ("#446f91", "#668d3c", "#b36b32", "#76568f")
    timing_summary = {}
    durations = (10, 100, 200, 500)
    for color, duration in zip(colors, durations):
        cells = sorted(
            (
                cell
                for cell in summary_scenarios
                if int(cell["duration_my"]) == duration
            ),
            key=lambda cell: cell["start_age_before_present_my"],
        )
        starts = np.asarray(
            [cell["start_age_before_present_my"] for cell in cells]
        )
        quantiles = np.asarray(
            [cell["temperature_c"]["central_95_and_median"] for cell in cells]
        )
        axis.fill_between(
            starts, quantiles[:, 1], quantiles[:, 2], color=color, alpha=0.14
        )
        axis.plot(
            starts, quantiles[:, 2], color=color, lw=1.8, label=f"{duration} My"
        )
        axis.plot(
            starts, quantiles[:, 1], color=color, lw=0.9, ls=":", alpha=0.9
        )
        timing_summary[str(duration)] = {
            str(int(start)): values.tolist()
            for start, values in zip(starts, quantiles)
        }
    axis.axhline(0, color="#5b67b7", ls="--", lw=1.1)
    axis.set_xlim(0, 1020)
    axis.set_xlabel("excursion start before present (My)")
    axis.set_ylabel("constant excursion temperature (°C)")
    axis.legend(frameon=False, title="duration", ncol=2)
    axis.text(
        0.99,
        0.02,
        "solid: 97.5th percentile   dotted: median",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#59636a",
    )
    axis.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(output / "joint_temperature_timing.png", dpi=200)
    plt.close(fig)
    return timing_summary


def draw_ppc(
    posterior: dict[str, np.ndarray],
    responsibilities: np.ndarray,
    scenarios: list[dict],
    output: Path,
    rng: np.random.Generator,
) -> dict:
    duration = 100
    start_age = 100
    index = scenario_index(scenarios, duration, start_age)
    flat_position = posterior["position"].reshape(
        -1, posterior["position"].shape[-1]
    )
    selected = conditional_indices(responsibilities, index, 4096, rng)
    positions = jnp.asarray(flat_position[selected])
    data, _, _, forward, _ = build_joint_target(
        event_duration_my=float(duration), event_end_lookback_my=0.0
    )
    predict_batch = jax.jit(jax.vmap(forward))
    prediction_parts = [
        predict_batch(positions[start : start + 256])
        for start in range(0, len(positions), 256)
    ]
    ar39 = np.concatenate(
        [np.asarray(part[0]) for part in prediction_parts], axis=0
    )
    ratio = np.concatenate(
        [np.asarray(part[1]) for part in prediction_parts], axis=0
    )
    discrepancy = np.concatenate(
        [np.asarray(part[2]) for part in prediction_parts], axis=0
    )
    ar39_rep = ar39 + rng.normal(size=ar39.shape) * data.ar39_sigma[None, :]
    ratio_scale = np.sqrt(
        data.normalized_ratio_sigma[None, :] ** 2 + discrepancy[:, None] ** 2
    )
    ratio_rep = ratio + rng.standard_t(4, size=ratio.shape) * ratio_scale

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    panels = [
        (
            axes[0],
            ar39,
            ar39_rep,
            data.observed_ar39_fraction,
            data.ar39_sigma,
            "fraction of retained ³⁹Ar",
            "laboratory diffusion schedule",
        ),
        (
            axes[1],
            ratio,
            ratio_rep,
            data.normalized_ratio,
            data.normalized_ratio_sigma,
            r"$R/R_{\mathrm{bulk}}$",
            "natural radiogenic-Ar spectrum",
        ),
    ]
    coverage = {}
    for axis, prediction, replicated, observed, error, ylabel, title in panels:
        low, high = np.quantile(replicated, [0.05, 0.95], axis=0)
        median = np.median(prediction, axis=0)
        axis.fill_between(
            data.temperatures_c, low, high, color=LIGHT_BLUE, alpha=0.55
        )
        axis.plot(data.temperatures_c, median, color=BLUE, lw=1.8)
        scored = data.likelihood_mask
        axis.errorbar(
            data.temperatures_c[scored],
            observed[scored],
            yerr=error[scored],
            fmt="o",
            color=ORANGE,
            ms=3.5,
            capsize=1.5,
        )
        axis.scatter(
            data.temperatures_c[~scored],
            observed[~scored],
            s=22,
            facecolor="white",
            edgecolor="#888888",
            linewidth=0.8,
        )
        axis.set_xlabel("extraction temperature (°C)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.12)
        inside = (observed[scored] >= low[scored]) & (
            observed[scored] <= high[scored]
        )
        coverage[title] = {
            "inside_90_percent": int(inside.sum()),
            "count": int(scored.sum()),
        }
    fig.tight_layout()
    fig.savefig(output / "joint_posterior_predictive.png", dpi=200)
    plt.close(fig)
    return coverage


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    posterior = dict(np.load(args.run_dir / "posterior.npz"))
    run_summary = json.loads((args.run_dir / "summary.json").read_text())
    responsibilities = posterior["scenario_responsibility"]
    scenarios = run_summary["expanded_ensemble"]["scenarios"]
    scalar = {
        key: diagnostics(posterior[key])
        for key in ("temperature_c", "ea_kj_mol", "discrepancy")
    }
    summary = {
        "source_run": str(args.run_dir),
        "diagnostics": scalar,
        "all_parameter_max_rhat": run_summary["max_rhat"],
        "all_parameter_min_bulk_ess": run_summary["minimum_bulk_ess"],
        "all_parameter_min_tail_ess": run_summary["minimum_tail_ess"],
        "divergences": run_summary["divergences"],
        "max_depth_fraction": run_summary["max_depth_fraction"],
        "minimum_bfmi": float(
            np.min(
                az.bfmi(
                    np.swapaxes(posterior["sampler_stats"][:, :, 3], 0, 1)
                )
            )
        ),
        "minimum_responsibility_ess": run_summary["expanded_ensemble"][
            "minimum_responsibility_ess"
        ],
        "maximum_relative_chain_mass_deviation": run_summary[
            "expanded_ensemble"
        ]["maximum_relative_chain_mass_deviation"],
    }
    rng = np.random.default_rng(20260723)
    draw_mixing(posterior, summary, args.output_dir)
    primary_index = scenario_index(scenarios, 100, 100)
    draw_distribution(
        posterior, responsibilities, primary_index, args.output_dir
    )
    draw_temperature_violins(
        posterior, responsibilities, scenarios, args.output_dir, rng
    )
    summary["temperature_timing"] = draw_timing(scenarios, args.output_dir)
    summary["posterior_predictive_coverage"] = draw_ppc(
        posterior, responsibilities, scenarios, args.output_dir, rng
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
