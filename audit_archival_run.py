"""Audit the tracked 2025 two-domain run and reproduce its website transform.

The tracked pickle contains JAX arrays and is awkward to unpickle outside the
original environment.  ``fit_tracked_2domain_combined.pkl`` is the NumPy copy
made for the website animation and contains the same 101 x 1000 trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")

import matplotlib
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt


R_GAS = 8.314
DURATIONS_MY = [10, 100, 200, 500, 1300]
DURATION_LABELS = ["10 My", "100 My", "200 My", "500 My", "isothermal"]


def split_rank_rhat(draws_by_chain: np.ndarray) -> dict[str, float]:
    """Return Vehtari-style rank and folded split R-hat diagnostics."""
    values = np.asarray(draws_by_chain)[1:]  # first row is the warmup endpoint
    half = values.shape[0] // 2
    split = np.concatenate([values[:half].T, values[-half:].T], axis=0)

    def rhat(values_2d: np.ndarray) -> float:
        ranks = rankdata(values_2d.ravel(), method="average")
        z = norm.ppf((ranks - 3 / 8) / (ranks.size + 1 / 4)).reshape(values_2d.shape)
        draws = z.shape[1]
        within = np.var(z, axis=1, ddof=1).mean()
        between = draws * np.var(z.mean(axis=1), ddof=1)
        return float(np.sqrt(((draws - 1) / draws * within + between / draws) / within))

    rank = rhat(split)
    folded = rhat(np.abs(split - np.median(split)))
    return {"rank": rank, "folded": folded, "max": max(rank, folded)}


def fractional_release(progress: float) -> float:
    if progress < 0.3:
        release = 6 * np.sqrt(progress / np.pi) - 3 * progress
    else:
        release = 1 - (6 / np.pi**2) * np.exp(-np.pi**2 * progress)
    return float(np.clip(release, 0, 1))


def maximum_temperature(ea_kj: float, log_d0_r2: float, duration_my: float) -> float:
    duration_seconds = duration_my * 1e6 * 3.15e7

    def residual(temp_c: float) -> float:
        temp_k = temp_c + 273.15
        log_progress = (
            log_d0_r2
            - ea_kj * 1e3 / (R_GAS * temp_k)
            + np.log(duration_seconds)
        )
        if log_progress > 20:
            release = 1.0
        elif log_progress < -745:
            release = 0.0
        else:
            release = fractional_release(float(np.exp(log_progress)))
        return release - 0.01

    try:
        return float(brentq(residual, -272.9, 1000))
    except (ValueError, OverflowError):
        return float("nan")


def temperature_distributions(ea: np.ndarray, log_d: np.ndarray) -> list[np.ndarray]:
    distributions = []
    for duration in DURATIONS_MY:
        values = np.array(
            [maximum_temperature(a, d, duration) for a, d in zip(ea, log_d)]
        )
        distributions.append(values[np.isfinite(values)])
    return distributions


def draw_archival_violin(distributions: list[np.ndarray], output: Path) -> None:
    # Match the old boxplot's 2.5--97.5% display range. The excluded tails are
    # especially pathological here, which is why the figure is a diagnostic.
    central = []
    intervals = []
    for values in distributions:
        lo, median, hi = np.percentile(values, [2.5, 50, 97.5])
        central.append(values[(values >= lo) & (values <= hi)])
        intervals.append((lo, median, hi))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    positions = np.arange(len(central))
    violin = ax.violinplot(
        central,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#8fb8df")
        body.set_edgecolor("#315f86")
        body.set_alpha(0.82)

    for position, (lo, median, hi) in zip(positions, intervals):
        ax.vlines(position, lo, hi, color="#163b59", linewidth=1.4)
        ax.scatter(position, median, color="#e36a2e", s=34, zorder=3)

    ax.axhline(0, color="#315bce", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, DURATION_LABELS, rotation=12)
    ax.set_ylabel("maximum temperature (°C)")
    ax.set_title("archival temperature transform — invalid diagnostic")
    ax.text(
        0.01,
        0.02,
        "Reproduces the old website figure from the low-retentivity column.\n"
        "Violins show the central 95%; dots are medians. This is not an HRD bound.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
    )
    ax.grid(axis="y", alpha=0.14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/fit_tracked_2domain_combined.pkl"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/archival_temperature_violin.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("results/archival_run_audit.json"),
    )
    args = parser.parse_args()

    with args.input.open("rb") as handle:
        samples = pickle.load(handle)["samples"]

    report: dict[str, object] = {
        "source": str(args.input),
        "shape": list(np.asarray(samples["Ea"]).shape),
        "interpretation": {
            "draw_0": "warmup endpoint",
            "draws_1_to_100": "ordinary retained NUTS draws",
            "domain_0": "low-retentivity after descending final-logD sort",
            "domain_1": "high-retentivity after descending final-logD sort",
        },
        "rhat": {},
    }

    for name in ("Ea", "logD0_r2", "phi"):
        values = np.asarray(samples[name])
        report["rhat"][name] = [
            split_rank_rhat(values[:, :, domain]) for domain in range(values.shape[-1])
        ]

    final_ea = np.asarray(samples["Ea"])[-1]
    final_log_d = np.asarray(samples["logD0_r2"])[-1]

    # This is the ordering mistake behind the old website figure: column zero
    # was treated as HRD by the newer plotting code, although the old notebook
    # sorted domains in descending logD and therefore put LRD first.
    website_distributions = temperature_distributions(final_ea[:, 0], final_log_d[:, 0])
    actual_hrd_distributions = temperature_distributions(final_ea[:, 1], final_log_d[:, 1])

    report["temperature_transform"] = {
        "old_website_lrd_column": {
            label: np.percentile(values, [2.5, 50, 97.5]).tolist()
            for label, values in zip(DURATION_LABELS, website_distributions)
        },
        "nominal_hrd_column": {
            label: np.percentile(values, [2.5, 50, 97.5]).tolist()
            for label, values in zip(DURATION_LABELS, actual_hrd_distributions)
        },
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    draw_archival_violin(website_distributions, args.plot)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.plot}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
