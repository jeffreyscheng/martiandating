"""Render chain-evolution diagnostics for the flexible MDD posterior.

The comet view preserves chronological draw order and chain identity in a
two-dimensional physical summary. The contraction view uses every chain and
the checkpointed rank/folded R-hat and ESS values from the actual production
run. Neither animation reweights or resamples the posterior.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


BG = "#080a0f"
PANEL = "#0d1119"
GRID = "#313641"
TEXT = "#edf0f5"
MUTED = "#9299a7"
COLORS = np.asarray(["#55a7e0", "#58c49b", "#f0be5a", "#e36f63"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=Path("results/flexible/primary_flex28_seed20260720"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--trail-draws", type=int, default=70)
    return parser.parse_args()


def physical_summaries(result_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    posterior = np.load(result_dir / "posterior.npz")
    position = posterior["position"]
    manifest = json.loads((result_dir / "manifest.json").read_text())
    flex_scale = float(manifest["args"]["flex_scale"])
    logits = posterior["base_logits"][None, None, :] + flex_scale * np.einsum(
        "tcl,bl->tcb", position[:, :, :-1], posterior["basis"]
    )
    logits -= logits.max(axis=2, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=2, keepdims=True)
    logd_mean = np.einsum("tcb,b->tc", logits, posterior["grid"])
    centered = posterior["grid"][None, None, :] - logd_mean[:, :, None]
    logd_sd = np.sqrt(np.einsum("tcb,tcb->tc", logits, centered**2))
    ea = 117.0 + 5.4 * position[:, :, -1]
    return ea, logd_mean, logd_sd


def load_checkpoints(result_dir: Path) -> list[dict[str, float]]:
    path = result_dir / "diagnostics.jsonl"
    checkpoints = []
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            checkpoints.append(
                {
                    "draws": int(item["draws"]),
                    "rhat": float(item["max_rhat"]),
                    "bulk": float(item["min_bulk_ess"]),
                    "tail": float(item["min_tail_ess"]),
                    "divergences": int(item["divergences"]),
                }
            )
    return checkpoints


def chain_groups(ea: np.ndarray) -> np.ndarray:
    """Color chains by their mean Ea in the first 100 retained draws."""
    initial = ea[: min(100, len(ea))].mean(axis=0)
    order = np.argsort(initial, kind="stable")
    groups = np.empty(len(initial), dtype=int)
    for group, indices in enumerate(np.array_split(order, 4)):
        groups[indices] = group
    return groups


def padded_limits(values: np.ndarray, low=0.1, high=99.9, pad=0.08):
    lower, upper = np.percentile(values, [low, high])
    margin = (upper - lower) * pad
    return lower - margin, upper + margin


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.grid(color=GRID, alpha=0.48, linewidth=0.7)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def checkpoint_at(checkpoints: list[dict[str, float]], draw: int):
    available = [item for item in checkpoints if item["draws"] <= draw]
    return available[-1] if available else None


def segments_for_frame(
    x: np.ndarray,
    y: np.ndarray,
    end: int,
    trail_draws: int,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, end - trail_draws + 1)
    segments = []
    colors = []
    for chain in range(x.shape[1]):
        points = np.column_stack((x[start : end + 1, chain], y[start : end + 1, chain]))
        if len(points) < 2:
            continue
        segments.extend(np.stack((points[:-1], points[1:]), axis=1))
        alpha = np.linspace(0.035, 0.62, len(points) - 1)
        base = matplotlib.colors.to_rgba(COLORS[groups[chain]])
        colors.extend([(base[0], base[1], base[2], value) for value in alpha])
    return np.asarray(segments), np.asarray(colors)


def save_animation(movie, path: Path, fps: int, dpi: int = 105) -> None:
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2300,
        extra_args=[
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    movie.save(path, writer=writer, dpi=dpi)


def mp4_to_gif(mp4_path: Path, gif_path: Path, fps: int) -> None:
    filter_graph = (
        f"fps={fps},scale=960:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=192:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4_path),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(gif_path),
        ],
        check=True,
    )


def render_comets(
    ea: np.ndarray,
    logd_mean: np.ndarray,
    groups: np.ndarray,
    checkpoints: list[dict[str, float]],
    output_dir: Path,
    fps: int,
    seconds: float,
    trail_draws: int,
) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig, axis = plt.subplots(figsize=(10.2, 6.2), facecolor=BG)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.14, top=0.80)
    style_axis(axis)
    axis.set_xlabel("shared activation energy  Ea  (kJ/mol)")
    axis.set_ylabel("mean diffusion scale  ln(D₀/r²)")
    axis.set_xlim(*padded_limits(ea))
    axis.set_ylim(*padded_limits(logd_mean))

    background_stride = max(1, len(ea) // 800)
    axis.hexbin(
        ea[::background_stride].ravel(),
        logd_mean[::background_stride].ravel(),
        gridsize=58,
        cmap="Blues",
        mincnt=1,
        bins="log",
        alpha=0.28,
        linewidths=0,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=6,
            label=f"initial Ea quartile {index + 1}",
        )
        for index, color in enumerate(COLORS)
    ]
    legend = axis.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=8,
        labelcolor=MUTED,
        title="fixed chain colors",
        title_fontsize=8,
    )
    legend.get_title().set_color(MUTED)

    collection = LineCollection([], linewidths=0.75)
    axis.add_collection(collection)
    heads = axis.scatter(
        ea[0],
        logd_mean[0],
        s=10,
        c=COLORS[groups],
        alpha=0.88,
        edgecolors="none",
        zorder=3,
    )
    fig.suptitle(
        "all 256 posterior chains move through the same region",
        x=0.105,
        y=0.945,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.105,
        0.885,
        "colors record each chain's first-100-draw Ea quartile; mixing makes those colors interpenetrate",
        color=MUTED,
        fontsize=9.5,
    )
    status = fig.text(0.105, 0.055, "", color=MUTED, fontsize=9.3)

    moving_frames = max(2, round(seconds * fps) - fps)
    positions = np.linspace(0, len(ea) - 1, moving_frames, dtype=int)
    positions = np.concatenate([positions, np.full(fps, len(ea) - 1)])

    def update(frame: int):
        end = int(positions[frame])
        segments, segment_colors = segments_for_frame(
            ea, logd_mean, end, trail_draws, groups
        )
        collection.set_segments(segments)
        collection.set_color(segment_colors)
        heads.set_offsets(np.column_stack((ea[end], logd_mean[end])))
        current = checkpoint_at(checkpoints, end + 1)
        if current is None:
            diagnostic = "checkpoint diagnostics begin at draw 250"
        else:
            diagnostic = (
                f"max rank/folded R̂ {current['rhat']:.4f}   ·   "
                f"min bulk ESS {current['bulk']:,.0f}   ·   "
                f"divergences {current['divergences']}"
            )
        status.set_text(
            f"retained draw {end + 1:,} / {len(ea):,}   ·   "
            f"{trail_draws}-draw trails   ·   {diagnostic}"
        )
        return collection, heads, status

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / fps,
        blit=False,
    )
    mp4_path = output_dir / "primary-chain-comets.mp4"
    gif_path = output_dir / "primary-chain-comets.gif"
    save_animation(movie, mp4_path, fps)
    mp4_to_gif(mp4_path, gif_path, min(fps, 10))
    plt.close(fig)
    return mp4_path, gif_path


def render_contraction(
    ea: np.ndarray,
    logd_mean: np.ndarray,
    groups: np.ndarray,
    checkpoints: list[dict[str, float]],
    output_dir: Path,
    fps: int,
    seconds: float,
) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(11.2, 6.3), facecolor=BG)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1],
        hspace=0.36,
        wspace=0.28,
        left=0.08,
        right=0.97,
        bottom=0.13,
        top=0.78,
    )
    means_axis = fig.add_subplot(grid[:, 0])
    rhat_axis = fig.add_subplot(grid[0, 1])
    ess_axis = fig.add_subplot(grid[1, 1])
    for axis in (means_axis, rhat_axis, ess_axis):
        style_axis(axis)

    cumulative_ea = np.cumsum(ea, axis=0)
    cumulative_logd = np.cumsum(logd_mean, axis=0)
    frame_count = max(2, round(seconds * fps) - fps)
    frame_draws = np.unique(
        np.linspace(25, len(ea), frame_count, dtype=int)
    )
    frame_draws = np.concatenate([frame_draws, np.full(fps, len(ea))])
    check_draws = np.asarray([item["draws"] for item in checkpoints])
    check_rhat = np.asarray([item["rhat"] for item in checkpoints])
    check_bulk = np.asarray([item["bulk"] for item in checkpoints])

    mean_frames_ea = np.asarray(
        [cumulative_ea[draw - 1] / draw for draw in np.unique(frame_draws)]
    )
    mean_frames_logd = np.asarray(
        [cumulative_logd[draw - 1] / draw for draw in np.unique(frame_draws)]
    )
    means_axis.set_xlim(*padded_limits(mean_frames_ea, low=0, high=100, pad=0.12))
    means_axis.set_ylim(*padded_limits(mean_frames_logd, low=0, high=100, pad=0.12))
    means_axis.set_xlabel("per-chain running mean Ea  (kJ/mol)")
    means_axis.set_ylabel("per-chain running mean ln(D₀/r²)")
    means_axis.set_title("256 chain means contract to one expectation", fontsize=11)
    points = means_axis.scatter(
        cumulative_ea[24] / 25,
        cumulative_logd[24] / 25,
        s=16,
        c=COLORS[groups],
        alpha=0.82,
        edgecolors="none",
    )
    center = means_axis.scatter(
        [ea[:25].mean()],
        [logd_mean[:25].mean()],
        marker="+",
        s=135,
        linewidths=2.0,
        color="#ffffff",
        zorder=4,
        label="pooled mean",
    )
    means_axis.legend(frameon=False, labelcolor=MUTED, fontsize=8, loc="upper right")

    rhat_axis.set_xlim(0, len(ea))
    rhat_axis.set_ylim(0.998, max(1.07, check_rhat.max() + 0.003))
    rhat_axis.set_ylabel("maximum rank/folded R̂")
    rhat_axis.set_xticklabels([])
    rhat_axis.axhline(1.01, color="#e36f63", linestyle="--", linewidth=1.2)
    rhat_axis.text(
        len(ea) * 0.985,
        1.0115,
        "registered threshold 1.01",
        ha="right",
        color="#e36f63",
        fontsize=7.8,
    )
    rhat_line, = rhat_axis.plot([], [], color="#58c49b", lw=2.0)
    rhat_head, = rhat_axis.plot([], [], "o", color="#58c49b", ms=5)

    ess_axis.set_xlim(0, len(ea))
    ess_axis.set_ylim(0, check_bulk.max() * 1.08)
    ess_axis.set_xlabel("retained draws per chain")
    ess_axis.set_ylabel("minimum bulk ESS")
    ess_axis.axhline(2000, color="#e36f63", linestyle="--", linewidth=1.2)
    ess_axis.text(
        len(ea) * 0.985,
        2600,
        "registered threshold 2,000",
        ha="right",
        color="#e36f63",
        fontsize=7.8,
    )
    ess_line, = ess_axis.plot([], [], color="#55a7e0", lw=2.0)
    ess_head, = ess_axis.plot([], [], "o", color="#55a7e0", ms=5)

    fig.suptitle(
        "between-chain differences vanish while effective samples accumulate",
        x=0.08,
        y=0.945,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.885,
        "running means use every retained draw; diagnostics are the production run's registered checkpoints",
        color=MUTED,
        fontsize=9.4,
    )
    status = fig.text(0.08, 0.045, "", color=MUTED, fontsize=9.2)

    def update(frame: int):
        draw = int(frame_draws[frame])
        chain_ea = cumulative_ea[draw - 1] / draw
        chain_logd = cumulative_logd[draw - 1] / draw
        points.set_offsets(np.column_stack((chain_ea, chain_logd)))
        center.set_offsets([[ea[:draw].mean(), logd_mean[:draw].mean()]])
        visible = check_draws <= draw
        if np.any(visible):
            rhat_line.set_data(check_draws[visible], check_rhat[visible])
            rhat_head.set_data([check_draws[visible][-1]], [check_rhat[visible][-1]])
            ess_line.set_data(check_draws[visible], check_bulk[visible])
            ess_head.set_data([check_draws[visible][-1]], [check_bulk[visible][-1]])
            latest = checkpoints[np.flatnonzero(visible)[-1]]
            diagnostic = (
                f"max R̂ {latest['rhat']:.4f}   ·   min bulk ESS {latest['bulk']:,.0f}   ·   "
                f"min tail ESS {latest['tail']:,.0f}   ·   divergences {latest['divergences']}"
            )
        else:
            rhat_line.set_data([], [])
            rhat_head.set_data([], [])
            ess_line.set_data([], [])
            ess_head.set_data([], [])
            diagnostic = "checkpoint diagnostics begin at draw 250"
        status.set_text(f"{draw:,} retained draws per chain   ·   {diagnostic}")
        return points, center, rhat_line, rhat_head, ess_line, ess_head, status

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_draws),
        interval=1000 / fps,
        blit=False,
    )
    mp4_path = output_dir / "primary-chain-contraction.mp4"
    gif_path = output_dir / "primary-chain-contraction.gif"
    save_animation(movie, mp4_path, fps)
    mp4_to_gif(mp4_path, gif_path, min(fps, 10))
    plt.close(fig)
    return mp4_path, gif_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.result_dir / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    ea, logd_mean, _ = physical_summaries(args.result_dir)
    checkpoints = load_checkpoints(args.result_dir)
    groups = chain_groups(ea)
    print(f"Loaded {ea.shape[1]} chains x {ea.shape[0]} retained draws")
    for path in render_comets(
        ea,
        logd_mean,
        groups,
        checkpoints,
        output_dir,
        args.fps,
        args.seconds,
        args.trail_draws,
    ):
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    for path in render_contraction(
        ea,
        logd_mean,
        groups,
        checkpoints,
        output_dir,
        args.fps,
        args.seconds,
    ):
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
