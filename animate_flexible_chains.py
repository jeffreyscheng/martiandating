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
    parser.add_argument("--trail-frames", type=int, default=22)
    parser.add_argument("--smoothing-draws", type=int, default=31)
    parser.add_argument("--domain-split", type=float, default=5.5)
    parser.add_argument(
        "--only", choices=("both", "comets", "contraction"), default="both"
    )
    return parser.parse_args()


def physical_summaries(
    result_dir: Path, domain_split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    slower = posterior["grid"] < domain_split
    faster = ~slower
    if not np.any(slower) or not np.any(faster):
        raise ValueError("--domain-split must lie inside the diffusion grid")
    slow_centroid = np.sum(
        logits[:, :, slower] * posterior["grid"][slower], axis=2
    ) / np.sum(logits[:, :, slower], axis=2)
    fast_centroid = np.sum(
        logits[:, :, faster] * posterior["grid"][faster], axis=2
    ) / np.sum(logits[:, :, faster], axis=2)
    ea = 117.0 + 5.4 * position[:, :, -1]
    return ea, logd_mean, logd_sd, slow_centroid, fast_centroid


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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered display smoothing with reflected edges; samples stay unmodified."""
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="reflect")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)))
    return (cumulative[window:] - cumulative[:-window]) / window


def representative_chains(ea: np.ndarray, count: int = 5) -> np.ndarray:
    """Select deterministic chains spanning the first-100-draw Ea distribution."""
    initial = ea[: min(100, len(ea))].mean(axis=0)
    order = np.argsort(initial, kind="stable")
    ranks = np.linspace(0.08, 0.92, count)
    return order[np.rint(ranks * (len(order) - 1)).astype(int)]


def fading_segments(
    x: np.ndarray,
    y: np.ndarray,
    color: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack((x, y))
    if len(points) < 2:
        return np.empty((0, 2, 2)), np.empty((0, 4)), np.empty(0)
    segments = np.stack((points[:-1], points[1:]), axis=1)
    recency = np.linspace(0.0, 1.0, len(segments))
    rgba = matplotlib.colors.to_rgba(color)
    colors = np.column_stack(
        (
            np.full(len(segments), rgba[0]),
            np.full(len(segments), rgba[1]),
            np.full(len(segments), rgba[2]),
            0.05 + 0.90 * recency**1.7,
        )
    )
    widths = 0.45 + 2.0 * recency**1.6
    return segments, colors, widths


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
    slow_centroid: np.ndarray,
    fast_centroid: np.ndarray,
    domain_split: float,
    output_dir: Path,
    fps: int,
    seconds: float,
    trail_frames: int,
    smoothing_draws: int,
) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig, axis = plt.subplots(figsize=(10.2, 6.2), facecolor=BG)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.14, top=0.80)
    style_axis(axis)
    axis.set_xlabel("shared activation energy  Ea  (kJ/mol)")
    axis.set_ylabel("domain diffusion scale  ln(D₀/r²)")
    axis.set_xlim(*padded_limits(ea))
    combined = np.concatenate((slow_centroid.ravel(), fast_centroid.ravel()))
    axis.set_ylim(*padded_limits(combined))

    background_stride = max(1, len(ea) // 800)
    axis.hexbin(
        ea[::background_stride].ravel(),
        slow_centroid[::background_stride].ravel(),
        gridsize=54,
        cmap="Blues",
        mincnt=1,
        bins="log",
        alpha=0.24,
        linewidths=0,
    )
    axis.hexbin(
        ea[::background_stride].ravel(),
        fast_centroid[::background_stride].ravel(),
        gridsize=54,
        cmap="Oranges",
        mincnt=1,
        bins="log",
        alpha=0.20,
        linewidths=0,
    )
    axis.axhspan(axis.get_ylim()[0], domain_split, color="#408fca", alpha=0.035)
    axis.axhspan(domain_split, axis.get_ylim()[1], color="#d88d3e", alpha=0.035)
    axis.axhline(domain_split, color=GRID, lw=0.8, ls="--")
    axis.text(
        0.014,
        0.955,
        "faster-diffusing region",
        transform=axis.transAxes,
        color="#e8a35d",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )
    axis.text(
        0.014,
        0.055,
        "slower-diffusing region",
        transform=axis.transAxes,
        color="#75b9e6",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    selected = representative_chains(ea, 5)
    chain_colors = np.asarray(["#62b6f0", "#60d2a6", "#f1c75b", "#ee8969", "#b18ae8"])
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=6,
            label=f"chain {chain + 1}",
        )
        for chain, color in zip(selected, chain_colors)
    ]
    legend_handles.extend(
        [
            Line2D(
                [0], [0], marker="o", color="white", markerfacecolor="none",
                linestyle="none", markersize=6, label="slower centroid"
            ),
            Line2D(
                [0], [0], marker="D", color="white", markerfacecolor="none",
                linestyle="none", markersize=5, label="faster centroid"
            ),
        ]
    )
    axis.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=4,
        fontsize=7.6,
        labelcolor=MUTED,
    )

    smooth_ea = moving_average(ea[:, selected], smoothing_draws)
    smooth_slow = moving_average(slow_centroid[:, selected], smoothing_draws)
    smooth_fast = moving_average(fast_centroid[:, selected], smoothing_draws)

    # Five chains enter one at a time, then overlap for most of the animation.
    # Each traverses all retained draws and vanishes as soon as it completes.
    life_frames = max(36, round(seconds * fps))
    entry_gap = max(3, round(0.55 * fps))
    pause_frames = max(3, round(0.45 * fps))
    frame_count = life_frames + entry_gap * (len(selected) - 1) + pause_frames
    draw_lookup = np.linspace(0, len(ea) - 1, life_frames, dtype=int)

    collections = []
    slow_heads = []
    fast_heads = []
    for color in chain_colors:
        slow_line = LineCollection([], linewidths=[])
        fast_line = LineCollection([], linewidths=[])
        axis.add_collection(slow_line)
        axis.add_collection(fast_line)
        slow_head, = axis.plot(
            [], [], marker="o", markersize=5.5, markerfacecolor="#ffffff",
            markeredgecolor=color, markeredgewidth=1.5, linestyle="none", zorder=4
        )
        fast_head, = axis.plot(
            [], [], marker="D", markersize=5.0, markerfacecolor="#ffffff",
            markeredgecolor=color, markeredgewidth=1.5, linestyle="none", zorder=4
        )
        collections.append((slow_line, fast_line))
        slow_heads.append(slow_head)
        fast_heads.append(fast_head)

    fig.suptitle(
        "paired diffusion regimes share one activation energy",
        x=0.105,
        y=0.945,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.105,
        0.885,
        "five representative chains · two synchronized comets per chain · old trajectory segments disappear",
        color=MUTED,
        fontsize=9.5,
    )
    status = fig.text(0.105, 0.055, "", color=MUTED, fontsize=9.3)

    def update(frame: int):
        active = 0
        progress = []
        artists = [status]
        for index in range(len(selected)):
            local = frame - index * entry_gap
            slow_line, fast_line = collections[index]
            if local < 0 or local >= life_frames:
                slow_line.set_segments([])
                fast_line.set_segments([])
                slow_heads[index].set_data([], [])
                fast_heads[index].set_data([], [])
            else:
                active += 1
                start_local = max(0, local - trail_frames + 1)
                local_indices = np.arange(start_local, local + 1)
                draws = draw_lookup[local_indices]
                color = chain_colors[index]
                slow_segments, slow_colors, slow_widths = fading_segments(
                    smooth_ea[draws, index], smooth_slow[draws, index], color
                )
                fast_segments, fast_colors, fast_widths = fading_segments(
                    smooth_ea[draws, index], smooth_fast[draws, index], color
                )
                slow_line.set_segments(slow_segments)
                slow_line.set_color(slow_colors)
                slow_line.set_linewidths(slow_widths)
                fast_line.set_segments(fast_segments)
                fast_line.set_color(fast_colors)
                fast_line.set_linewidths(fast_widths)
                draw = int(draw_lookup[local])
                slow_heads[index].set_data(
                    [smooth_ea[draw, index]], [smooth_slow[draw, index]]
                )
                fast_heads[index].set_data(
                    [smooth_ea[draw, index]], [smooth_fast[draw, index]]
                )
                progress.append(f"{selected[index] + 1}: {draw + 1:,}")
            artists.extend((slow_line, fast_line, slow_heads[index], fast_heads[index]))
        status.set_text(
            f"{active} active chains   ·   retained draw by chain  "
            + ("  |  ".join(progress) if progress else "complete")
            + f"   ·   {smoothing_draws}-draw moving-average display path"
        )
        return artists

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
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
    ea, logd_mean, _, slow_centroid, fast_centroid = physical_summaries(
        args.result_dir, args.domain_split
    )
    checkpoints = load_checkpoints(args.result_dir)
    groups = chain_groups(ea)
    print(f"Loaded {ea.shape[1]} chains x {ea.shape[0]} retained draws")
    if args.only in ("both", "comets"):
        for path in render_comets(
            ea,
            slow_centroid,
            fast_centroid,
            args.domain_split,
            output_dir,
            args.fps,
            args.seconds,
            args.trail_frames,
            args.smoothing_draws,
        ):
            print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if args.only in ("both", "contraction"):
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
