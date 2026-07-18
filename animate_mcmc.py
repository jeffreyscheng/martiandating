"""Animate genuine post-warmup NUTS traces as simultaneous comet trails.

Unlike the original chain-by-chain animation, this version never treats the
addition of more chains as convergence.  Every comet is a retained draw from a
fixed (post-adaptation) kernel, all chains move at once, and the actual R-hat,
ESS, and divergence diagnostics stay visible throughout.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/martiandating-matplotlib")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", nargs="?", default="comet_nuts")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=9.0)
    parser.add_argument("--display-draws", type=int, default=240)
    parser.add_argument("--gif", action="store_true", help="Also write a larger GIF copy")
    return parser.parse_args()


ARGS = parse_args()
INPUT = Path(f"results/fit_{ARGS.tag}.pkl")
if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}; run run_comet_nuts.py first")

with INPUT.open("rb") as handle:
    fit = pickle.load(handle)

samples = fit["samples"]
diagnostics = fit["diagnostics"]
logd_all = np.asarray(samples["logD0_r2"])
phi_all = np.asarray(samples["phi"])
draws, chains, domains = logd_all.shape
if domains != 2:
    raise SystemExit("This focused animation expects the ordered two-domain fit")

n_display = min(ARGS.display_draws, draws)
draw_indices = np.linspace(0, draws - 1, n_display, dtype=int)
logd = logd_all[draw_indices]
phi = phi_all[draw_indices]
x = draw_indices + 1

DOMAIN_COLORS = ["#2196f3", "#ff9800"]
DOMAIN_NAMES = ["HRD", "LRD"]
BG = "#080a0f"
GRID = "#313641"
TEXT = "#d8dbe2"
MUTED = "#8b919e"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(14, 7), facecolor=BG)
grid = GridSpec(
    2,
    2,
    width_ratios=[5.3, 1.15],
    hspace=0.28,
    wspace=0.05,
    left=0.075,
    right=0.955,
    top=0.84,
    bottom=0.10,
)
ax_logd = fig.add_subplot(grid[0, 0])
ax_logd_density = fig.add_subplot(grid[0, 1], sharey=ax_logd)
ax_phi = fig.add_subplot(grid[1, 0])
ax_phi_density = fig.add_subplot(grid[1, 1], sharey=ax_phi)


def padded_limits(values, lower=0.5, upper=99.5, fraction=0.12):
    lo, hi = np.percentile(values, [lower, upper])
    pad = max((hi - lo) * fraction, 1e-3)
    return lo - pad, hi + pad


logd_limits = padded_limits(logd_all)
for axis in (ax_logd, ax_phi):
    axis.grid(True, alpha=0.32, color=GRID, linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.set_xlim(x[0], x[-1])

ax_logd.set_ylim(*logd_limits)
ax_phi.set_ylim(0, 1)
ax_logd.set_ylabel("diffusion intercept  ln(D₀/ρ²)", fontsize=11)
ax_phi.set_ylabel("domain volume fraction  φ", fontsize=11)
ax_phi.set_xlabel("retained post-warmup draw  (thinned for display)", fontsize=10, color=MUTED)
ax_logd.tick_params(labelbottom=False)

for axis in (ax_logd_density, ax_phi_density):
    axis.set_facecolor(BG)
    axis.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def add_density(axis, values, limits, color):
    values = np.asarray(values).ravel()
    support = np.linspace(limits[0], limits[1], 300)
    kde = gaussian_kde(values, bw_method=0.22)
    density = kde(support)
    axis.fill_betweenx(support, 0, density, color=color, alpha=0.24)
    axis.plot(density, support, color=color, alpha=0.9, linewidth=1.4)


for domain in range(domains):
    add_density(ax_logd_density, logd_all[:, :, domain], logd_limits, DOMAIN_COLORS[domain])
    add_density(ax_phi_density, phi_all[:, :, domain], (0, 1), DOMAIN_COLORS[domain])

ax_logd_density.set_xlim(left=0)
ax_phi_density.set_xlim(left=0)
ax_logd_density.set_title("posterior", fontsize=9, color=MUTED, pad=7)

# Complete traces are present from frame one at very low opacity.  They orient
# the reader before playback without obscuring the moving comet trails.
for chain in range(chains):
    for domain in range(domains):
        ax_logd.plot(x, logd[:, chain, domain], color=DOMAIN_COLORS[domain], alpha=0.055, lw=0.65)
        ax_phi.plot(x, phi[:, chain, domain], color=DOMAIN_COLORS[domain], alpha=0.055, lw=0.65)

for domain, (name, color) in enumerate(zip(DOMAIN_NAMES, DOMAIN_COLORS)):
    ax_logd.text(
        0.014,
        0.95 - domain * 0.095,
        name,
        transform=ax_logd.transAxes,
        color=color,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

max_rhat = diagnostics["max_rhat"]
min_ess = diagnostics["min_bulk_ess"]
divergences = diagnostics["divergences"]
fig.suptitle(
    f"ordered 2-domain NUTS — {chains} chains after warmup",
    x=0.075,
    y=0.955,
    ha="left",
    fontsize=15,
    fontweight="bold",
    color="#f2f3f5",
)
fig.text(
    0.075,
    0.905,
    f"max rank R̂ {max_rhat:.3f}   |   min bulk ESS {min_ess:,.0f}   |   divergences {divergences}",
    ha="left",
    va="center",
    fontsize=10.5,
    color=MUTED,
)
trail_length = 30
trail_lines = []
heads = []
for axis, values in ((ax_logd, logd), (ax_phi, phi)):
    axis_lines = []
    axis_heads = []
    for chain in range(chains):
        chain_lines = []
        chain_heads = []
        for domain in range(domains):
            color = DOMAIN_COLORS[domain]
            line, = axis.plot([], [], color=color, alpha=0.78, lw=1.35, solid_capstyle="round")
            head, = axis.plot(
                [x[0]],
                [values[0, chain, domain]],
                marker="o",
                markersize=4.6,
                markerfacecolor="#ffffff",
                markeredgecolor=color,
                markeredgewidth=1.3,
                linestyle="none",
                alpha=0.9,
            )
            chain_lines.append(line)
            chain_heads.append(head)
        axis_lines.append(chain_lines)
        axis_heads.append(chain_heads)
    trail_lines.append(axis_lines)
    heads.append(axis_heads)

progress_text = ax_phi.text(
    0.985,
    0.06,
    "",
    transform=ax_phi.transAxes,
    ha="right",
    va="bottom",
    color=MUTED,
    fontsize=9,
)

moving_frames = max(2, int(ARGS.seconds * ARGS.fps) - ARGS.fps)
pause_frames = max(1, int(ARGS.fps))
frame_positions = np.linspace(0, n_display - 1, moving_frames).astype(int)
frame_positions = np.concatenate([frame_positions, np.full(pause_frames, n_display - 1)])


def update(frame_number):
    end = int(frame_positions[frame_number])
    start = max(0, end - trail_length + 1)
    for axis_index, values in enumerate((logd, phi)):
        for chain in range(chains):
            for domain in range(domains):
                line = trail_lines[axis_index][chain][domain]
                head = heads[axis_index][chain][domain]
                line.set_data(x[start : end + 1], values[start : end + 1, chain, domain])
                head.set_data([x[end]], [values[end, chain, domain]])
    progress_text.set_text(f"draw {x[end]:,} / {draws:,}")
    artists = [progress_text]
    for axis_lines, axis_heads in zip(trail_lines, heads):
        for chain_lines, chain_heads in zip(axis_lines, axis_heads):
            artists.extend(chain_lines)
            artists.extend(chain_heads)
    return artists


movie = animation.FuncAnimation(
    fig,
    update,
    frames=len(frame_positions),
    interval=1000 / ARGS.fps,
    blit=True,
)

mp4_path = Path(f"results/{ARGS.tag}_mcmc_comet.mp4")
print(f"Rendering {len(frame_positions)} frames to {mp4_path}...")
writer = animation.FFMpegWriter(
    fps=ARGS.fps,
    codec="libx264",
    bitrate=1800,
    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
)
movie.save(mp4_path, writer=writer, dpi=100)
print(f"Saved {mp4_path} ({mp4_path.stat().st_size / 1e6:.1f} MB)")

if ARGS.gif:
    gif_path = Path(f"results/{ARGS.tag}_mcmc_comet.gif")
    frame_dir = Path(f"results/tmp_{ARGS.tag}_gif")
    frame_dir.mkdir(exist_ok=True)
    frame_paths = []
    for index in range(len(frame_positions)):
        update(index)
        path = frame_dir / f"frame_{index:04d}.png"
        fig.savefig(path, dpi=100, facecolor=BG)
        frame_paths.append(path)
    imageio.mimsave(gif_path, [imageio.imread(path) for path in frame_paths], fps=ARGS.fps, loop=0)
    for path in frame_paths:
        path.unlink()
    frame_dir.rmdir()
    print(f"Saved {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")

plt.close(fig)
