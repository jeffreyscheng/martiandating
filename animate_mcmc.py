"""Comet-style MCMC convergence animation — logD + phi only."""
import pickle, numpy as np, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import imageio.v2 as imageio

tag = sys.argv[1] if len(sys.argv) > 1 else "PF_NUTS_f003"
pkl = f"results/fit_{tag}_combined.pkl"
if not os.path.exists(pkl):
    pkl = f"results/fit_{tag}_gpu0.pkl"
d = pickle.load(open(pkl, 'rb'))
fs = d['samples']

logD = fs['logD0_r2']
phi = fs['phi']

nl, nc, nd = logD.shape
print(f"Data: {nl} loops, {nc} chains, {nd} domains")

FPS = 30
PRELOAD_CHAINS = 3
N_SLOW_COMETS = 8
SLOW_CROSS_S = 0.2
SLOW_PAUSE_S = 0.2
n_chains_show = min(nc, 100)

slow_cross_frames = max(1, int(SLOW_CROSS_S * FPS))
slow_pause_frames = max(1, int(SLOW_PAUSE_S * FPS))
slow_total = N_SLOW_COMETS * (slow_cross_frames + slow_pause_frames)
total_frames = 15 * FPS
barrage_frames = total_frames - slow_total - FPS
n_barrage = n_chains_show - PRELOAD_CHAINS - N_SLOW_COMETS
barrage_per = max(1, barrage_frames // max(n_barrage, 1))

domain_colors_rgb = [
    (0.13, 0.59, 0.95),
    (1.00, 0.60, 0.00),
    (0.30, 0.69, 0.31),
    (0.91, 0.12, 0.39),
    (0.61, 0.15, 0.69),
]

logd_range = [np.percentile(logD, 0.5), np.percentile(logD, 99.5)]
logd_pad = (logd_range[1] - logd_range[0]) * 0.15
phi_range = [0, min(1, np.percentile(phi, 99.5) * 1.2)]

plt.rcParams.update({
    'figure.facecolor': 'black', 'axes.facecolor': 'black',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#cccccc',
    'text.color': '#cccccc', 'xtick.color': '#666666',
    'ytick.color': '#666666',
})

fig = plt.figure(figsize=(14, 7), facecolor='black')
gs = GridSpec(2, 2, width_ratios=[5, 1], hspace=0.3, wspace=0.05,
             left=0.08, right=0.95, top=0.92, bottom=0.08)

ax_logd = fig.add_subplot(gs[0, 0])
ax_logd_m = fig.add_subplot(gs[0, 1], sharey=ax_logd)
ax_phi = fig.add_subplot(gs[1, 0])
ax_phi_m = fig.add_subplot(gs[1, 1], sharey=ax_phi)

axes_trace = [ax_logd, ax_phi]
axes_marg = [ax_logd_m, ax_phi_m]

tmp_dir = f"results/tmp_anim_{tag}"
os.makedirs(tmp_dir, exist_ok=True)
frame_paths = []


def draw_comet_trail(ax, xs, ys, color_rgb, trail_len=15):
    n = len(xs)
    if n < 2: return
    fade_start = max(0, n - trail_len)
    for i in range(fade_start, n - 1):
        progress = (i - fade_start) / max(trail_len, 1)
        alpha = 0.1 + 0.7 * progress
        c_bright = tuple(min(1, c + 0.3 * progress) for c in color_rgb)
        ax.plot(xs[i:i+2], ys[i:i+2], color=c_bright, alpha=alpha, lw=1.8)
    if fade_start > 0:
        ax.plot(xs[:fade_start+1], ys[:fade_start+1], color=color_rgb, alpha=0.08, lw=0.8)
    hx, hy = xs[-1], ys[-1]
    for offset in range(4, 0, -1):
        idx = max(0, n - 1 - offset)
        ax.scatter(xs[idx], ys[idx], s=max(80-offset*12, 10), color='white',
                  alpha=0.15*(1-offset/5), edgecolors='none', zorder=10)
    ax.scatter(hx, hy, s=120, color=color_rgb, alpha=0.3, edgecolors='none', zorder=11)
    ax.scatter(hx, hy, s=60, color='white', alpha=0.5, edgecolors='none', zorder=12)
    ax.scatter(hx, hy, s=20, color='white', alpha=0.95, edgecolors='none', zorder=13)


def draw_marginals(n_chains_so_far):
    half = nl // 2
    for ax_m in axes_marg:
        ax_m.cla(); ax_m.set_facecolor('black')
        ax_m.tick_params(labelleft=False, left=False); ax_m.set_xticks([])
        for spine in ax_m.spines.values(): spine.set_color('#333333')

    for dd in range(nd):
        c = domain_colors_rgb[dd % len(domain_colors_rgb)]
        logd_vals = logD[half:, :n_chains_so_far, dd].flatten()
        phi_vals = phi[half:, :n_chains_so_far, dd].flatten()

        if len(logd_vals) > 20 and np.std(logd_vals) > 0.01:
            y_grid = np.linspace(logd_range[0]-logd_pad, logd_range[1]+logd_pad, 200)
            try:
                kde = gaussian_kde(logd_vals, bw_method=0.3)
                ax_logd_m.fill_betweenx(y_grid, 0, kde(y_grid), color=c, alpha=0.4)
                ax_logd_m.plot(kde(y_grid), y_grid, color=c, alpha=0.7, lw=1)
            except: pass

        if len(phi_vals) > 20 and np.std(phi_vals) > 0.001:
            y_grid = np.linspace(phi_range[0], phi_range[1], 200)
            try:
                kde = gaussian_kde(phi_vals, bw_method=0.3)
                ax_phi_m.fill_betweenx(y_grid, 0, kde(y_grid), color=c, alpha=0.4)
                ax_phi_m.plot(kde(y_grid), y_grid, color=c, alpha=0.7, lw=1)
            except: pass


def setup_axes():
    for ax in axes_trace:
        ax.set_facecolor('black'); ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values(): spine.set_color('#333333')
    ax_logd.set_xlim(0, nl); ax_phi.set_xlim(0, nl)
    ax_logd.set_ylim(logd_range[0]-logd_pad, logd_range[1]+logd_pad)
    ax_phi.set_ylim(phi_range[0], phi_range[1])
    ax_logd.set_ylabel('Diffusion rate ln(D₀/ρ²)', fontsize=11)
    ax_phi.set_ylabel('Volume fraction φ', fontsize=11)
    ax_phi.set_xlabel('MCMC sample index', fontsize=11, color='#999999')


def render_frame(frame_i, chain_being_animated, comet_progress, chains_completed):
    for ax in axes_trace: ax.cla()
    setup_axes()
    chain_label = chains_completed + 1 if comet_progress > 0 else chains_completed
    ax_logd.set_title(f'Bayesian MDD Thermochronology — {chain_label} / {n_chains_show} chains',
                     fontsize=12, color='#dddddd', fontweight='bold')

    for c in range(chains_completed):
        for dd in range(nd):
            col = domain_colors_rgb[dd % len(domain_colors_rgb)]
            ax_logd.plot(range(nl), logD[:, c, dd], color=col, alpha=0.05, lw=0.5)
            ax_phi.plot(range(nl), phi[:, c, dd], color=col, alpha=0.05, lw=0.5)

    if comet_progress > 0 and chain_being_animated < nc:
        t_end = max(2, int(nl * comet_progress))
        ts = list(range(t_end))
        c = chain_being_animated
        for dd in range(nd):
            col = domain_colors_rgb[dd % len(domain_colors_rgb)]
            draw_comet_trail(ax_logd, ts, logD[:t_end, c, dd].tolist(), col)
            draw_comet_trail(ax_phi, ts, phi[:t_end, c, dd].tolist(), col)

    for dd in range(nd):
        col = domain_colors_rgb[dd % len(domain_colors_rgb)]
        ax_logd.text(0.02, 0.95-dd*0.1, f'D{dd+1}', transform=ax_logd.transAxes,
                    color=col, fontsize=10, fontweight='bold', va='top')

    draw_marginals(chains_completed)

    path = os.path.join(tmp_dir, f"frame_{frame_i:04d}.png")
    fig.savefig(path, dpi=100, facecolor='black')
    frame_paths.append(path)


# Build schedule
print("Building frame schedule...")
schedule = []
chains_completed = PRELOAD_CHAINS

for comet_i in range(N_SLOW_COMETS):
    c = PRELOAD_CHAINS + comet_i
    for f in range(slow_cross_frames):
        schedule.append((c, (f+1)/slow_cross_frames, chains_completed))
    chains_completed = c + 1
    for f in range(slow_pause_frames):
        schedule.append((c, -1, chains_completed))

for comet_i in range(n_barrage):
    c = PRELOAD_CHAINS + N_SLOW_COMETS + comet_i
    schedule.append((c, 1.0, chains_completed))
    chains_completed = c + 1
    for _ in range(barrage_per - 1):
        schedule.append((c, -1, chains_completed))

for _ in range(FPS):
    schedule.append((-1, -1, chains_completed))

print(f"Total frames: {len(schedule)} ({len(schedule)/FPS:.1f}s at {FPS}fps)")

for fi, (c_anim, prog, c_done) in enumerate(schedule):
    render_frame(fi, c_anim, max(prog, 0), c_done)
    if (fi+1) % 30 == 0: print(f"  Frame {fi+1}/{len(schedule)}")

plt.close(fig)
print("Assembling GIF...")
images = [imageio.imread(p) for p in frame_paths]
out_path = f"results/{tag}_mcmc_comet.gif"
imageio.mimsave(out_path, images, fps=FPS, loop=0)
print(f"Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
for p in frame_paths: os.remove(p)
os.rmdir(tmp_dir)
print("Done!")
