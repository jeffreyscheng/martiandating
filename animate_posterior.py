import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import gaussian_kde


def animate_posterior_densities(
        fs_np, ll, *,
        duration=10, fps=12,
        filename="posterior_evolution.gif",
        n_grid=200):
    """
    Animated GIF of weighted posterior densities for Ea, logD0_r2, φ.

    duration : seconds of video  →  n_frames_target = fps * duration
    fps      : frames per second
    """

    # ------------------- set-up ------------------------------------------
    params  = [('Ea', r'$E_a$ (kJ mol$^{-1}$)'),
               ('logD0_r2', r'$\log(D_0/\rho^2)$'),
               ('phi', r'$\varphi$')]
    
    # Get number of domains from the data shape
    num_domains = fs_np['Ea'].shape[-1]  # should be 2 from your k=2 setup
    colors = [plt.cm.tab10(i) for i in range(num_domains)]
    T, C    = ll.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax in axes:
        ax.set_ylabel("Posterior density")       # labels before tight_layout
        ax.grid(True, alpha=0.3)

    # reserve room: left=6 %, bottom=12 %, right=98 %, top=92 %
    fig.tight_layout(rect=[0.06, 0.12, 0.98, 0.92])

    # create a single title artist; we'll just rewrite its text
    title = fig.suptitle("", fontsize=14)

    lines, fills = [], [None] * (3 * num_domains)          # objects (3 params × k domains)
    for ax in axes:
        for _ in range(num_domains):
            (l,) = ax.plot([], [], lw=2)
            lines.append(l)

    # x-axis ranges (constant)
    x_ranges = []
    for key, _ in params:
        vals = fs_np[key].ravel()                    # all time/chain/domain
        xr   = np.linspace(vals.min() - 0.1*np.ptp(vals),
                           vals.max() + 0.1*np.ptp(vals),
                           n_grid)
        x_ranges.append(xr)
    for ax, (_, lab) in zip(axes, params):
        ax.set_xlabel(lab)

    # frame selection (thinning if needed)
    n_frames_target = int(fps * duration)
    if T > n_frames_target:
        spacing   = math.floor(T / n_frames_target)
        frames_it = range(0, T, spacing)
    else:
        frames_it = range(T)

    # Helper function to create weighted KDE
    def weighted_kde(data, weights, x_eval):
        """Create KDE with weights, handling numerical issues."""
        # Check for and remove NaN/inf values in both data and weights
        data_finite = np.isfinite(data)
        weights_finite = np.isfinite(weights)
        valid_mask = data_finite & weights_finite
        
        data_clean = data[valid_mask]
        weights_clean = weights[valid_mask]
        
        if len(data_clean) == 0:
            return np.zeros_like(x_eval)
        
        # Remove zero weights (can cause numerical issues)
        nonzero_mask = weights_clean > 1e-12
        data_clean = data_clean[nonzero_mask]
        weights_clean = weights_clean[nonzero_mask]
        
        if len(data_clean) == 0:
            return np.zeros_like(x_eval)
            
        # Normalize weights
        weights_clean = weights_clean / weights_clean.sum()
        
        # Additional check for finite weights after normalization
        if not np.all(np.isfinite(weights_clean)):
            return np.zeros_like(x_eval)
        
        try:
            kde = gaussian_kde(data_clean, weights=weights_clean)
            return kde(x_eval)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Handle cases where KDE fails (e.g., all data points are identical)
            print(f"KDE failed: {e}")
            return np.zeros_like(x_eval)

    # ------------------- pre-compute y-limits ---------------------------
    y_max = [0.0, 0.0, 0.0]                            # per-parameter
    for t in frames_it:
        w = np.exp(ll[t] - ll[t].max())
        w /= w.sum()
        for p_idx, (key, _) in enumerate(params):
            vals_now = fs_np[key][t]                  # (chains, domains)
            for d in range(num_domains):
                dens = weighted_kde(vals_now[:, d], w, x_ranges[p_idx])
                y_max[p_idx] = max(y_max[p_idx], dens.max())

    for ax, ymax in zip(axes, y_max):
        ax.set_ylim(0, 1.05 * ymax)                   # fixed y-scale

    # ------------------- animation update ------------------------------
    def update(t_idx):
        w = np.exp(ll[t_idx] - ll[t_idx].max())
        w /= w.sum()

        obj = 0
        for (key, _), xr, ax in zip(params, x_ranges, axes):
            vals_now = fs_np[key][t_idx]              # (chains, domains)
            for d in range(num_domains):
                dens = weighted_kde(vals_now[:, d], w, xr)

                lines[obj].set_data(xr, dens)
                lines[obj].set_color(colors[d])

                if fills[obj] is not None:
                    fills[obj].remove()
                fills[obj] = ax.fill_between(xr, dens, alpha=0.3,
                                             color=colors[d])
                obj += 1

        title.set_text(f"Posterior density – iteration {t_idx + 1}/{T}")

        return lines + fills

    # ------------------- build & save -----------------------------------
    anim = FuncAnimation(fig, update, frames=frames_it, blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF saved to '{filename}'")


if __name__ == "__main__":
    # Example usage:
    # animate_posterior_densities(fs_np, ll, duration=5, fps=12, filename="posterior_evolution.gif")
    pass 