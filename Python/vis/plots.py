# vis/test_plots.py

import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# Air vs Speed plot
# =====================================================

def plot_air_speed(b, title=None, tlim=None, ax=None):
    """
    Plot running speed with air ON overlay.

    Parameters
    ----------
    b : dict
        Behavior struct from processing pipeline
    title : str or None
    tlim : (t0, t1) or None
        Time window in seconds
    ax : matplotlib axis or None

    Returns
    -------
    ax : matplotlib axis
    """

    t = b["t"]
    speed = b["speed_path_cms"]
    air = b["air_bin"].astype(float)

    # -----------------------------
    # Optional time window
    # -----------------------------
    if tlim is not None:
        mask = (t >= tlim[0]) & (t <= tlim[1])
        t = t[mask]
        speed = speed[mask]
        air = air[mask]

    # -----------------------------
    # Axis handling
    # -----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # -----------------------------
    # Plot speed
    # -----------------------------
    ax.plot(t, speed, linewidth=1.5, label="Speed (cm/s)")

    # -----------------------------
    # Air shading
    # -----------------------------
    ymax = np.nanmax(speed) if np.nanmax(speed) > 0 else 1

    ax.fill_between(
        t,
        0,
        air * ymax,
        alpha=0.25,
        label="Air ON"
    )

    # -----------------------------
    # Labels
    # -----------------------------
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")

    if title:
        ax.set_title(title)

    ax.legend()
    ax.grid(alpha=0.2)

    return ax


# =====================================================
# Quick convenience wrapper
# =====================================================

def quick_plot(results, animal, date, tlim=None):
    """
    Convenience function for fast inspection.
    """

    b = results[animal][date]

    title = f"{animal} {date}"

    plot_air_speed(b, title=title, tlim=tlim)
    plt.tight_layout()
    plt.show()
