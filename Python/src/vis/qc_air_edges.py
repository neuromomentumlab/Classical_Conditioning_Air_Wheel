# vis/qc_air_edges.py

import numpy as np
import matplotlib.pyplot as plt


def qc_plot_air_edges(b, max_seconds=120, overlay_speed=True, title=None):
    """
    QC plot for air trials:
      - air_raw
      - air_bin
      - Air_r (rising edges)
      - Air_f (falling edges)
      - optional speed overlay

    Parameters
    ----------
    b : dict
        behavior struct
    max_seconds : float
        plot first N seconds (set None to plot full session)
    overlay_speed : bool
        overlays speed (scaled) on air_raw axis
    title : str or None
    """

    fs = float(b["fs"])
    t = np.asarray(b["t"]).ravel()
    air_raw = np.asarray(b["air_raw"]).ravel()
    air_bin = np.asarray(b["air_bin"]).ravel()
    Air_r = np.asarray(b["Air_r"]).ravel().astype(int)
    Air_f = np.asarray(b["Air_f"]).ravel().astype(int)

    # decide plotting range
    if max_seconds is None:
        idx_end = len(t)
    else:
        idx_end = int(min(len(t), max_seconds * fs))

    tt = t[:idx_end]
    air_raw_plot = air_raw[:idx_end]
    air_bin_plot = air_bin[:idx_end]

    # filter edges to range
    Air_r_plot = Air_r[Air_r < idx_end]
    Air_f_plot = Air_f[Air_f < idx_end]

    fig, ax = plt.subplots(figsize=(12, 4))

    # air raw
    ax.plot(tt, air_raw_plot, linewidth=1, label="air_raw")

    # air bin (scaled to raw amplitude for visibility)
    raw_span = np.nanmax(air_raw_plot) - np.nanmin(air_raw_plot)
    if raw_span == 0:
        raw_span = 1.0
    air_bin_scaled = (air_bin_plot * raw_span) + np.nanmin(air_raw_plot)
    ax.plot(tt, air_bin_scaled, linewidth=1, label="air_bin (scaled)")

    # edges
    for x in t[Air_r_plot]:
        ax.axvline(x, linestyle="--", linewidth=1, label=None)
    for x in t[Air_f_plot]:
        ax.axvline(x, linestyle=":", linewidth=1, label=None)

    # optional speed overlay (scaled)
    if overlay_speed and "speed_net_cms" in b:
        speed = np.asarray(b["speed_net_cms"]).ravel()[:idx_end]
        # scale speed into air_raw range
        sp = speed.copy()
        sp = (sp - np.nanmin(sp)) / (np.nanmax(sp) - np.nanmin(sp) + 1e-12)
        sp = sp * raw_span + np.nanmin(air_raw_plot)
        ax.plot(tt, sp, linewidth=1, label="speed (scaled)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (scaled units)")
    if title is None:
        title = "QC: air_raw / air_bin with Air_r (--) and Air_f (:)"
    ax.set_title(title)

    # legend entries for edges (manual)
    handles, labels = ax.get_legend_handles_labels()
    # add dummy lines for edge types
    handles += [
        plt.Line2D([0], [0], linestyle="--", linewidth=1),
        plt.Line2D([0], [0], linestyle=":", linewidth=1),
    ]
    labels += ["Air_r (rising)", "Air_f (falling)"]
    ax.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    return fig, ax