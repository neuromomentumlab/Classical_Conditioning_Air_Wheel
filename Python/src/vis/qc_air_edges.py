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



import numpy as np
import matplotlib.pyplot as plt


def plot_event_qc_interactive(
    b,
    speed_key="speed_net_cms",
    plot_fs=50,
    title=None,
    show_event_lines=True,
):
    """
    Interactive QC plot for checking air_raw, air_bin, and detected events.

    Use with:
        %matplotlib widget

    Parameters
    ----------
    b : dict
        Behavior dictionary loaded from H5.

    speed_key : str
        "speed_net_cms" or "speed_path_cms"

    plot_fs : float
        Downsampled plotting frequency for display only.

    title : str
        Plot title.

    show_event_lines : bool
        If True, show Air_r and Air_f as vertical lines.
    """

    t = np.asarray(b["t"]).reshape(-1)
    fs = float(np.asarray(b["fs"]).squeeze())

    air_raw = np.asarray(b["air_raw"]).reshape(-1)
    air_bin = np.asarray(b["air_bin"]).reshape(-1).astype(float)
    speed = np.asarray(b[speed_key]).reshape(-1)

    Air_r = np.asarray(b["Air_r"]).reshape(-1).astype(int)
    Air_f = np.asarray(b["Air_f"]).reshape(-1).astype(int)

    # -----------------------------
    # Match lengths
    # -----------------------------
    n = min(len(t), len(air_raw), len(air_bin), len(speed))
    t = t[:n]
    air_raw = air_raw[:n]
    air_bin = air_bin[:n]
    speed = speed[:n]

    # -----------------------------
    # Downsample for plotting
    # -----------------------------
    step = max(1, int(round(fs / plot_fs)))

    t_ds = t[::step]
    air_raw_ds = air_raw[::step]
    air_bin_ds = air_bin[::step]
    speed_ds = speed[::step]

    # Normalize raw air for visual comparison
    raw_min = np.nanmin(air_raw_ds)
    raw_max = np.nanmax(air_raw_ds)

    air_raw_norm = (air_raw_ds - raw_min) / (raw_max - raw_min + 1e-9)

    # -----------------------------
    # Create figure
    # -----------------------------
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]}
    )

    ax0, ax1 = axes

    # -----------------------------
    # Air / LED signal
    # -----------------------------
    ax0.plot(t_ds, air_raw_norm, label="air_raw normalized", linewidth=1)
    ax0.plot(t_ds, air_bin_ds, label="air_bin", linewidth=1)

    if show_event_lines:
        for r in Air_r:
            if 0 <= r < len(t):
                ax0.axvline(t[r], color="green", linestyle="--", alpha=0.4)

        for f in Air_f:
            if 0 <= f < len(t):
                ax0.axvline(t[f], color="red", linestyle=":", alpha=0.4)

    ax0.set_ylabel("Air / LED")
    ax0.set_ylim(-0.1, 1.2)
    ax0.legend(loc="upper left")
    ax0.grid(alpha=0.2)

    # -----------------------------
    # Speed
    # -----------------------------
    ax1.plot(t_ds, speed_ds, linewidth=1, label=speed_key)

    # Shade ON periods
    for r, f in zip(Air_r, Air_f):
        if 0 <= r < len(t) and 0 <= f < len(t):
            ax1.axvspan(t[r], t[f], alpha=0.2)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (cm/s)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.2)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    return fig, axes