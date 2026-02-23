# vis/qc_encoder_distance.py

import numpy as np
import matplotlib.pyplot as plt


def qc_plot_encoder_distance(b, max_seconds=120, title=None):
    """
    QC plot for encoder and distance signals.

    Plots:
      1) encoderCount
      2) dist_net_cm
      3) dist_path_cm

    Parameters
    ----------
    b : dict
        behavior struct
    max_seconds : float or None
        plot first N seconds (None = full session)
    title : str or None
    """

    fs = float(b["fs"])
    t = np.asarray(b["t"]).ravel()

    enc = np.asarray(b["encoderCount"]).ravel()
    dnet = np.asarray(b["dist_net_cm"]).ravel()
    dpath = np.asarray(b["dist_path_cm"]).ravel()

    # -----------------------------
    # time window
    # -----------------------------
    if max_seconds is None:
        idx_end = len(t)
    else:
        idx_end = int(min(len(t), max_seconds * fs))

    tt = t[:idx_end]

    enc = enc[:idx_end]
    dnet = dnet[:idx_end]
    dpath = dpath[:idx_end]

    # -----------------------------
    # figure
    # -----------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # Encoder
    axes[0].plot(tt, enc, linewidth=1)
    axes[0].set_ylabel("Encoder Count")
    axes[0].set_title("Encoder")

    # Net distance
    axes[1].plot(tt, dnet, linewidth=1)
    axes[1].set_ylabel("Net Distance (cm)")
    axes[1].set_title("Net Distance")

    # Path distance
    axes[2].plot(tt, dpath, linewidth=1)
    axes[2].set_ylabel("Path Distance (cm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Path Distance (cumulative)")

    if title is None:
        title = "QC: Encoder and Distance Signals"

    fig.suptitle(title)

    plt.tight_layout()
    return fig, axes