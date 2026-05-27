# vis/test_plots.py

import numpy as np
import matplotlib.pyplot as plt
import src.utils.pdata_io as pdio


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
    speed = b["speed_net_cms"]
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


# =====================================================
# Quick convenience wrapper from H5 / cc_data
# =====================================================

def quick_plot_from_h5(cc_data, animal, date, tlim=None, verbose=True):
    """
    Convenience function for fast inspection using cc_data and H5.

    Loads only the variables needed for plot_air_speed:
        t
        speed_net_cms
        air_bin
    """

    import src.utils.pdata_io as pdio

    if animal not in cc_data:
        raise KeyError(f"Animal not found in cc_data: {animal}")

    if date not in cc_data[animal]:
        raise KeyError(f"Date not found for {animal}: {date}")

    keys = [
        "t",
        "speed_net_cms",
        "air_bin",
    ]

    if verbose:
        print(f"[LOAD H5] {animal} {date}")
        print(f"Loading keys: {keys}")

    b = pdio.load_behavior_h5(
        animal,
        date,
        keys=keys
    )

    if b is None:
        raise FileNotFoundError(f"No behavior H5 file found for {animal} {date}")

    title = f"{animal} {date}"

    plot_air_speed(b, title=title, tlim=tlim)
    plt.tight_layout()
    plt.show()


# =====================================================
# Generic fast plotting function
# =====================================================

def quick_plot_keys_from_h5(
    animal,
    date,
    ykeys,
    xkey="t",
    tlim=None,
    root=None,
    ax=None,
    verbose=False,
):
    """
    Fast generic plotting directly from H5.

    Parameters
    ----------
    animal : str
    date : str

    ykeys : str or list
        Variables to plot

    xkey : str
        X-axis variable (default = "t")

    tlim : tuple or None
        (t0, t1) time limits

    root : optional H5 root path

    ax : matplotlib axis or None

    verbose : bool
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import src.utils.pdata_io as pdio

    # -------------------------------------------------
    # normalize keys
    # -------------------------------------------------

    if isinstance(ykeys, str):
        ykeys = [ykeys]

    keys_to_load = [xkey] + ykeys

    if verbose:
        print(f"[LOAD H5] {animal} {date}")
        print(f"Loading keys: {keys_to_load}")

    # -------------------------------------------------
    # load only required variables
    # -------------------------------------------------

    if root is None:
        b = pdio.load_behavior_h5(
            animal,
            date,
            keys=keys_to_load
        )
    else:
        b = pdio.load_behavior_h5(
            animal,
            date,
            keys=keys_to_load,
            root=root
        )

    if b is None:
        raise FileNotFoundError(
            f"No H5 file found for {animal} {date}"
        )

    # -------------------------------------------------
    # x variable
    # -------------------------------------------------

    x = np.asarray(b[xkey])

    # -------------------------------------------------
    # time window
    # -------------------------------------------------

    if tlim is not None:
        mask = (x >= tlim[0]) & (x <= tlim[1])
        x = x[mask]
    else:
        mask = slice(None)

    # -------------------------------------------------
    # axis
    # -------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # -------------------------------------------------
    # plot
    # -------------------------------------------------

    for key in ykeys:

        y = np.asarray(b[key])

        n = min(len(x), len(y))

        if tlim is not None:
            y = y[mask]

        ax.plot(
            x[:n],
            y[:n],
            linewidth=1
        )

    ax.grid(alpha=0.2)

    return ax

def interactive_plot_keys_from_h5(
    animal,
    date,
    ykeys,
    xkey="t",
    tlim=None,
    root=None,
    verbose=True,
):
    """
    Interactive MATLAB-like plot from H5.
    Supports zoom, pan, autoscale, hover, save image.

    Loads only requested keys.
    """

    import numpy as np
    import plotly.graph_objects as go
    import src.utils.pdata_io as pdio

    if isinstance(ykeys, str):
        ykeys = [ykeys]

    keys_to_load = list(dict.fromkeys([xkey] + ykeys))

    if verbose:
        print(f"[LOAD H5] {animal} {date}")
        print(f"Loading keys: {keys_to_load}")

    if root is None:
        b = pdio.load_behavior_h5(animal, date, keys=keys_to_load)
    else:
        b = pdio.load_behavior_h5(animal, date, keys=keys_to_load, root=root)

    if b is None:
        raise FileNotFoundError(f"No H5 file found for {animal} {date}")

    x = np.asarray(b[xkey]).ravel()

    if tlim is not None:
        mask = (x >= tlim[0]) & (x <= tlim[1])
    else:
        mask = np.ones_like(x, dtype=bool)

    fig = go.Figure()

    for key in ykeys:
        y = np.asarray(b[key]).ravel()

        n = min(len(x), len(y))
        x_use = x[:n]
        y_use = y[:n]
        mask_use = mask[:n]

        fig.add_trace(
            go.Scattergl(
                x=x_use[mask_use],
                y=y_use[mask_use],
                mode="lines",
                name=key
            )
        )

    fig.update_layout(
        xaxis_title=xkey,
        yaxis_title="value",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )

    fig.show()
    return fig

import numpy as np
import matplotlib.pyplot as plt


def quick_plot_speed_air(
    b,
    speed_key="speed_net_cms",
    plot_fs=50,
    tlim=None,
    title=None,
    figsize=(10, 4),
):
    """
    Fast speed + air plot.

    Downsamples speed for visualization and uses axvspan for air periods.
    """

    t = np.asarray(b["t"])
    speed = np.asarray(b[speed_key])
    air = np.asarray(b["air_bin"]).astype(int)

    fs = float(b["fs"])

    # -----------------------------
    # Optional time limit
    # -----------------------------
    if tlim is not None:
        keep = (t >= tlim[0]) & (t <= tlim[1])
        t_plot = t[keep]
        speed_plot = speed[keep]
        air_plot = air[keep]
        offset_idx = np.where(keep)[0][0]
    else:
        t_plot = t
        speed_plot = speed
        air_plot = air
        offset_idx = 0

    # -----------------------------
    # Downsample for plotting
    # -----------------------------
    step = max(1, int(round(fs / plot_fs)))

    t_ds = t_plot[::step]
    speed_ds = speed_plot[::step]

    # -----------------------------
    # Detect air on/off edges in plotted window
    # -----------------------------
    air_diff = np.diff(np.concatenate([[0], air_plot, [0]]))
    air_r = np.where(air_diff == 1)[0]
    air_f = np.where(air_diff == -1)[0] - 1

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t_ds, speed_ds, linewidth=1.0, label=speed_key)

    ymax = np.nanmax(speed_ds)
    ymin = np.nanmin(speed_ds)

    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1

    for r, f in zip(air_r, air_f):
        if r < len(t_plot) and f < len(t_plot):
            ax.axvspan(
                t_plot[r],
                t_plot[f],
                alpha=0.25
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title(title if title is not None else speed_key)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()

    return fig, ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_raw_speed_traces_around_anchor(
    epoch_matrix_df,
    animal,
    phase,
    anchor_name,
    phase_day_number_good=1,
    speed_key="speed_path_cms",
    pre_s=1.0,
    post_s=1.0,
    max_traces=None,
):
    """
    Plot raw speed traces around one anchor for one animal and one phase day.

    Gray lines = individual events/trials
    Black line = mean
    Dotted black lines = mean ± std
    """

    # --------------------------------------------------
    # Select rows for this animal / phase / anchor / day
    # --------------------------------------------------
    sub = epoch_matrix_df[
        (epoch_matrix_df["animal"] == animal) &
        (epoch_matrix_df["phase"] == phase) &
        (epoch_matrix_df["anchor_name"] == anchor_name) &
        (epoch_matrix_df["phase_day_number_good"] == phase_day_number_good)
    ].copy()

    if sub.empty:
        print(f"No rows found for {animal}, {phase}, {anchor_name}, day {phase_day_number_good}")
        return None

    # Get the date for this phase day
    dates = sorted(sub["date"].dropna().unique())

    if len(dates) == 0:
        print("No date found.")
        return None

    if len(dates) > 1:
        print("More than one date found. Using first:", dates[0])

    date = dates[0]

    sub = sub[sub["date"] == date].copy()

    # One anchor per event, not separate pre/post rows
    anchor_events = (
        sub[
            [
                "animal",
                "date",
                "phase",
                "event_number",
                "anchor_name",
                "anchor_time_s",
            ]
        ]
        .drop_duplicates()
        .sort_values("event_number")
        .reset_index(drop=True)
    )

    if max_traces is not None:
        anchor_events = anchor_events.iloc[:max_traces].copy()

    # --------------------------------------------------
    # Load raw speed data
    # --------------------------------------------------
    b = pdio.load_behavior_h5(
        animal,
        date,
        keys=[speed_key, "fs"]
    )

    speed = np.asarray(b[speed_key]).reshape(-1)
    fs = float(np.asarray(b["fs"]).squeeze())

    n_pre = int(round(pre_s * fs))
    n_post = int(round(post_s * fs))

    expected_n = n_pre + n_post

    traces = []

    for _, ev in anchor_events.iterrows():

        anchor_time = ev["anchor_time_s"]

        if not np.isfinite(anchor_time):
            continue

        anchor_idx = int(round(anchor_time * fs))

        start_idx = anchor_idx - n_pre
        end_idx = anchor_idx + n_post

        if start_idx < 0:
            continue

        if end_idx > len(speed):
            continue

        trace = speed[start_idx:end_idx]

        if len(trace) == expected_n:
            traces.append(trace)

    if len(traces) == 0:
        print("No valid traces extracted.")
        return None

    traces = np.vstack(traces)

    t = np.arange(-n_pre, n_post) / fs

    mean_trace = np.nanmean(traces, axis=0)
    std_trace = np.nanstd(traces, axis=0)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    for tr in traces:
        ax.plot(t, tr, color="0.75", linewidth=0.8, alpha=0.7)

    ax.plot(
        t,
        mean_trace,
        color="black",
        linewidth=2.5,
        label="Mean"
    )

    ax.plot(
        t,
        mean_trace + std_trace,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Mean ± SD"
    )

    ax.plot(
        t,
        mean_trace - std_trace,
        color="black",
        linestyle=":",
        linewidth=1.5
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(f"Time from {anchor_name} (s)")
    ax.set_ylabel("Speed path (cm/s)")
    ax.set_title(
        f"{animal} | {phase} | {anchor_name} | day {phase_day_number_good} | {date}\n"
        f"n events = {traces.shape[0]}"
    )

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "traces": traces,
        "time": t,
        "animal": animal,
        "date": date,
        "phase": phase,
        "anchor_name": anchor_name,
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_event_aligned_speed_traces_one_animal(
    epoch_matrix_df,
    animal,
    phase,
    anchor_name,
    phase_day_number_good=1,
    speed_key="speed_path_cms",
    pre_s=1.0,
    post_s=1.0,
):
    """
    Extract all raw speed traces for one animal on one phase-day around one anchor.
    Returns:
        traces: array (n_events, n_timepoints)
        t: time vector
        date_used: date corresponding to the requested phase_day_number_good
    """

    sub = epoch_matrix_df[
        (epoch_matrix_df["animal"] == animal) &
        (epoch_matrix_df["phase"] == phase) &
        (epoch_matrix_df["anchor_name"] == anchor_name) &
        (epoch_matrix_df["phase_day_number_good"] == phase_day_number_good)
    ].copy()

    if sub.empty:
        return None, None, None

    dates = sorted(sub["date"].dropna().unique())
    if len(dates) == 0:
        return None, None, None

    date_used = dates[0]
    sub = sub[sub["date"] == date_used].copy()

    anchor_events = (
        sub[
            ["event_number", "anchor_time_s"]
        ]
        .drop_duplicates()
        .sort_values("event_number")
        .reset_index(drop=True)
    )

    if anchor_events.empty:
        return None, None, date_used

    b = pdio.load_behavior_h5(
        animal,
        date_used,
        keys=[speed_key, "fs"]
    )

    speed = np.asarray(b[speed_key]).reshape(-1)
    fs = float(np.asarray(b["fs"]).squeeze())

    n_pre = int(round(pre_s * fs))
    n_post = int(round(post_s * fs))
    n_expected = n_pre + n_post

    traces = []

    for _, ev in anchor_events.iterrows():
        anchor_time = ev["anchor_time_s"]

        if not np.isfinite(anchor_time):
            continue

        anchor_idx = int(round(anchor_time * fs))
        start_idx = anchor_idx - n_pre
        end_idx = anchor_idx + n_post

        if start_idx < 0 or end_idx > len(speed):
            continue

        tr = speed[start_idx:end_idx]

        if len(tr) == n_expected:
            traces.append(tr)

    if len(traces) == 0:
        return None, None, date_used

    traces = np.vstack(traces)
    t = np.arange(-n_pre, n_post) / fs

    return traces, t, date_used


def plot_group_animal_mean_traces(
    epoch_matrix_df,
    animals,
    phase,
    anchor_name,
    phase_day_number_good=1,
    speed_key="speed_path_cms",
    pre_s=1.0,
    post_s=1.0,
    title=None,
    use_shaded_sem=True,
):
    """
    Plot animal-mean traces as gray lines, plus group mean ± SEM.

    Gray lines: mean trace for each animal
    Black line: grand mean across animals
    Shaded area or dotted lines: SEM across animal means
    """

    animal_mean_traces = []
    used_animals = []
    used_dates = {}

    t_ref = None

    for animal in animals:
        traces, t, date_used = extract_event_aligned_speed_traces_one_animal(
            epoch_matrix_df=epoch_matrix_df,
            animal=animal,
            phase=phase,
            anchor_name=anchor_name,
            phase_day_number_good=phase_day_number_good,
            speed_key=speed_key,
            pre_s=pre_s,
            post_s=post_s,
        )

        if traces is None or t is None:
            print(f"[SKIP] {animal}: no usable traces")
            continue

        animal_mean = np.nanmean(traces, axis=0)

        animal_mean_traces.append(animal_mean)
        used_animals.append(animal)
        used_dates[animal] = date_used

        if t_ref is None:
            t_ref = t

    if len(animal_mean_traces) == 0:
        print("No animal traces available.")
        return None

    animal_mean_traces = np.vstack(animal_mean_traces)

    grand_mean = np.nanmean(animal_mean_traces, axis=0)
    sem = np.nanstd(animal_mean_traces, axis=0, ddof=1) / np.sqrt(animal_mean_traces.shape[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # plot each animal mean as gray
    for i, animal in enumerate(used_animals):
        ax.plot(
            t_ref,
            animal_mean_traces[i],
            color="0.6",
            linewidth=1.5,
            alpha=0.9,
        )

    # plot group mean
    ax.plot(
        t_ref,
        grand_mean,
        color="black",
        linewidth=2.5,
        label="Group mean"
    )

    # SEM
    if use_shaded_sem:
        ax.fill_between(
            t_ref,
            grand_mean - sem,
            grand_mean + sem,
            color="black",
            alpha=0.15,
            label="Mean ± SEM"
        )
    else:
        ax.plot(
            t_ref,
            grand_mean + sem,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Mean ± SEM"
        )
        ax.plot(
            t_ref,
            grand_mean - sem,
            color="black",
            linestyle=":",
            linewidth=1.5
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(f"Time from {anchor_name} (s)")
    ax.set_ylabel("Speed path (cm/s)")

    if title is None:
        title = f"{phase} | {anchor_name} | phase_day_number_good = {phase_day_number_good}"

    ax.set_title(title + f"\nAnimal means (n = {len(used_animals)})")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "time": t_ref,
        "animal_mean_traces": animal_mean_traces,
        "grand_mean": grand_mean,
        "sem": sem,
        "used_animals": used_animals,
        "used_dates": used_dates,
    }

def plot_habituation_forward_backward(hab_session_df):
    """
    Plot forward and backward fractions across habituation days.
    """

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Individual animals
    for animal, sub in hab_session_df.groupby("animal"):
        sub = sub.sort_values("normalized_phase_day_good")

        ax.plot(
            sub["normalized_phase_day_good"],
            sub["frac_forward"],
            color="0.65",
            linewidth=1.2,
            alpha=0.8
        )

        ax.plot(
            sub["normalized_phase_day_good"],
            sub["frac_backward"],
            color="0.65",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

    # Group mean by normalized day binned into actual good day numbers
    summary = (
        hab_session_df
        .groupby("phase_day_number_good")
        .agg(
            normalized_day=("normalized_phase_day_good", "mean"),
            mean_forward=("frac_forward", "mean"),
            sem_forward=("frac_forward", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            mean_backward=("frac_backward", "mean"),
            sem_backward=("frac_backward", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            n_animals=("animal", "nunique"),
        )
        .reset_index()
    )

    ax.errorbar(
        summary["normalized_day"],
        summary["mean_forward"],
        yerr=summary["sem_forward"],
        color="black",
        linewidth=2.5,
        marker="o",
        capsize=3,
        label="Forward"
    )

    ax.errorbar(
        summary["normalized_day"],
        summary["mean_backward"],
        yerr=summary["sem_backward"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        capsize=3,
        label="Backward"
    )

    ax.set_xlabel("Normalized habituation day")
    ax.set_ylabel("Fraction of session")
    ax.set_title("Forward and backward running across habituation")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return fig, ax, summary


import matplotlib.pyplot as plt

def plot_hab_summary(summary_df, y_col, sem_col, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))

    x_order = ["early", "middle", "late"]
    x = np.arange(len(x_order))

    for day_bin, sub in summary_df.groupby("phase_day_bin", observed=False):
        sub = sub.set_index("session_time_bin").loc[x_order].reset_index()

        ax.errorbar(
            x,
            sub[y_col],
            yerr=sub[sem_col],
            marker="o",
            capsize=4,
            label=f"{day_bin} days"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Early session", "Middle session", "Late session"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# speed_summary = (
#     hab_led_rm_df
#     .groupby(["phase_day_bin", "session_time_bin"], observed=False)
#     .agg(
#         mean_speed=("mean_speed_path_cms", "mean"),
#         sem_speed=("mean_speed_path_cms", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
#         n=("animal", "nunique"),
#     )
#     .reset_index()
# )

# stationary_summary = (
#     hab_led_rm_df
#     .groupby(["phase_day_bin", "session_time_bin"], observed=False)
#     .agg(
#         mean_stationary=("frac_stationary", "mean"),
#         sem_stationary=("frac_stationary", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
#         n=("animal", "nunique"),
#     )
#     .reset_index()
# )

# plot_hab_summary(
#     speed_summary,
#     y_col="mean_speed",
#     sem_col="sem_speed",
#     ylabel="Mean path speed (cm/s)",
#     title="Speed across habituation"
# )

# plot_hab_summary(
#     stationary_summary,
#     y_col="mean_stationary",
#     sem_col="sem_stationary",
#     ylabel="Stationary fraction",
#     title="Stationary behavior across habituation"
# )