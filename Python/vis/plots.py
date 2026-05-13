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


# =====================================================
# Quick convenience wrapper from H5 / cc_data
# =====================================================

def quick_plot_from_h5(cc_data, animal, date, tlim=None, verbose=True):
    """
    Convenience function for fast inspection using cc_data and H5.

    Loads only the variables needed for plot_air_speed:
        t
        speed_path_cms
        air_bin
    """

    import src.utils.pdata_io as pdio

    if animal not in cc_data:
        raise KeyError(f"Animal not found in cc_data: {animal}")

    if date not in cc_data[animal]:
        raise KeyError(f"Date not found for {animal}: {date}")

    keys = [
        "t",
        "speed_path_cms",
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