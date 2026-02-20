# proc/process_behavior_signals.py

import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path


# =====================================================
# Main entry
# =====================================================

def process_cc_data(cc_data,
                    animals=None,
                    phase="air_training",
                    verbose=True):
    """
    Iterate through cc_data and build behavior structs.

    Parameters
    ----------
    cc_data : dict
    animals : set or None
    phase : str
    verbose : bool

    Returns
    -------
    results : dict
        results[animal][date] = behavior dict
    """

    results = {}

    for animal_id, days in cc_data.items():

        if animals is not None and animal_id not in animals:
            continue

        for date, info in days.items():

            if info.get("phase") != phase:
                continue

            mat_file = info.get("recording")
            if mat_file is None or not Path(mat_file).exists():
                if verbose:
                    print(f"[WARN] Missing MAT: {animal_id} {date}")
                continue

            if verbose:
                print(f"Processing {animal_id} {date}")

            try:
                b = make_behavior_struct(mat_file)
            except Exception as e:
                print(f"[ERROR] {animal_id} {date}: {e}")
                continue

            results.setdefault(animal_id, {})[date] = b

    return results


# =====================================================
# Core builder (MATLAB equivalent)
# =====================================================

def make_behavior_struct(mat_file):
    """
    Build behavioral struct from raw DAQ MAT file.
    Mirrors MATLAB logic.
    """
    print(f"DEBUG loading: {mat_file}")
    # S = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    # S = load_mat_file(mat_file)
    S = load_mat_v73(mat_file)

    X = S["X"]
    fs = float(S["fs"])
    channel_names = list(S["channelNames"])

    # -------------------------------------------------
    # Identify channels (AIR Wheel format first)
    # -------------------------------------------------

    # Preferred fast path: Nx2 matrix
    if X.ndim == 2 and X.shape[1] == 2:
        air_raw = X[:, 0]
        enc_raw = X[:, 1]

    else:
        # fallback to channelNames (legacy support)
        channel_names = list(S["channelNames"])

        air_idx = _find_channel(channel_names, "Air")
        enc_idx = _find_channel(channel_names, "EncCount")

        air_raw = X[:, air_idx]
        enc_raw = X[:, enc_idx]

    b = {}

    # -------------------------------------------------
    # Timing
    # -------------------------------------------------
    b["fs"] = fs
    b["si"] = 1 / fs
    b["number_of_samples"] = X.shape[0]

    t = S["t"].reshape(-1)
    b["t"] = t
    b["tm"] = t / 60

    # -------------------------------------------------
    # Air signal
    # -------------------------------------------------
    b["air_raw"] = air_raw.reshape(-1)

    # binary air (match MATLAB behavior)
    b["air_bin"] = b["air_raw"] > 0.5

    # edges
    Air_r = find_rising_edge(b["air_raw"], 0.5, 2)
    Air_f = find_falling_edge(b["air_raw"], -0.5, 2)

    Air_r, Air_f, report = sanitize_air_edges(Air_r, Air_f)

    b["Air_r"] = Air_r
    b["Air_f"] = Air_f
    b["Air_edges_report"] = report

    # -------------------------------------------------
    # Encoder
    # -------------------------------------------------
    b["encoderCount"] = enc_raw.reshape(-1)
    b["countsPerRev"] = float(S["countsPerRev"])

    # -------------------------------------------------
    # Distance
    # -------------------------------------------------
    wheel_diameter_cm = 32
    wheel_circumference_cm = np.pi * wheel_diameter_cm
    cm_per_count = wheel_circumference_cm / b["countsPerRev"]

    enc = b["encoderCount"].astype(float)
    dEnc = np.diff(enc, prepend=enc[0])

    b["dist_net_cm"] = enc * cm_per_count
    b["dist_path_cm"] = np.cumsum(np.abs(dEnc)) * cm_per_count

    # -------------------------------------------------
    # Speed (windowed)
    # -------------------------------------------------
    win_ms = 100
    win = int(round(win_ms / 1000 * fs))

    dEnc_win = moving_sum(dEnc, win)

    b["speed_net_cms"] = (dEnc_win * cm_per_count) / (win / fs)
    b["speed_path_cms"] = (np.abs(dEnc_win) * cm_per_count) / (win / fs)
    b["speed_window_ms"] = win_ms

    # -------------------------------------------------
    # Direction bias
    # -------------------------------------------------
    b["bias"] = (
        (b["dist_net_cm"][-1] - b["dist_net_cm"][0]) /
        (b["dist_path_cm"][-1] - b["dist_path_cm"][0] + np.finfo(float).eps)
    )

    return b


# =====================================================
# Helpers
# =====================================================

def _find_channel(channel_names, name):
    for i, ch in enumerate(channel_names):
        if str(ch).lower() == name.lower():
            return i
    raise ValueError(f"Channel '{name}' not found")


def moving_sum(x, win):
    """MATLAB movsum equivalent (centered causal)."""
    kernel = np.ones(win)
    return np.convolve(x, kernel, mode="same")


# =====================================================
# Edge detection (faithful to MATLAB intent)
# =====================================================

def find_rising_edge(signal, thresh, min_sep):
    sig = signal > thresh
    edges = np.where(np.diff(np.concatenate([[0], sig.astype(int)])) == 1)[0]
    return edges


def find_falling_edge(signal, thresh, min_sep):
    sig = signal > thresh
    edges = np.where(np.diff(np.concatenate([sig.astype(int), [0]])) == -1)[0]
    return edges


def sanitize_air_edges(Air_r, Air_f):
    report = {
        "initial_r": len(Air_r),
        "initial_f": len(Air_f),
        "actions": []
    }

    Air_r = np.asarray(Air_r)
    Air_f = np.asarray(Air_f)

    if len(Air_r) and len(Air_f):
        if Air_r[0] > Air_f[0]:
            Air_f = Air_f[1:]
            report["actions"].append(
                "Removed first Air_f (started during Air ON)"
            )

    n = min(len(Air_r), len(Air_f))
    Air_r = Air_r[:n]
    Air_f = Air_f[:n]

    valid = Air_f > Air_r
    Air_r = Air_r[valid]
    Air_f = Air_f[valid]

    report["final_r"] = len(Air_r)
    report["final_f"] = len(Air_f)

    return Air_r, Air_f, report

def load_mat_file(mat_file):
    """
    Robust MATLAB loader supporting both pre-7.3 and v7.3 files.
    """
    print("DEBUG: using HDF5 loader")  # inside is_hdf5 branch

    # --- Detect if file is HDF5 (MAT v7.3) ---
    with open(mat_file, "rb") as f:
        header = f.read(8)

    is_hdf5 = header.startswith(b"\x89HDF")

    if is_hdf5:
        return load_mat_v73(mat_file)
    else:
        return sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)



def load_mat_v73(mat_file):
    """
    Reader for MATLAB v7.3 (HDF5) files.
    """

    out = {}

    with h5py.File(mat_file, "r") as f:

        def read_dataset(name):
            if name not in f:
                raise KeyError(f"{name} not found in MAT file")
            return np.array(f[name]).T

        out["X"] = read_dataset("X")
        out["t"] = read_dataset("t").reshape(-1)
        out["fs"] = float(np.array(f["fs"])[()])
        out["countsPerRev"] = float(np.array(f["countsPerRev"])[()])

        # --- channelNames (robust MATLAB cell reader) ---
        if "channelNames" in f:
            ch = f["channelNames"]
            names = []

            # MATLAB cell array → array of object references
            for ref in ch[0]:
                obj = f[ref]
                # decode uint16/uint8 char array to string
                names.append("".join(chr(c) for c in obj[:].flatten()))

            out["channelNames"] = names
        else:
            raise KeyError("channelNames missing")


    return out
