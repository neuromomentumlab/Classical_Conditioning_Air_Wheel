# proc/process_behavior_signals.py

import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path

from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import src.utils.pdata_io as pdio


@dataclass
class BehaviorStateConfig:
    analysis_hz: float = 100.0
    move_on_cms: float = 1.0
    move_off_cms: float = 0.5
    min_move_s: float = 0.10
    join_move_gaps_s: float = 0.30
    quiet_cms: float = 0.25
    min_quiet_s: float = 1.00
    join_quiet_gaps_s: float = 0.10
    pre_air_max_s: float = 5.0
    immediate_post_s: float = 1.0
    sustain_fraction: float = 0.60
    min_sustain_s: float = 0.20


# =====================================================
# Main entry
# =====================================================

# =====================================================
# Main entry (UPDATED)
# =====================================================

from src.utils.pdata_io import (
    save_behavior_npz,
    load_behavior_npz,
    save_behavior_h5,
    load_behavior_h5,
)

def process_cc_data(
    cc_data,
    animals=None,
    phase="air_training",
    dates=None,              # NEW: explicit list/set of dates
    date_min=None,           # NEW: inclusive lower bound (YYYY_MM_DD)
    date_max=None,           # NEW: inclusive upper bound (YYYY_MM_DD)
    overwrite=False,
    save=True,
    verbose=True,
):
    """
    Iterate through cc_data and build behavior structs.

    Filters:
      - animals: set/list of animal IDs
      - phase: session phase
      - dates: explicit iterable of dates to include (preferred for QC)
      - date_min/date_max: inclusive range filter (YYYY_MM_DD)

    Returns
    -------
    results : dict
        results[animal][date] = behavior dict
    """

    # normalize date filter
    if dates is not None:
        dates = set(dates)

    def date_ok(d):
        if dates is not None and d not in dates:
            return False
        if date_min is not None and d < date_min:
            return False
        if date_max is not None and d > date_max:
            return False
        return True

    results = {}

    for animal_id, days in cc_data.items():

        if animals is not None and animal_id not in animals:
            continue

        for date, info in days.items():

            if info.get("phase") != phase:
                continue

            if not date_ok(date):
                continue

            # cache
            if not overwrite:
                # cached = load_behavior_npz(animal_id, date)
                # cached = load_behavior_h5(animal_id, date)
                cached = {}
                if cached is not None:
                    if verbose:
                        print(f"[CACHE] {animal_id} {date}")
                    results.setdefault(animal_id, {})[date] = cached
                    continue

            mat_file = info.get("recording")
            if mat_file is None or not Path(mat_file).exists():
                if verbose:
                    print(f"[WARN] Missing MAT: {animal_id} {date}")
                continue

            if verbose:
                print(f"[PROCESS] {animal_id} {date}")

            try:
                b = make_behavior_struct(mat_file)
            except Exception as e:
                print(f"[ERROR] {animal_id} {date}: {e}")
                continue

            if save:
                # save_behavior_npz(b, animal_id, date)
                save_behavior_h5(b, animal_id, date)
                # return

            # results.setdefault(animal_id, {})[date] = b

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

    # Binary air
    b["air_bin"] = (b["air_raw"] > 0.5).astype(int)

    # Edge detection from binary signal
    air_diff = np.diff(np.concatenate([[0], b["air_bin"], [0]]))

    Air_r = np.where(air_diff == 1)[0]
    Air_f = np.where(air_diff == -1)[0] - 1

    # Sanitize
    # Air_r, Air_f, report = sanitize_air_edges(Air_r, Air_f)

    b["Air_r"] = np.asarray(Air_r).ravel().astype(int)
    b["Air_f"] = np.asarray(Air_f).ravel().astype(int)
    # b["Air_edges_report"] = report

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

    b = compute_forward_progress_per_trial(b)

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

def sanitize_air_edges(Air_r, Air_f, drop_last=True):
    """
    Ensure valid air/LED epochs.

    Rules
    -----
    1. Remove leading partial ON epoch if recording starts during ON
    2. Enforce one-to-one pairing
    3. Remove non-positive duration trials
    4. Optionally drop the final event/trial
    """

    Air_r = np.asarray(Air_r).ravel()
    Air_f = np.asarray(Air_f).ravel()

    report = {
        "initial_r": len(Air_r),
        "initial_f": len(Air_f),
        "actions": []
    }

    # -------------------------------------------------
    # Rule 1: recording started during ON
    # -------------------------------------------------
    if len(Air_r) and len(Air_f):
        if Air_r[0] == 0:
            Air_r = Air_r[1:]
            Air_f = Air_f[1:]
            report["actions"].append(
                "Removed first epoch (recording started during ON)"
            )

    # -------------------------------------------------
    # Rule 2: enforce equal lengths
    # -------------------------------------------------
    n = min(len(Air_r), len(Air_f))
    Air_r = Air_r[:n]
    Air_f = Air_f[:n]

    # -------------------------------------------------
    # Rule 3: remove invalid durations
    # -------------------------------------------------
    valid = Air_f > Air_r
    if not np.all(valid):
        report["actions"].append("Removed non-positive duration trials")

    Air_r = Air_r[valid]
    Air_f = Air_f[valid]

    # -------------------------------------------------
    # Rule 4: drop final event/trial
    # -------------------------------------------------
    if drop_last and len(Air_r) > 0:
        Air_r = Air_r[:-1]
        Air_f = Air_f[:-1]
        report["actions"].append("Dropped last event/trial")

    report["final_r"] = len(Air_r)
    report["final_f"] = len(Air_f)

    return Air_r.astype(int), Air_f.astype(int), report


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

def compute_forward_progress_per_trial(b):
    """
    Compute true Arduino-style forward progress per air trial.
    """

    enc = b["encoderCount"].astype(float)
    fs  = b["fs"]

    # encoder increments
    dEnc = np.diff(enc, prepend=enc[0])

    # Arduino behavior: clamp negatives to zero
    dEnc_forward = np.maximum(dEnc, 0)

    # convert to cm
    wheel_diameter_cm = 32
    wheel_circumference_cm = np.pi * wheel_diameter_cm
    cm_per_count = wheel_circumference_cm / b["countsPerRev"]

    forward_cm = dEnc_forward * cm_per_count

    Air_r = np.asarray(b["Air_r"])
    Air_f = np.asarray(b["Air_f"])

    progress = []

    for r, f in zip(Air_r, Air_f):
        prog = np.sum(forward_cm[r:f])
        progress.append(prog)

    b["trial_forward_cm"] = np.array(progress)

    return b



def _runs(mask):
    """Contiguous True intervals, represented as [start, end)."""
    mask = np.asarray(mask, dtype=bool)
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _clean_mask(mask, min_true=1, max_false_gap=0):
    """Join short gaps, then remove short True episodes."""
    out = np.asarray(mask, dtype=bool).copy()

    for start, end in _runs(~out):
        if start > 0 and end < len(out) and end - start <= max_false_gap:
            out[start:end] = True

    for start, end in _runs(out):
        if end - start < min_true:
            out[start:end] = False

    return out


def _hysteresis(speed, on_threshold, off_threshold):
    if off_threshold > on_threshold:
        raise ValueError("move_off_cms must be <= move_on_cms")

    out = np.zeros(len(speed), dtype=bool)
    moving = False

    for i, value in enumerate(speed):
        if not moving and value >= on_threshold:
            moving = True
        elif moving and value <= off_threshold:
            moving = False
        out[i] = moving

    return out


def _block_mean(x, block_size):
    x = np.asarray(x)
    n = len(x) // block_size
    return x[:n * block_size].reshape(n, block_size).mean(axis=1)


def _state_row(
    animal, date, phase, trial, state,
    start, end, air_on, block_size, fs,
    bout_number=np.nan, note="",
):
    if start is None or end is None or end <= start:
        return None

    onset_sample = int(start * block_size)
    offset_sample = int(end * block_size)

    return {
        "animal": animal,
        "date": date,
        "phase": phase,
        "trial": trial,
        "state": state,
        "bout_number": bout_number,
        "onset_sample": onset_sample,
        "offset_sample": offset_sample,
        "onset_s": onset_sample / fs,
        "offset_s": offset_sample / fs,
        "duration_s": (offset_sample - onset_sample) / fs,
        "onset_relative_air_s": (start - air_on) * block_size / fs,
        "offset_relative_air_s": (end - air_on) * block_size / fs,
        "note": note,
    }


def identify_behavior_states(
    animal,
    date,
    cc_data=None,
    state_config=None,
    root=None,
):
    """
    Load behavior_v1.h5 through pdio and identify:
      pre-air immobility
      air-on immobility
      locomotor initiation
      sustained locomotion
      deceleration
      interbout immobility
      immediate post-run
      longer quiet immobility

    Returns
    -------
    trial_summary_df, state_episodes_df, state_signals
    """
    cfg = state_config or BehaviorStateConfig()

    keys = [
        "Air_r", "Air_f", "fs", "number_of_samples",
        "speed_path_cms", "speed_net_cms", "trial_forward_cm",
    ]

    kwargs = {"keys": keys}
    if root is not None:
        kwargs["root"] = root

    b = pdio.load_behavior_h5(animal, date, **kwargs)

    if b is None:
        raise FileNotFoundError(f"No behavior H5 for {animal}, {date}")

    missing = [key for key in keys if key not in b]
    if missing:
        raise KeyError(f"Missing H5 variables: {missing}")

    phase = "unknown"
    if cc_data is not None:
        phase = cc_data.get(animal, {}).get(date, {}).get("phase", "unknown")

    fs = float(np.asarray(b["fs"]).squeeze())
    n_samples = int(np.asarray(b["number_of_samples"]).squeeze())

    air_on_samples = np.asarray(b["Air_r"], dtype=np.int64)
    air_off_samples = np.asarray(b["Air_f"], dtype=np.int64) + 1
    trial_forward_cm = np.asarray(b["trial_forward_cm"], dtype=float)

    speed_path = np.asarray(b["speed_path_cms"], dtype=float)
    speed_net = np.asarray(b["speed_net_cms"], dtype=float)

    if len(speed_path) != n_samples:
        raise ValueError("speed_path_cms length != number_of_samples")

    block_size = max(1, round(fs / cfg.analysis_hz))
    analysis_hz = fs / block_size

    speed_path_ds = _block_mean(speed_path, block_size)
    speed_net_ds = _block_mean(speed_net, block_size)
    n_ds = len(speed_path_ds)

    moving = _hysteresis(
        speed_path_ds,
        cfg.move_on_cms,
        cfg.move_off_cms,
    )
    moving = _clean_mask(
        moving,
        min_true=max(1, round(cfg.min_move_s * analysis_hz)),
        max_false_gap=round(cfg.join_move_gaps_s * analysis_hz),
    )

    quiet = _clean_mask(
        speed_path_ds <= cfg.quiet_cms,
        min_true=max(1, round(cfg.min_quiet_s * analysis_hz)),
        max_false_gap=round(cfg.join_quiet_gaps_s * analysis_hz),
    )

    move_runs = _runs(moving)
    quiet_runs = _runs(quiet)

    air_on_ds = (air_on_samples // block_size).astype(int)
    air_off_ds = np.ceil(air_off_samples / block_size).astype(int)

    episode_rows = []
    summary_rows = []

    for trial, (air_on, air_off) in enumerate(
        zip(air_on_ds, air_off_ds), start=1
    ):
        previous_air_off = air_off_ds[trial - 2] if trial > 1 else 0
        next_air_on = air_on_ds[trial] if trial < len(air_on_ds) else n_ds

        moving_at_air_on = bool(moving[min(air_on, n_ds - 1)])

        # 1. Pre-air immobility
        pre_limit = max(
            previous_air_off,
            air_on - round(cfg.pre_air_max_s * analysis_hz),
        )
        pre_candidates = [
            (start, end)
            for start, end in quiet_runs
            if start < air_on and end >= air_on
        ]
        pre_start = (
            max(pre_candidates[-1][0], pre_limit)
            if pre_candidates else air_on
        )
        pre_end = air_on

        row = _state_row(
            animal, date, phase, trial, "pre_air_immobility",
            pre_start, pre_end, air_on, block_size, fs,
        )
        if row:
            episode_rows.append(row)

        # Movement bouts intersecting this air trial
        bouts = [
            (start, end)
            for start, end in move_runs
            if end > air_on and start < air_off
        ]

        first_move = bouts[0][0] if bouts else None
        air_immobile_end = (
            min(max(first_move, air_on), air_off)
            if first_move is not None else air_off
        )

        # 2. Air on, but still immobile
        row = _state_row(
            animal, date, phase, trial, "air_on_immobility",
            air_on, air_immobile_end, air_on, block_size, fs,
        )
        if row:
            episode_rows.append(row)

        state_totals = {
            "locomotor_initiation": 0.0,
            "sustained_locomotion": 0.0,
            "deceleration": 0.0,
        }

        # 3-5. Split each locomotor bout
        for bout_number, (bout_start, bout_end) in enumerate(bouts, start=1):
            bout_speed = speed_path_ds[bout_start:bout_end]
            cruise = float(np.percentile(bout_speed, 75))
            sustain_threshold = max(
                cfg.move_on_cms,
                cfg.sustain_fraction * cruise,
            )

            sustain_mask = _clean_mask(
                bout_speed >= sustain_threshold,
                min_true=max(1, round(cfg.min_sustain_s * analysis_hz)),
                max_false_gap=round(0.20 * analysis_hz),
            )
            sustain_runs = _runs(sustain_mask)

            if sustain_runs:
                sustained_start = bout_start + sustain_runs[0][0]
                sustained_end = bout_start + sustain_runs[-1][1]
                note = "ok"
            else:
                peak = bout_start + int(np.argmax(bout_speed)) + 1
                sustained_start = peak
                sustained_end = peak
                note = "no_clear_sustained_phase"

            intervals = {
                "locomotor_initiation": (bout_start, sustained_start),
                "sustained_locomotion": (sustained_start, sustained_end),
                "deceleration": (sustained_end, bout_end),
            }

            for state, (start, end) in intervals.items():
                row = _state_row(
                    animal, date, phase, trial, state,
                    start, end, air_on, block_size, fs,
                    bout_number=bout_number,
                    note=note,
                )
                if row:
                    episode_rows.append(row)
                    state_totals[state] += row["duration_s"]

        # Immobility pauses between multiple bouts
        for i in range(len(bouts) - 1):
            row = _state_row(
                animal, date, phase, trial, "interbout_immobility",
                bouts[i][1], bouts[i + 1][0],
                air_on, block_size, fs,
                bout_number=i + 1,
            )
            if row:
                episode_rows.append(row)

        # 6-7. Post-run states
        if bouts:
            final_move_offset = bouts[-1][1]
            immediate_end = min(
                next_air_on,
                final_move_offset + round(cfg.immediate_post_s * analysis_hz),
            )

            row = _state_row(
                animal, date, phase, trial, "immediate_post_run",
                final_move_offset, immediate_end,
                air_on, block_size, fs,
            )
            if row:
                episode_rows.append(row)

            quiet_candidates = [
                (start, end)
                for start, end in quiet_runs
                if end > immediate_end and start < next_air_on
            ]

            if quiet_candidates:
                long_quiet_start = max(quiet_candidates[0][0], immediate_end)
                long_quiet_end = min(quiet_candidates[0][1], next_air_on)
            else:
                long_quiet_start = None
                long_quiet_end = None

            row = _state_row(
                animal, date, phase, trial, "longer_quiet_immobility",
                long_quiet_start, long_quiet_end,
                air_on, block_size, fs,
            )
            if row:
                episode_rows.append(row)
        else:
            final_move_offset = None
            immediate_end = None
            long_quiet_start = None
            long_quiet_end = None

        total_moving_during_air_s = (
            sum(
                max(0, min(end, air_off) - max(start, air_on))
                for start, end in bouts
            )
            * block_size / fs
        )

        summary_rows.append({
            "animal": animal,
            "date": date,
            "phase": phase,
            "trial": trial,
            "air_on_s": air_on_samples[trial - 1] / fs,
            "air_off_s": air_off_samples[trial - 1] / fs,
            "air_duration_s": (
                air_off_samples[trial - 1] - air_on_samples[trial - 1]
            ) / fs,
            "trial_forward_cm": float(trial_forward_cm[trial - 1]),
            "pre_air_immobility_duration_s": (
                pre_end - pre_start
            ) * block_size / fs,
            "air_on_immobility_duration_s": (
                air_immobile_end - air_on
            ) * block_size / fs,
            "first_locomotor_onset_s": (
                first_move * block_size / fs if first_move is not None else np.nan
            ),
            "air_to_locomotion_latency_s": (
                (first_move - air_on) * block_size / fs
                if first_move is not None else np.nan
            ),
            "final_locomotor_offset_s": (
                final_move_offset * block_size / fs
                if final_move_offset is not None else np.nan
            ),
            "continued_running_after_air_off_s": (
                max(0, final_move_offset - air_off) * block_size / fs
                if final_move_offset is not None else np.nan
            ),
            "n_locomotor_bouts": len(bouts),
            "total_locomotion_during_air_s": total_moving_during_air_s,
            "initiation_total_s": state_totals["locomotor_initiation"],
            "sustained_locomotion_total_s": state_totals["sustained_locomotion"],
            "deceleration_total_s": state_totals["deceleration"],
            "immediate_post_run_duration_s": (
                (immediate_end - final_move_offset) * block_size / fs
                if final_move_offset is not None else np.nan
            ),
            "longer_quiet_immobility_duration_s": (
                (long_quiet_end - long_quiet_start) * block_size / fs
                if long_quiet_start is not None else np.nan
            ),
            "moving_at_air_on": moving_at_air_on,
            "valid_pre_air_immobility": (
                not moving_at_air_on and pre_end > pre_start
            ),
            "no_locomotor_response": len(bouts) == 0,
            "complex_trial": len(bouts) > 1,
        })

    trial_summary_df = pd.DataFrame(summary_rows)
    state_episodes_df = pd.DataFrame(episode_rows)

    if not state_episodes_df.empty:
        state_episodes_df = (
            state_episodes_df
            .sort_values(["trial", "onset_sample", "state"])
            .reset_index(drop=True)
        )

    state_signals = {
        "animal": animal,
        "date": date,
        "phase": phase,
        "fs": fs,
        "block_size": block_size,
        "analysis_hz": analysis_hz,
        "speed_path_cms": speed_path_ds,
        "speed_net_cms": speed_net_ds,
        "moving_mask": moving,
        "quiet_mask": quiet,
        "config": asdict(cfg),
    }

    return trial_summary_df, state_episodes_df, state_signals


def save_behavior_state_results(
    animal,
    date,
    trial_summary_df,
    state_episodes_df,
    root=None,
    filename="behavior_states_v1.h5",
):
    """Save state tables beside behavior_v1.h5."""
    behavior_h5 = Path(
        pdio.behavior_h5_path(animal, date)
        if root is None
        else pdio.behavior_h5_path(animal, date, root=root)
    )
    output_h5 = behavior_h5.parent / filename

    pdio.save_df_h5(
        trial_summary_df,
        output_h5,
        key="trial_summary",
        overwrite=True,
    )
    pdio.save_df_h5(
        state_episodes_df,
        output_h5,
        key="state_episodes",
        overwrite=True,
    )

    return output_h5


def plot_behavior_state_trial(
    trial,
    trial_summary_df,
    state_episodes_df,
    state_signals,
    pre_air_s=3,
    post_air_s=3,
):
    """Quality-control plot for one trial."""
    row = trial_summary_df.loc[
        trial_summary_df["trial"] == trial
    ].iloc[0]

    hz = state_signals["analysis_hz"]
    speed_path = state_signals["speed_path_cms"]
    speed_net = state_signals["speed_net_cms"]

    start_s = max(0, row["air_on_s"] - pre_air_s)
    end_s = min(len(speed_path) / hz, row["air_off_s"] + post_air_s)

    i0 = int(np.floor(start_s * hz))
    i1 = int(np.ceil(end_s * hz))
    t = np.arange(i0, i1) / hz

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, speed_path[i0:i1], label="speed_path_cms")
    ax.plot(t, speed_net[i0:i1], label="speed_net_cms", alpha=0.7)
    ax.axvline(row["air_on_s"], linestyle="--", label="air onset")
    ax.axvline(row["air_off_s"], linestyle=":", label="air offset")

    trial_states = state_episodes_df[
        state_episodes_df["trial"] == trial
    ]
    seen = set()

    for _, state_row in trial_states.iterrows():
        state = state_row["state"]
        ax.axvspan(
            state_row["onset_s"],
            state_row["offset_s"],
            alpha=0.12,
            label=state if state not in seen else None,
        )
        seen.add(state)

    ax.set_xlim(start_s, end_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title(f"{state_signals['animal']} | {state_signals['date']} | trial {trial}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax