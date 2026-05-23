import numpy as np
import pandas as pd
from datetime import datetime
import src.utils.pdata_io as pdio
import statsmodels.formula.api as smf



# =====================================================
# Utility
# =====================================================

def _safe_slice(x, i0, i1):
    """Clamp-safe slicing."""
    i0 = max(0, int(i0))
    i1 = min(len(x), int(i1))
    if i1 <= i0:
        return np.array([])
    return x[i0:i1]


# =====================================================
# Trial table builder
# =====================================================

def build_trial_table(b, drop_n=3, speed_thresh=1.0):
    """
    Build per-trial behavioral metrics.

    Parameters
    ----------
    b : behavior dict
    drop_n : int
        Number of initial trials to discard
    speed_thresh : float
        Movement threshold (cm/s)

    Returns
    -------
    df : pandas.DataFrame
    """

    Air_r = np.asarray(b["Air_r"])
    Air_f = np.asarray(b["Air_f"])
    t = b["t"]
    fs = b["fs"]

    speed = b["speed_net_cms"]  # forward proxy OK
    forward_cm = np.asarray(b["trial_forward_cm"])

    n_trials = len(Air_r)
    if n_trials == 0:
        return pd.DataFrame()

    rows = []

    for k in range(n_trials):

        r = Air_r[k]
        f = Air_f[k]

        # ----------------------------
        # Duration
        # ----------------------------
        duration_s = t[f] - t[r]

        # ----------------------------
        # Forward distance
        # ----------------------------
        fwd = forward_cm[k] if k < len(forward_cm) else np.nan

        # ----------------------------
        # Mean speed ON
        # ----------------------------
        seg_on = _safe_slice(speed, r, f)
        mean_speed_on = np.nanmean(seg_on) if len(seg_on) else np.nan
        peak_speed_on = np.nanmax(seg_on) if len(seg_on) else np.nan

        # ----------------------------
        # Pre-window (same duration)
        # ----------------------------
        win = f - r
        pre_start = r - win
        seg_pre = _safe_slice(speed, pre_start, r)

        mean_speed_pre = np.nanmean(seg_pre) if len(seg_pre) else np.nan

        # ----------------------------
        # AMI (paired)
        # ----------------------------
        ami = (
            (mean_speed_on - mean_speed_pre)
            / (mean_speed_on + mean_speed_pre + 1e-9)
        )

        delta_speed = mean_speed_on - mean_speed_pre

        # ----------------------------
        # Latency to move
        # ----------------------------
        latency_s = np.nan
        if len(seg_on):
            above = np.where(seg_on > speed_thresh)[0]
            if len(above):
                latency_s = above[0] / fs

        rows.append({
            "trial": k,
            "forward_cm": fwd,
            "duration_s": duration_s,
            "mean_speed_on": mean_speed_on,
            "peak_speed_on": peak_speed_on,
            "mean_speed_pre": mean_speed_pre,
            "AMI_speed": ami,
            "delta_speed": delta_speed,
            "latency_s": latency_s,
        })

    df = pd.DataFrame(rows)

    # -------------------------------------------------
    # Drop first N trials (warm-up)
    # -------------------------------------------------
    if drop_n > 0 and len(df) > drop_n:
        df = df.iloc[drop_n:].reset_index(drop=True)

    return df


# =====================================================
# Session summary
# =====================================================

def summarize_session(b, trial_df, speed_thresh=1.0):
    """
    Compute session-level metrics.
    """
    speedR = np.asarray(b["speed_net_cms"])
    air   = np.asarray(b["air_bin"]).astype(bool)

    eps = 1e-9

    # --- speed separation ---
    mean_on  = np.mean(speedR[air]) if np.any(air) else np.nan
    mean_off = np.mean(speedR[~air]) if np.any(~air) else np.nan
    speed_ratio = mean_on / (mean_off + eps)

    # --- stationary fraction during OFF ---
    stationary_off = np.mean(speedR[~air] < speed_thresh) if np.any(~air) else np.nan

    speed = b["speed_net_cms"]
    air_bin = b["air_bin"]

    moving = speed > speed_thresh

    summary = {}

    # ----------------------------
    # Trial-based metrics
    # ----------------------------
    if len(trial_df):

        summary["n_trials"] = len(trial_df)
        summary["median_forward_cm"] = np.nanmedian(trial_df["forward_cm"])
        summary["median_AMI"] = np.nanmedian(trial_df["AMI_speed"])
        summary["mean_AMI"] = np.nanmean(trial_df["AMI_speed"])
        summary["median_latency_s"] = np.nanmedian(trial_df["latency_s"])

        # success definition (adjustable) adjusted the threshold to 15cm
        summary["success_rate"] = np.nanmean(
            trial_df["forward_cm"] >= 15.0 # 15cm
        )

        summary["cv_forward"] = (
            np.nanstd(trial_df["forward_cm"])
            / (np.nanmean(trial_df["forward_cm"]) + 1e-9)
        )

    else:
        summary["n_trials"] = 0

    # ----------------------------
    # Time-based metrics
    # ----------------------------
    summary["frac_move_on"] = np.mean(moving[air_bin == 1])
    summary["frac_move_off"] = np.mean(moving[air_bin == 0])

    summary["mean_speed_on_time"] = np.nanmean(speed[air_bin == 1])
    summary["mean_speed_off_time"] = np.nanmean(speed[air_bin == 0])
    summary["speed_ratio"] = speed_ratio
    summary["stationary_off_frac"] = stationary_off

    return summary


def is_session_engaged(metrics,
                       speed_ratio_thr=2.0,
                       ami_thr=0.3,
                       off_frac_thr=0.6):

    return (
        (metrics["speed_ratio"] > speed_ratio_thr) and
        (metrics["mean_AMI"] > ami_thr) and
        (metrics["stationary_off_frac"] > off_frac_thr)
    )


def build_session_summary(results,
                          speed_ratio_thr=2.0,
                          ami_thr=0.3,
                          off_frac_thr=0.6,
                          speed_thresh=1.0):

    rows = []

    for animal, days in results.items():
        for date, b in days.items():

            try:
                trial_df = build_trial_table(b, drop_n=3)
                metrics = summarize_session(b, trial_df, speed_thresh = speed_thresh)
                # metrics = compute_session_metrics(
                #     b,
                #     speed_thresh=speed_thresh
                # )

                engaged = is_session_engaged(
                    metrics,
                    speed_ratio_thr=speed_ratio_thr,
                    ami_thr=ami_thr,
                    off_frac_thr=off_frac_thr
                )

                row = {
                    "animal": animal,
                    "date": date,
                    "engaged": engaged,
                    **metrics
                }

                rows.append(row)

            except Exception as e:
                rows.append({
                    "animal": animal,
                    "date": date,
                    "engaged": False,
                    "error": str(e)
                })

    df = pd.DataFrame(rows)

    # nice ordering
    if not df.empty:
        df = df.sort_values(["animal", "date"]).reset_index(drop=True)

    return df




def parse_date(date_str):
    return datetime.strptime(date_str, "%Y_%m_%d")


def classify_locomotor_state(b, speed_thresh=1.0):
    """
    Classify each time point into locomotor states using path speed
    and signed/net speed.

    States:
        stationary
        forward
        backward
        low_net_movement
    """

    speed_net = np.asarray(b["speed_net_cms"], dtype=float)
    speed_path = np.asarray(b["speed_path_cms"], dtype=float)

    # Make sure arrays are same length
    n = min(len(speed_net), len(speed_path))
    speed_net = speed_net[:n]
    speed_path = speed_path[:n]

    valid = np.isfinite(speed_net) & np.isfinite(speed_path)

    state = np.full(n, "invalid", dtype=object)

    # Stationary is based on path speed / movement magnitude
    state[valid & (speed_path <= speed_thresh)] = "stationary"

    # Directional movement is based on signed speed
    state[valid & (speed_path > speed_thresh) & (speed_net > speed_thresh)] = "forward"
    state[valid & (speed_path > speed_thresh) & (speed_net < -speed_thresh)] = "backward"

    # Movement without strong net direction
    state[
        valid
        & (speed_path > speed_thresh)
        & (np.abs(speed_net) <= speed_thresh)
    ] = "low_net_movement"

    return state, speed_net, speed_path


def summarize_habituation_locomotor_states(resultsH, speed_thresh=1.0):
    """
    Build one row per animal per habituation day with locomotor-state fractions.
    """

    rows = []

    for animal, days in resultsH.items():

        sorted_dates = sorted(days.keys(), key=parse_date)
        n_days = len(sorted_dates)

        for i, date in enumerate(sorted_dates, start=1):

            b = days[date]

            state, speed_net, speed_path = classify_locomotor_state(
                b,
                speed_thresh=speed_thresh
            )

            valid = state != "invalid"
            n_valid = np.sum(valid)

            if n_valid == 0:
                continue

            frac_stationary = np.mean(state[valid] == "stationary")
            frac_forward = np.mean(state[valid] == "forward")
            frac_backward = np.mean(state[valid] == "backward")
            frac_low_net = np.mean(state[valid] == "low_net_movement")

            frac_moving = 1.0 - frac_stationary

            rows.append({
                "animal": animal,
                "date": date,
                "phase": "Habituation",
                "speed_thresh": speed_thresh,

                "habituation_session": i,
                "n_habituation_sessions": n_days,
                "normalized_day": 0 if n_days == 1 else (i - 1) / (n_days - 1),

                "frac_stationary": frac_stationary,
                "frac_moving": frac_moving,
                "frac_forward": frac_forward,
                "frac_backward": frac_backward,
                "frac_low_net_movement": frac_low_net,

                "mean_path_speed": np.nanmean(speed_path),
                "median_path_speed": np.nanmedian(speed_path),
                "mean_net_speed": np.nanmean(speed_net),
                "median_net_speed": np.nanmedian(speed_net),

                "total_path_distance_cm": np.nanmax(b["dist_path_cm"]) - np.nanmin(b["dist_path_cm"]),
                "net_distance_cm": b["dist_net_cm"][-1] - b["dist_net_cm"][0],
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["animal", "date"]).reset_index(drop=True)

    return df

def build_session_summary_from_h5(
    cc_data,
    phase="air_training",
    animals=None,
    speed_ratio_thr=2.0,
    ami_thr=0.3,
    off_frac_thr=0.6,
    speed_thresh=1.0,
):

    rows = []

    for animal, days in cc_data.items():

        if animals is not None and animal not in animals:
            continue

        for date, info in days.items():

            if info.get("phase") != phase:
                continue

            try:
                b = pdio.load_behavior_for_analysis(
                    animal,
                    date,
                    analysis_name="session_summary"
                )

                if b is None:
                    continue

                trial_df = build_trial_table(b, drop_n=3)
                metrics = summarize_session(
                    b,
                    trial_df,
                    speed_thresh=speed_thresh
                )

                engaged = is_session_engaged(
                    metrics,
                    speed_ratio_thr=speed_ratio_thr,
                    ami_thr=ami_thr,
                    off_frac_thr=off_frac_thr,
                )

                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "engaged": engaged,
                    **metrics,
                })

            except Exception as e:
                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "engaged": False,
                    "error": str(e),
                })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["animal", "date"]).reset_index(drop=True)

    return df

def summarize_habituation_locomotor_states_from_h5(
    cc_data,
    speed_thresh=1.0,
    animals=None,
    verbose=True,
):
    """
    Build one row per animal per habituation day with locomotor-state fractions.
    Loads only the required variables from each H5 file.
    """

    rows = []

    for animal, days in cc_data.items():

        if animals is not None and animal not in animals:
            continue

        if verbose:
            print("\n================================================")
            print(f"Processing animal: {animal}")
            print("================================================")

        hab_dates = sorted(
            [
                date for date, info in days.items()
                if info.get("phase") == "habituation"
            ],
            key=parse_date
        )

        n_days = len(hab_dates)

        if verbose:
            print(f"Found {n_days} habituation sessions")

        for i, date in enumerate(hab_dates, start=1):

            if verbose:
                print(f"[{animal}] Session {i}/{n_days} Date: {date}")

            try:
                b = pdio.load_behavior_for_analysis(
                    animal,
                    date,
                    analysis_name="locomotor_state"
                )

                if b is None:
                    if verbose:
                        print(f"[SKIP] No H5 file found for {animal} {date}")
                    continue

                if verbose:
                    print(f"Loaded keys: {list(b.keys())}")

                state, speed_net, speed_path = classify_locomotor_state(
                    b,
                    speed_thresh=speed_thresh
                )

                valid = state != "invalid"
                n_valid = np.sum(valid)

                if n_valid == 0:
                    if verbose:
                        print(f"[SKIP] No valid samples for {animal} {date}")
                    continue

                frac_stationary = np.mean(state[valid] == "stationary")
                frac_moving = 1.0 - frac_stationary
                frac_forward = np.mean(state[valid] == "forward")
                frac_backward = np.mean(state[valid] == "backward")
                frac_low_net = np.mean(state[valid] == "low_net_movement")

                if verbose:
                    print(f"Computed locomotor states: {n_valid} valid samples")

                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": "habituation",
                    "speed_thresh": speed_thresh,

                    "habituation_session": i,
                    "n_habituation_sessions": n_days,
                    "normalized_day": 0 if n_days == 1 else (i - 1) / (n_days - 1),

                    "frac_stationary": frac_stationary,
                    "frac_moving": frac_moving,
                    "frac_forward": frac_forward,
                    "frac_backward": frac_backward,
                    "frac_low_net_movement": frac_low_net,

                    "mean_path_speed": np.nanmean(speed_path),
                    "median_path_speed": np.nanmedian(speed_path),
                    "mean_net_speed": np.nanmean(speed_net),
                    "median_net_speed": np.nanmedian(speed_net),

                    "total_path_distance_cm": (
                        np.nanmax(b["dist_path_cm"]) - np.nanmin(b["dist_path_cm"])
                    ),
                    "net_distance_cm": (
                        b["dist_net_cm"][-1] - b["dist_net_cm"][0]
                    ),
                })

            except Exception as e:

                if verbose:
                    print(f"[ERROR] {animal} {date}: {e}")

                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": "habituation",
                    "error": str(e),
                })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["animal", "date"]).reset_index(drop=True)

    return df


def classify_window_locomotion(speed_path, speed_net, speed_thresh=1.0):
    """
    Classify locomotor state within a window.
    """

    speed_path = np.asarray(speed_path, dtype=float)
    speed_net = np.asarray(speed_net, dtype=float)

    valid = np.isfinite(speed_path) & np.isfinite(speed_net)

    stationary = valid & (speed_path <= speed_thresh)
    forward = valid & (speed_path > speed_thresh) & (speed_net > speed_thresh)
    backward = valid & (speed_path > speed_thresh) & (speed_net < -speed_thresh)
    low_net = valid & (speed_path > speed_thresh) & (np.abs(speed_net) <= speed_thresh)

    n_valid = np.sum(valid)

    if n_valid == 0:
        return {
            "frac_stationary": np.nan,
            "frac_moving": np.nan,
            "frac_forward": np.nan,
            "frac_backward": np.nan,
            "frac_low_net_movement": np.nan,
            "dominant_locomotor_state": "invalid",
        }

    fractions = {
        "stationary": np.mean(stationary[valid]),
        "forward": np.mean(forward[valid]),
        "backward": np.mean(backward[valid]),
        "low_net_movement": np.mean(low_net[valid]),
    }

    dominant_state = max(fractions, key=fractions.get)

    return {
        "frac_stationary": fractions["stationary"],
        "frac_moving": 1.0 - fractions["stationary"],
        "frac_forward": fractions["forward"],
        "frac_backward": fractions["backward"],
        "frac_low_net_movement": fractions["low_net_movement"],
        "dominant_locomotor_state": dominant_state,
    }


def compute_encoder_epoch_metrics(
    windows_df,
    speed_thresh=1.0,
):
    """
    Compute encoder-derived metrics for each valid epoch/window.
    Loads each animal/date session once, then extracts all windows for that session.
    """

    rows = []

    # Only use valid windows
    df = windows_df[windows_df["valid_window"]].copy()

    keys = [
        "t",
        "fs",
        "speed_path_cms",
        "speed_net_cms",
        "dist_path_cm",
        "dist_net_cm",
    ]

    for (animal, date), sub in df.groupby(["animal", "date"]):

        try:
            b = pdio.load_behavior_h5(
                animal,
                date,
                keys=keys
            )

            speed_path = np.asarray(b["speed_path_cms"]).reshape(-1)
            speed_net = np.asarray(b["speed_net_cms"]).reshape(-1)
            dist_path = np.asarray(b["dist_path_cm"]).reshape(-1)
            dist_net = np.asarray(b["dist_net_cm"]).reshape(-1)

            n = min(len(speed_path), len(speed_net), len(dist_path), len(dist_net))

            speed_path = speed_path[:n]
            speed_net = speed_net[:n]
            dist_path = dist_path[:n]
            dist_net = dist_net[:n]

            for _, w in sub.iterrows():

                start_idx = int(w["start_idx"])
                end_idx = int(w["end_idx"])

                # Python slice is end-exclusive
                start_idx = max(0, start_idx)
                end_idx = min(n, end_idx)

                seg_path = speed_path[start_idx:end_idx]
                seg_net = speed_net[start_idx:end_idx]

                if len(seg_path) == 0:
                    continue

                state_metrics = classify_window_locomotion(
                    seg_path,
                    seg_net,
                    speed_thresh=speed_thresh
                )

                distance_path_cm = dist_path[end_idx - 1] - dist_path[start_idx]
                distance_net_cm = dist_net[end_idx - 1] - dist_net[start_idx]

                rows.append({
                    **w.to_dict(),

                    "speed_thresh_cms": speed_thresh,
                    "n_samples_window": len(seg_path),

                    "mean_speed_path_cms": np.nanmean(seg_path),
                    "median_speed_path_cms": np.nanmedian(seg_path),
                    "peak_speed_path_cms": np.nanmax(seg_path),

                    "mean_speed_net_cms": np.nanmean(seg_net),
                    "median_speed_net_cms": np.nanmedian(seg_net),
                    "min_speed_net_cms": np.nanmin(seg_net),
                    "max_speed_net_cms": np.nanmax(seg_net),

                    "distance_path_cm": distance_path_cm,
                    "distance_net_cm": distance_net_cm,

                    "net_direction_bias": (
                        np.nanmean(seg_net) /
                        (np.nanmean(seg_path) + 1e-9)
                    ),

                    **state_metrics,
                })

        except Exception as e:
            print(f"[ERROR] {animal} {date}: {e}")

    metrics_df = pd.DataFrame(rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["phase", "animal", "date", "event_number", "epoch_name"]
        ).reset_index(drop=True)

    return metrics_df
    

import numpy as np
import pandas as pd


def add_normalized_phase_day(df):
    """
    Add session number and normalized phase day within each animal x phase.
    """

    df = df.copy()

    # make sure date sorting works because date format is YYYY_MM_DD
    session_table = (
        df[["animal", "phase", "date"]]
        .drop_duplicates()
        .sort_values(["animal", "phase", "date"])
        .reset_index(drop=True)
    )

    session_table["phase_session_number"] = np.nan
    session_table["n_sessions_in_phase"] = np.nan
    session_table["normalized_phase_day"] = np.nan

    for (animal, phase), idx in session_table.groupby(["animal", "phase"]).groups.items():
        idx = list(idx)
        n = len(idx)

        session_numbers = np.arange(1, n + 1)

        if n == 1:
            normalized = np.zeros(n)
        else:
            normalized = (session_numbers - 1) / (n - 1)

        session_table.loc[idx, "phase_session_number"] = session_numbers
        session_table.loc[idx, "n_sessions_in_phase"] = n
        session_table.loc[idx, "normalized_phase_day"] = normalized

    df = df.merge(
        session_table,
        on=["animal", "phase", "date"],
        how="left"
    )

    return df

import numpy as np
import pandas as pd
from datetime import datetime
import src.utils.pdata_io as pdio



def classify_locomotor_samples(speed_path, speed_net, speed_thresh=1.0):
    """
    Classify each sample within a window.

    speed_path: movement magnitude
    speed_net: signed movement direction
    """

    speed_path = np.asarray(speed_path, dtype=float)
    speed_net = np.asarray(speed_net, dtype=float)

    valid = np.isfinite(speed_path) & np.isfinite(speed_net)

    state = np.full(len(speed_path), "invalid", dtype=object)

    state[valid & (speed_path <= speed_thresh)] = "stationary"

    state[
        valid &
        (speed_path > speed_thresh) &
        (speed_net > speed_thresh)
    ] = "forward"

    state[
        valid &
        (speed_path > speed_thresh) &
        (speed_net < -speed_thresh)
    ] = "backward"

    state[
        valid &
        (speed_path > speed_thresh) &
        (np.abs(speed_net) <= speed_thresh)
    ] = "low_net_movement"

    return state


def summarize_locomotor_state_in_window(speed_path, speed_net, speed_thresh=1.0):
    """
    Return fraction of samples in each locomotor state within one window.
    """

    state = classify_locomotor_samples(
        speed_path,
        speed_net,
        speed_thresh=speed_thresh
    )

    valid = state != "invalid"
    n_valid = np.sum(valid)

    if n_valid == 0:
        return {
            "frac_stationary": np.nan,
            "frac_moving": np.nan,
            "frac_forward": np.nan,
            "frac_backward": np.nan,
            "frac_low_net_movement": np.nan,
            "dominant_locomotor_state": "invalid",
        }

    frac_stationary = np.mean(state[valid] == "stationary")
    frac_forward = np.mean(state[valid] == "forward")
    frac_backward = np.mean(state[valid] == "backward")
    frac_low_net = np.mean(state[valid] == "low_net_movement")
    frac_moving = 1.0 - frac_stationary

    state_fracs = {
        "stationary": frac_stationary,
        "forward": frac_forward,
        "backward": frac_backward,
        "low_net_movement": frac_low_net,
    }

    dominant_state = max(state_fracs, key=state_fracs.get)

    return {
        "frac_stationary": frac_stationary,
        "frac_moving": frac_moving,
        "frac_forward": frac_forward,
        "frac_backward": frac_backward,
        "frac_low_net_movement": frac_low_net,
        "dominant_locomotor_state": dominant_state,
    }


def compute_encoder_metrics_for_windows(
    windows_df,
    speed_thresh=1.0,
):
    """
    Compute encoder-derived metrics for each valid epoch/window.

    One output row = one animal/date/event/epoch/window.
    """

    rows = []

    df = windows_df[windows_df["valid_window"]].copy()

    keys = [
        "speed_path_cms",
        "speed_net_cms",
        "dist_path_cm",
        "dist_net_cm",
        "fs",
    ]

    for (animal, date), sub in df.groupby(["animal", "date"]):

        try:
            b = pdio.load_behavior_h5(
                animal,
                date,
                keys=keys
            )

            speed_path = np.asarray(b["speed_path_cms"]).reshape(-1)
            speed_net = np.asarray(b["speed_net_cms"]).reshape(-1)
            dist_path = np.asarray(b["dist_path_cm"]).reshape(-1)
            dist_net = np.asarray(b["dist_net_cm"]).reshape(-1)
            fs = float(np.asarray(b["fs"]).squeeze())

            n = min(
                len(speed_path),
                len(speed_net),
                len(dist_path),
                len(dist_net)
            )

            speed_path = speed_path[:n]
            speed_net = speed_net[:n]
            dist_path = dist_path[:n]
            dist_net = dist_net[:n]

            for _, w in sub.iterrows():

                start_idx = int(w["start_idx"])
                end_idx = int(w["end_idx"])

                start_idx = max(0, start_idx)
                end_idx = min(n, end_idx)

                if end_idx <= start_idx:
                    continue

                seg_path = speed_path[start_idx:end_idx]
                seg_net = speed_net[start_idx:end_idx]

                state_metrics = summarize_locomotor_state_in_window(
                    seg_path,
                    seg_net,
                    speed_thresh=speed_thresh
                )

                distance_path_cm = dist_path[end_idx - 1] - dist_path[start_idx]
                distance_net_cm = dist_net[end_idx - 1] - dist_net[start_idx]

                row = {
                    **w.to_dict(),

                    "speed_thresh_cms": speed_thresh,
                    "fs": fs,
                    "n_samples_window": len(seg_path),

                    "mean_speed_path_cms": np.nanmean(seg_path),
                    "median_speed_path_cms": np.nanmedian(seg_path),
                    "peak_speed_path_cms": np.nanmax(seg_path),

                    "mean_speed_net_cms": np.nanmean(seg_net),
                    "median_speed_net_cms": np.nanmedian(seg_net),
                    "min_speed_net_cms": np.nanmin(seg_net),
                    "max_speed_net_cms": np.nanmax(seg_net),

                    "distance_path_cm": distance_path_cm,
                    "distance_net_cm": distance_net_cm,

                    "net_direction_bias": (
                        np.nanmean(seg_net) /
                        (np.nanmean(seg_path) + 1e-9)
                    ),

                    **state_metrics,
                }

                rows.append(row)

        except Exception as e:
            print(f"[ERROR] {animal} {date}: {e}")

    metrics_df = pd.DataFrame(rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["phase", "animal", "date", "event_number", "epoch_name"]
        ).reset_index(drop=True)

    return metrics_df


def make_session_epoch_summary(encoder_epoch_df):
    """
    Average window metrics within each animal/date/phase/anchor/window_position/session_time_bin.

    One output row = one animal/session/anchor/pre-post/time-bin summary.
    """

    metrics = [
        "mean_speed_path_cms",
        "median_speed_path_cms",
        "peak_speed_path_cms",
        "mean_speed_net_cms",
        "median_speed_net_cms",
        "min_speed_net_cms",
        "max_speed_net_cms",
        "distance_path_cm",
        "distance_net_cm",
        "frac_stationary",
        "frac_moving",
        "frac_forward",
        "frac_backward",
        "frac_low_net_movement",
        "net_direction_bias",
    ]

    desired_group_cols = [
        "animal",
        "date",
        "phase",
        "phase_session_number",
        "normalized_phase_day",
        "session_time_bin",
        "anchor_name",
        "anchor_type",
        "window_position",
        "window_s",
        "epoch_name",
    ]

    group_cols = [
        c for c in desired_group_cols
        if c in encoder_epoch_df.columns
    ]

    available_metrics = [
        m for m in metrics
        if m in encoder_epoch_df.columns
    ]

    session_epoch_df = (
        encoder_epoch_df
        .groupby(group_cols, dropna=False)
        .agg(
            **{m: (m, "mean") for m in available_metrics},
            n_windows=("epoch_name", "count"),
            n_events=("event_number", "nunique"),
        )
        .reset_index()
    )

    return session_epoch_df



def fit_phase_lmm(
    df,
    phase,
    outcome,
    include_day=True,
):
    """
    Fit mixed model within one phase.

    Model:
        outcome ~ anchor_name * window_position + normalized_phase_day + (1|animal)
    """

    sub = df[df["phase"] == phase].copy()

    sub = sub[
        np.isfinite(sub[outcome])
    ].copy()

    if include_day:
        formula = f"{outcome} ~ C(anchor_name) * C(window_position) + normalized_phase_day"
    else:
        formula = f"{outcome} ~ C(anchor_name) * C(window_position)"

    model = smf.mixedlm(
        formula,
        data=sub,
        groups=sub["animal"]
    )

    result = model.fit(method="lbfgs")

    return result



def make_prepost_delta_table(session_epoch_df, outcome):
    """
    Create delta table:
        delta = post - pre
    for each animal/date/phase/anchor_name.
    """

    id_cols = [
        "animal",
        "date",
        "phase",
        "phase_session_number",
        "normalized_phase_day",
        "anchor_name",
    ]

    pivot = session_epoch_df.pivot_table(
        index=id_cols,
        columns="window_position",
        values=outcome,
        aggfunc="mean"
    ).reset_index()

    if "pre" not in pivot.columns or "post" not in pivot.columns:
        raise ValueError("Both pre and post windows are required.")

    pivot[f"delta_{outcome}"] = pivot["post"] - pivot["pre"]

    return pivot



def fit_delta_lmm(delta_df, phase, outcome):
    """
    Fit LMM on post-pre delta within one phase.
    """

    delta_col = f"delta_{outcome}"

    sub = delta_df[
        (delta_df["phase"] == phase) &
        (np.isfinite(delta_df[delta_col]))
    ].copy()

    formula = f"{delta_col} ~ C(anchor_name) + normalized_phase_day"

    model = smf.mixedlm(
        formula,
        data=sub,
        groups=sub["animal"]
    )

    result = model.fit(method="lbfgs")

    return result


import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon


def animal_level_delta_summary(delta_df, phase, outcome):
    """
    Average post-pre delta within each animal and anchor.
    """

    delta_col = f"delta_{outcome}"

    sub = delta_df[
        (delta_df["phase"] == phase) &
        (np.isfinite(delta_df[delta_col]))
    ].copy()

    animal_summary = (
        sub
        .groupby(["animal", "anchor_name"], as_index=False)
        .agg(
            mean_delta=(delta_col, "mean"),
            median_delta=(delta_col, "median"),
            n_sessions=(delta_col, "count")
        )
    )

    return animal_summary


def test_animal_level_deltas(animal_summary):
    """
    Test each anchor against zero at the animal level.
    """

    rows = []

    for anchor, sub in animal_summary.groupby("anchor_name"):
        vals = sub["mean_delta"].dropna().values

        if len(vals) < 2:
            continue

        t_res = ttest_1samp(vals, 0)

        try:
            w_res = wilcoxon(vals)
            w_p = w_res.pvalue
        except Exception:
            w_p = np.nan

        rows.append({
            "anchor_name": anchor,
            "n_animals": len(vals),
            "mean_delta": np.mean(vals),
            "sem_delta": np.std(vals, ddof=1) / np.sqrt(len(vals)),
            "median_delta": np.median(vals),
            "t_stat_vs_0": t_res.statistic,
            "p_ttest_vs_0": t_res.pvalue,
            "p_wilcoxon_vs_0": w_p
        })

    return pd.DataFrame(rows)


def paired_anchor_comparison(animal_summary, anchor_a, anchor_b):
    """
    Paired comparison between two anchors at the animal level.
    """

    wide = animal_summary.pivot(
        index="animal",
        columns="anchor_name",
        values="mean_delta"
    )

    vals_a = wide[anchor_a]
    vals_b = wide[anchor_b]

    valid = vals_a.notna() & vals_b.notna()

    vals_a = vals_a[valid]
    vals_b = vals_b[valid]

    t_res = ttest_rel(vals_a, vals_b)

    try:
        w_res = wilcoxon(vals_a, vals_b)
        w_p = w_res.pvalue
    except Exception:
        w_p = np.nan

    return {
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "n_animals": valid.sum(),
        "mean_delta_a": vals_a.mean(),
        "mean_delta_b": vals_b.mean(),
        "mean_difference_a_minus_b": (vals_a - vals_b).mean(),
        "p_paired_ttest": t_res.pvalue,
        "p_wilcoxon": w_p,
        "wide_table": wide
    }

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM


def make_animal_condition_table(session_epoch_df, phase, outcome):
    """
    Average values across sessions for each animal × anchor × pre/post condition.

    This creates one value per animal per repeated-measures condition.
    """

    sub = session_epoch_df[
        (session_epoch_df["phase"] == phase) &
        (np.isfinite(session_epoch_df[outcome]))
    ].copy()

    animal_cond = (
        sub
        .groupby(["animal", "anchor_name", "window_position"], as_index=False)
        .agg(
            value=(outcome, "mean"),
            n_sessions=(outcome, "count")
        )
    )

    return animal_cond


def run_2way_rm_anova(session_epoch_df, phase, outcome):
    """
    Run 2-way repeated-measures ANOVA:
        outcome ~ anchor_name × window_position
    with animal as subject.
    """

    animal_cond = make_animal_condition_table(
        session_epoch_df,
        phase=phase,
        outcome=outcome
    )

    # Keep only animals with complete data for all conditions
    n_conditions = (
        animal_cond[["anchor_name", "window_position"]]
        .drop_duplicates()
        .shape[0]
    )

    complete_animals = (
        animal_cond
        .groupby("animal")
        .filter(lambda x: len(x) == n_conditions)
        ["animal"]
        .unique()
    )

    animal_cond = animal_cond[
        animal_cond["animal"].isin(complete_animals)
    ].copy()

    aov = AnovaRM(
        data=animal_cond,
        depvar="value",
        subject="animal",
        within=["anchor_name", "window_position"]
    ).fit()

    return aov, animal_cond


from statsmodels.stats.anova import AnovaRM


def run_delta_rm_anova(delta_df, phase, outcome):
    """
    Run one-way repeated-measures ANOVA on post-pre deltas.

    Repeated factor:
        anchor_name
    Subject:
        animal
    """

    delta_col = f"delta_{outcome}"

    sub = delta_df[
        (delta_df["phase"] == phase) &
        (np.isfinite(delta_df[delta_col]))
    ].copy()

    animal_delta = (
        sub
        .groupby(["animal", "anchor_name"], as_index=False)
        .agg(
            value=(delta_col, "mean"),
            n_sessions=(delta_col, "count")
        )
    )

    # keep only complete animals
    n_conditions = animal_delta["anchor_name"].nunique()

    complete_animals = (
        animal_delta
        .groupby("animal")
        .filter(lambda x: len(x) == n_conditions)
        ["animal"]
        .unique()
    )

    animal_delta = animal_delta[
        animal_delta["animal"].isin(complete_animals)
    ].copy()

    aov = AnovaRM(
        data=animal_delta,
        depvar="value",
        subject="animal",
        within=["anchor_name"]
    ).fit()

    return aov, animal_delta



def add_day_bin(df, n_bins=3):
    """
    Add early/middle/late or early/late bins based on normalized_phase_day.
    """

    df = df.copy()

    if n_bins == 2:
        df["day_bin"] = pd.cut(
            df["normalized_phase_day"],
            bins=[-0.001, 0.5, 1.001],
            labels=["early", "late"]
        )

    elif n_bins == 3:
        df["day_bin"] = pd.cut(
            df["normalized_phase_day"],
            bins=[-0.001, 1/3, 2/3, 1.001],
            labels=["early", "middle", "late"]
        )

    else:
        raise ValueError("Use n_bins=2 or n_bins=3.")

    df["day_bin"] = df["day_bin"].astype(str)

    return df


def run_3way_rm_anova_with_day(session_epoch_df, phase, outcome):
    """
    Repeated-measures ANOVA on raw pre/post values.

    Within-subject factors:
        anchor_name
        window_position
        day_bin
    """

    sub = session_epoch_df[
        (session_epoch_df["phase"] == phase) &
        (np.isfinite(session_epoch_df[outcome]))
    ].copy()

    animal_cond = (
        sub
        .groupby(["animal", "day_bin", "anchor_name", "window_position"], as_index=False)
        .agg(
            value=(outcome, "mean"),
            n_sessions=(outcome, "count")
        )
    )

    n_conditions = (
        animal_cond[["day_bin", "anchor_name", "window_position"]]
        .drop_duplicates()
        .shape[0]
    )

    complete_animals = (
        animal_cond
        .groupby("animal")
        .filter(lambda x: len(x) == n_conditions)
        ["animal"]
        .unique()
    )

    animal_cond = animal_cond[
        animal_cond["animal"].isin(complete_animals)
    ].copy()

    aov = AnovaRM(
        data=animal_cond,
        depvar="value",
        subject="animal",
        within=["anchor_name", "window_position", "day_bin"]
    ).fit()

    return aov, animal_cond


from statsmodels.stats.anova import AnovaRM
import numpy as np
import pandas as pd


def run_anchor_specific_rm_anova(session_epoch_df, phase, anchor_name, outcome):
    """
    Repeated-measures ANOVA for one event anchor.

    Model:
        outcome ~ window_position × day_bin

    Subject:
        animal
    """

    sub = session_epoch_df[
        (session_epoch_df["phase"] == phase) &
        (session_epoch_df["anchor_name"] == anchor_name) &
        (np.isfinite(session_epoch_df[outcome]))
    ].copy()

    animal_cond = (
        sub
        .groupby(["animal", "day_bin", "window_position"], as_index=False)
        .agg(
            value=(outcome, "mean"),
            n_sessions=(outcome, "count")
        )
    )

    # Keep only animals with complete repeated-measures conditions
    n_conditions = (
        animal_cond[["day_bin", "window_position"]]
        .drop_duplicates()
        .shape[0]
    )

    complete_animals = (
        animal_cond
        .groupby("animal")
        .filter(lambda x: len(x) == n_conditions)
        ["animal"]
        .unique()
    )

    animal_cond = animal_cond[
        animal_cond["animal"].isin(complete_animals)
    ].copy()

    aov = AnovaRM(
        data=animal_cond,
        depvar="value",
        subject="animal",
        within=["window_position", "day_bin"]
    ).fit()

    return aov, animal_cond


def run_anchor_delta_day_rm_anova(delta_df, phase, anchor_name, outcome):
    """
    For one anchor, run repeated-measures ANOVA on delta values across day_bin.

    Model:
        delta ~ day_bin

    Subject:
        animal
    """

    delta_col = f"delta_{outcome}"

    sub = delta_df[
        (delta_df["phase"] == phase) &
        (delta_df["anchor_name"] == anchor_name) &
        (np.isfinite(delta_df[delta_col]))
    ].copy()

    animal_day = (
        sub
        .groupby(["animal", "day_bin"], as_index=False)
        .agg(
            value=(delta_col, "mean"),
            n_sessions=(delta_col, "count")
        )
    )

    n_conditions = animal_day["day_bin"].nunique()

    complete_animals = (
        animal_day
        .groupby("animal")
        .filter(lambda x: len(x) == n_conditions)
        ["animal"]
        .unique()
    )

    animal_day = animal_day[
        animal_day["animal"].isin(complete_animals)
    ].copy()

    aov = AnovaRM(
        data=animal_day,
        depvar="value",
        subject="animal",
        within=["day_bin"]
    ).fit()

    return aov, animal_day


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_whole_session_locomotor_metrics(
    session_day_df,
    phase="habituation",
    speed_thresh=0.5,
    good_sessions_only=True,
):
    """
    Compute whole-session locomotor metrics for each animal/date/phase.

    One row = one animal/session.
    """

    rows = []

    df = session_day_df[session_day_df["phase"] == phase].copy()

    if good_sessions_only and "good_session_basic" in df.columns:
        df = df[df["good_session_basic"] == True].copy()

    for _, sess in df.iterrows():

        animal = sess["animal"]
        date = sess["date"]

        try:
            b = pdio.load_behavior_h5(
                animal,
                date,
                keys=[
                    "speed_path_cms",
                    "speed_net_cms",
                    "fs",
                    "dist_path_cm",
                    "dist_net_cm",
                ]
            )

            speed_path = np.asarray(b["speed_path_cms"]).reshape(-1)
            speed_net = np.asarray(b["speed_net_cms"]).reshape(-1)
            fs = float(np.asarray(b["fs"]).squeeze())

            n = min(len(speed_path), len(speed_net))
            speed_path = speed_path[:n]
            speed_net = speed_net[:n]

            state_metrics = summarize_locomotor_state_in_window(
                speed_path,
                speed_net,
                speed_thresh=speed_thresh
            )

            duration_s = n / fs

            row = {
                **sess.to_dict(),

                "speed_thresh_cms": speed_thresh,
                "fs": fs,
                "n_samples_used": n,
                "duration_s_used": duration_s,

                "mean_speed_path_cms": np.nanmean(speed_path),
                "median_speed_path_cms": np.nanmedian(speed_path),
                "peak_speed_path_cms": np.nanmax(speed_path),

                "mean_speed_net_cms": np.nanmean(speed_net),
                "median_speed_net_cms": np.nanmedian(speed_net),
                "min_speed_net_cms": np.nanmin(speed_net),
                "max_speed_net_cms": np.nanmax(speed_net),

                **state_metrics,
            }

            rows.append(row)

        except Exception as e:
            print(f"[ERROR] {animal} {date}: {e}")

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["animal", "phase_day_number_good", "date"]
        ).reset_index(drop=True)

    return out