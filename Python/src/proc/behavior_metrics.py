import numpy as np
import pandas as pd
from datetime import datetime
import src.utils.pdata_io as pdio



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