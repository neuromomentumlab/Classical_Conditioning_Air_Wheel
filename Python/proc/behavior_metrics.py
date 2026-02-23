import numpy as np
import pandas as pd


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