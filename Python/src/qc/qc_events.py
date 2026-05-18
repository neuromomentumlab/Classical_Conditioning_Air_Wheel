import numpy as np
import pandas as pd
import src.utils.pdata_io as pdio


def _as_1d(x):
    """Convert loaded H5 variable to 1D numpy array."""
    return np.asarray(x).reshape(-1)


def check_one_session_event_consistency(
    animal,
    date,
    phase=None,
    expected_off_s=15.0,
    expected_on_s=None,
    tolerance_s=2.0,
):
    """
    Check consistency of ON/OFF events for one session.

    Uses stored Air_r and Air_f from the processed H5 behavior file.

    Parameters
    ----------
    animal : str
    date : str
    phase : str or None
    expected_off_s : float
        Expected OFF/intertrial interval duration.
    expected_on_s : float or None
        Expected ON duration. For habituation LED pseudo-events, this may be 15 s.
        For air training, this should usually be None because air-on duration is behavior-dependent.
    tolerance_s : float
        Allowed deviation for expected durations.

    Returns
    -------
    event_df : pandas.DataFrame
        One row per ON/OFF event pair.

    session_summary : dict
        Session-level QC summary.
    """

    keys = ["t", "fs", "Air_r", "Air_f", "air_bin"]

    b = pdio.load_behavior_h5(
        animal,
        date,
        keys=keys
    )

    t = _as_1d(b["t"])
    fs = float(np.asarray(b["fs"]).squeeze())
    Air_r = _as_1d(b["Air_r"]).astype(int)
    Air_f = _as_1d(b["Air_f"]).astype(int)
    air_bin = _as_1d(b["air_bin"]).astype(int)

    n_r = len(Air_r)
    n_f = len(Air_f)
    n_pairs = min(n_r, n_f)

    rows = []

    for i in range(n_pairs):
        r = Air_r[i]
        f = Air_f[i]

        on_duration_s = t[f] - t[r] if f < len(t) and r < len(t) else np.nan

        if i < n_pairs - 1:
            next_r = Air_r[i + 1]
            off_duration_s = t[next_r] - t[f] if next_r < len(t) and f < len(t) else np.nan
        else:
            off_duration_s = np.nan

        valid_pair = (
            np.isfinite(on_duration_s)
            and f > r
            and on_duration_s > 0
        )

        on_duration_ok = True
        if expected_on_s is not None and np.isfinite(on_duration_s):
            on_duration_ok = abs(on_duration_s - expected_on_s) <= tolerance_s

        off_duration_ok = True
        if expected_off_s is not None and np.isfinite(off_duration_s):
            off_duration_ok = abs(off_duration_s - expected_off_s) <= tolerance_s

        rows.append({
            "animal": animal,
            "date": date,
            "phase": phase,
            "event_number": i,
            "on_idx": r,
            "off_idx": f,
            "on_time_s": t[r] if r < len(t) else np.nan,
            "off_time_s": t[f] if f < len(t) else np.nan,
            "on_duration_s": on_duration_s,
            "off_interval_to_next_on_s": off_duration_s,
            "valid_pair": valid_pair,
            "on_duration_ok": on_duration_ok,
            "off_duration_ok": off_duration_ok,
        })

    event_df = pd.DataFrame(rows)

    # Binary signal sanity check
    signal_starts_on = bool(air_bin[0] == 1) if len(air_bin) else False
    signal_ends_on = bool(air_bin[-1] == 1) if len(air_bin) else False

    # Session-level summary
    if len(event_df):
        n_invalid_pairs = int((~event_df["valid_pair"]).sum())
        n_bad_on = int((~event_df["on_duration_ok"]).sum())
        n_bad_off = int((~event_df["off_duration_ok"]).sum())

        median_on = float(np.nanmedian(event_df["on_duration_s"]))
        median_off = float(np.nanmedian(event_df["off_interval_to_next_on_s"]))
        min_on = float(np.nanmin(event_df["on_duration_s"]))
        max_on = float(np.nanmax(event_df["on_duration_s"]))
        min_off = float(np.nanmin(event_df["off_interval_to_next_on_s"]))
        max_off = float(np.nanmax(event_df["off_interval_to_next_on_s"]))
    else:
        n_invalid_pairs = np.nan
        n_bad_on = np.nan
        n_bad_off = np.nan
        median_on = np.nan
        median_off = np.nan
        min_on = np.nan
        max_on = np.nan
        min_off = np.nan
        max_off = np.nan

    needs_inspection = (
        n_r != n_f
        or signal_starts_on
        or signal_ends_on
        or (n_invalid_pairs > 0 if np.isfinite(n_invalid_pairs) else True)
        or (n_bad_on > 0 if expected_on_s is not None and np.isfinite(n_bad_on) else False)
        or (n_bad_off > 0 if expected_off_s is not None and np.isfinite(n_bad_off) else False)
    )

    session_summary = {
        "animal": animal,
        "date": date,
        "phase": phase,
        "n_onsets": n_r,
        "n_offsets": n_f,
        "n_pairs": n_pairs,
        "signal_starts_on": signal_starts_on,
        "signal_ends_on": signal_ends_on,
        "n_invalid_pairs": n_invalid_pairs,
        "n_bad_on_duration": n_bad_on,
        "n_bad_off_interval": n_bad_off,
        "median_on_duration_s": median_on,
        "median_off_interval_s": median_off,
        "min_on_duration_s": min_on,
        "max_on_duration_s": max_on,
        "min_off_interval_s": min_off,
        "max_off_interval_s": max_off,
        "needs_inspection": needs_inspection,
        "status": "ok",
    }

    return event_df, session_summary


def check_all_event_consistency(
    cc_data,
    phase_filter=None,
    tolerance_s=2.0,
):
    """
    Check ON/OFF event consistency for all sessions in cc_data.

    For habituation:
        expected ON ≈ 15 s, expected OFF ≈ 15 s.

    For air_training:
        ON duration is behavior-dependent, OFF interval ≈ 15 s.

    For tone_air_training:
        This checks the air signal only unless tone event keys are added later.
        ON duration is behavior-dependent, OFF interval ≈ 15 s.
    """

    if phase_filter is not None:
        if isinstance(phase_filter, str):
            phase_filter = {phase_filter}
        else:
            phase_filter = set(phase_filter)

    all_events = []
    all_summaries = []

    for animal, days in cc_data.items():
        for date, info in days.items():

            phase = info.get("phase", "unknown")

            if phase_filter is not None and phase not in phase_filter:
                continue

            # Phase-specific expectations
            if phase == "habituation":
                expected_on_s = 15.0
                expected_off_s = 15.0
            elif phase in ["air_training", "tone_air_training"]:
                expected_on_s = None
                expected_off_s = 15.0
            else:
                expected_on_s = None
                expected_off_s = None

            try:
                event_df, summary = check_one_session_event_consistency(
                    animal=animal,
                    date=date,
                    phase=phase,
                    expected_on_s=expected_on_s,
                    expected_off_s=expected_off_s,
                    tolerance_s=tolerance_s,
                )

                all_events.append(event_df)
                all_summaries.append(summary)

            except Exception as e:
                all_summaries.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "n_onsets": np.nan,
                    "n_offsets": np.nan,
                    "n_pairs": np.nan,
                    "signal_starts_on": np.nan,
                    "signal_ends_on": np.nan,
                    "n_invalid_pairs": np.nan,
                    "n_bad_on_duration": np.nan,
                    "n_bad_off_interval": np.nan,
                    "median_on_duration_s": np.nan,
                    "median_off_interval_s": np.nan,
                    "min_on_duration_s": np.nan,
                    "max_on_duration_s": np.nan,
                    "min_off_interval_s": np.nan,
                    "max_off_interval_s": np.nan,
                    "needs_inspection": True,
                    "status": f"error: {e}",
                })

    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    summary_all = pd.DataFrame(all_summaries)

    if not summary_all.empty:
        summary_all = summary_all.sort_values(
            ["phase", "animal", "date"]
        ).reset_index(drop=True)

    return events_all, summary_all

import numpy as np
import pandas as pd
import src.utils.pdata_io as pdio

def make_habituation_led_event_table(
    animal,
    date,
    min_led_on_s=14.5,
):
    """
    Build a cleaned habituation LED-event table.

    Habituation LED events are expected to be approximately 15 s ON.
    Events shorter than min_led_on_s are marked invalid and should not be used
    for epoch/window extraction.
    """

    keys = ["t", "fs", "Air_r", "Air_f", "air_bin"]

    b = pdio.load_behavior_h5(
        animal,
        date,
        keys=keys
    )

    t = np.asarray(b["t"]).reshape(-1)
    Air_r = np.asarray(b["Air_r"]).reshape(-1).astype(int)
    Air_f = np.asarray(b["Air_f"]).reshape(-1).astype(int)

    n_pairs = min(len(Air_r), len(Air_f))

    rows = []

    for i in range(n_pairs):
        r = Air_r[i]
        f = Air_f[i]

        if r < len(t) and f < len(t):
            on_duration_s = t[f] - t[r]
            on_time_s = t[r]
            off_time_s = t[f]
        else:
            on_duration_s = np.nan
            on_time_s = np.nan
            off_time_s = np.nan

        valid = True
        discard_reason = ""

        if not np.isfinite(on_duration_s):
            valid = False
            discard_reason = "missing_duration"

        elif f <= r:
            valid = False
            discard_reason = "offset_before_or_equal_onset"

        elif on_duration_s < min_led_on_s:
            valid = False
            discard_reason = f"short_LED_ON_duration_{on_duration_s:.2f}s"

        rows.append({
            "animal": animal,
            "date": date,
            "phase": "habituation",
            "event_type": "LED_sync_only",
            "event_number_original": i,
            "on_idx": r,
            "off_idx": f,
            "on_time_s": on_time_s,
            "off_time_s": off_time_s,
            "on_duration_s": on_duration_s,
            "valid_for_epoch_extraction": valid,
            "discard_reason": discard_reason,
        })

    event_df = pd.DataFrame(rows)

    return event_df

def make_all_habituation_led_event_tables(
    cc_data,
    min_led_on_s=14.5,
):
    """
    Build cleaned LED-event tables for all habituation sessions in cc_data.
    """

    all_events = []

    for animal, days in cc_data.items():
        for date, info in days.items():

            phase = info.get("phase", "unknown")

            if phase != "habituation":
                continue

            try:
                event_df = make_habituation_led_event_table(
                    animal,
                    date,
                    min_led_on_s=min_led_on_s
                )

                all_events.append(event_df)

            except Exception as e:
                all_events.append(pd.DataFrame([{
                    "animal": animal,
                    "date": date,
                    "phase": "habituation",
                    "event_type": "LED_sync_only",
                    "event_number_original": np.nan,
                    "on_idx": np.nan,
                    "off_idx": np.nan,
                    "on_time_s": np.nan,
                    "off_time_s": np.nan,
                    "on_duration_s": np.nan,
                    "valid_for_epoch_extraction": False,
                    "discard_reason": f"load_or_processing_error: {e}",
                }]))

    if len(all_events):
        return pd.concat(all_events, ignore_index=True)

    return pd.DataFrame()