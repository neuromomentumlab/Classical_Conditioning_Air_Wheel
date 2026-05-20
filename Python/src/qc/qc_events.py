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

import numpy as np
import pandas as pd
from pathlib import Path

import src.utils.pdata_io as pdio


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _as_1d(x):
    return np.asarray(x).reshape(-1)


def _safe_float(x):
    return float(np.asarray(x).squeeze())


def _safe_int(x):
    return int(np.asarray(x).squeeze())


# ------------------------------------------------------------
# Event QC table
# ------------------------------------------------------------

def build_event_qc_table(
    cc_data,
    phase_filter=None,
    drop_first_n=1,
    drop_last_n=1,
    min_hab_led_on_s=14.5,
):
    """
    Build one event-QC table across habituation, air-training, and tone-air-training.

    Uses stored Air_r and Air_f from behavior H5 files.

    For habituation:
        Air_r/Air_f are LED pseudo-events, not physical air.

    For air_training:
        Air_r/Air_f are physical air-on and air-off events.

    For tone_air_training:
        Air_r/Air_f are physical air-on and air-off events.
        Tone onset is inferred as air_on_time - 3 s.
        Tone offset is inferred as air_on_time + 2 s.

    Returns
    -------
    events_df : pandas.DataFrame
        One row per detected event/trial.

    session_summary_df : pandas.DataFrame
        One row per animal/date/phase.
    """

    if phase_filter is not None:
        if isinstance(phase_filter, str):
            phase_filter = {phase_filter}
        else:
            phase_filter = set(phase_filter)

    event_rows = []
    session_rows = []

    for animal, days in cc_data.items():
        for date, info in days.items():

            phase = info.get("phase", "unknown")

            if phase_filter is not None and phase not in phase_filter:
                continue

            try:
                keys = [
                    "t",
                    "fs",
                    "number_of_samples",
                    "Air_r",
                    "Air_f",
                    "air_bin",
                    "trial_forward_cm",
                ]

                b = pdio.load_behavior_h5(
                    animal,
                    date,
                    keys=keys
                )

                t = _as_1d(b["t"])
                fs = _safe_float(b["fs"])
                n_samples = _safe_int(b["number_of_samples"])

                Air_r = _as_1d(b["Air_r"]).astype(int)
                Air_f = _as_1d(b["Air_f"]).astype(int)

                if "trial_forward_cm" in b:
                    trial_forward_cm = _as_1d(b["trial_forward_cm"])
                else:
                    trial_forward_cm = np.array([])

                n_pairs = min(len(Air_r), len(Air_f))

                n_invalid_pair = 0
                n_edge_excluded = 0
                n_short_led = 0
                n_tone_structure_bad = 0

                for i in range(n_pairs):

                    r = int(Air_r[i])
                    f = int(Air_f[i])

                    on_time_s = t[r] if 0 <= r < len(t) else np.nan
                    off_time_s = t[f] if 0 <= f < len(t) else np.nan
                    on_duration_s = off_time_s - on_time_s

                    trial_forward = (
                        trial_forward_cm[i]
                        if i < len(trial_forward_cm)
                        else np.nan
                    )

                    valid_pair = (
                        np.isfinite(on_duration_s)
                        and f > r
                        and on_duration_s > 0
                    )

                    edge_excluded = (
                        i < drop_first_n
                        or i >= (n_pairs - drop_last_n)
                    )

                    short_led = False
                    tone_structure_bad = False

                    discard_reasons = []

                    if not valid_pair:
                        discard_reasons.append("invalid_on_off_pair")
                        n_invalid_pair += 1

                    if edge_excluded:
                        discard_reasons.append("edge_event_excluded")
                        n_edge_excluded += 1

                    if phase == "habituation":
                        if np.isfinite(on_duration_s) and on_duration_s < min_hab_led_on_s:
                            short_led = True
                            discard_reasons.append(
                                f"short_LED_ON_duration_{on_duration_s:.2f}s"
                            )
                            n_short_led += 1

                    if phase == "tone_air_training":
                        # In tone-air, tone starts 3 s before air onset,
                        # and tone turns off 2 s after air onset.
                        tone_on_time_s = on_time_s - 3.0
                        tone_off_time_s = on_time_s + 2.0

                        if (
                            tone_on_time_s < 0
                            or not np.isfinite(off_time_s)
                            or tone_off_time_s > off_time_s
                        ):
                            tone_structure_bad = True
                            discard_reasons.append("incomplete_tone_air_structure")
                            n_tone_structure_bad += 1

                    valid_for_epoch_extraction = (
                        valid_pair
                        and not edge_excluded
                        and not short_led
                        and not tone_structure_bad
                    )

                    event_rows.append({
                        "animal": animal,
                        "date": date,
                        "phase": phase,
                        "event_number": i,

                        "on_idx": r,
                        "off_idx": f,
                        "on_time_s": on_time_s,
                        "off_time_s": off_time_s,
                        "on_duration_s": on_duration_s,

                        "trial_forward_cm": trial_forward,

                        "valid_pair": valid_pair,
                        "edge_excluded": edge_excluded,
                        "short_led": short_led,
                        "tone_structure_bad": tone_structure_bad,

                        "valid_for_epoch_extraction": valid_for_epoch_extraction,
                        "discard_reason": "; ".join(discard_reasons),

                        "fs": fs,
                        "n_samples": n_samples,
                        "recording_duration_s": n_samples / fs,
                    })

                session_rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "n_onsets": len(Air_r),
                    "n_offsets": len(Air_f),
                    "n_pairs": n_pairs,
                    "n_invalid_pair": n_invalid_pair,
                    "n_edge_excluded": n_edge_excluded,
                    "n_short_led": n_short_led,
                    "n_tone_structure_bad": n_tone_structure_bad,
                    "status": "ok",
                })

            except Exception as e:
                session_rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "n_onsets": np.nan,
                    "n_offsets": np.nan,
                    "n_pairs": np.nan,
                    "n_invalid_pair": np.nan,
                    "n_edge_excluded": np.nan,
                    "n_short_led": np.nan,
                    "n_tone_structure_bad": np.nan,
                    "status": f"error: {e}",
                })

    events_df = pd.DataFrame(event_rows)
    session_summary_df = pd.DataFrame(session_rows)

    if not events_df.empty:
        events_df = events_df.sort_values(
            ["phase", "animal", "date", "event_number"]
        ).reset_index(drop=True)

    if not session_summary_df.empty:
        session_summary_df = session_summary_df.sort_values(
            ["phase", "animal", "date"]
        ).reset_index(drop=True)

    return events_df, session_summary_df


def build_epoch_window_table(
    events_df,
    window_s=1.0,
):
    """
    Build fixed-duration windows around event anchors.

    Each row is one analysis window.

    For habituation:
        LED_on_center
        LED_off_center

    For air_training:
        air_on_center
        air_off_center

    For tone_air_training:
        tone_on_center      = air_on - 3 s
        air_on_center       = air_on
        tone_off_center     = air_on + 2 s
        air_off_center      = air_off
    """

    import numpy as np
    import pandas as pd

    half_window = window_s / 2
    rows = []

    for _, ev in events_df.iterrows():

        animal = ev["animal"]
        date = ev["date"]
        phase = ev["phase"]
        event_number = ev["event_number"]

        fs = ev.get("fs", np.nan)
        n_samples = ev.get("n_samples", np.nan)

        on_time = ev.get("on_time_s", np.nan)
        off_time = ev.get("off_time_s", np.nan)

        base_valid = bool(ev.get("valid_for_epoch_extraction", False))

        anchors = []

        if phase == "habituation":
            anchors = [
                ("LED_on_center", on_time),
                ("LED_off_center", off_time),
            ]

        elif phase == "air_training":
            anchors = [
                ("air_on_center", on_time),
                ("air_off_center", off_time),
            ]

        elif phase == "tone_air_training":
            tone_on_time = on_time - 3.0 if np.isfinite(on_time) else np.nan
            tone_off_time = on_time + 2.0 if np.isfinite(on_time) else np.nan

            anchors = [
                ("tone_on_center", tone_on_time),
                ("air_on_center", on_time),
                ("tone_off_center", tone_off_time),
                ("air_off_center", off_time),
            ]

        else:
            continue

        for epoch_name, anchor_time in anchors:

            # -------------------------------------------------
            # Default invalid values
            # -------------------------------------------------
            window_start_s = np.nan
            window_end_s = np.nan
            start_idx = np.nan
            end_idx = np.nan
            valid_window = False
            invalid_reason = ""

            # -------------------------------------------------
            # Check parent event first
            # -------------------------------------------------
            if not base_valid:
                invalid_reason = "parent_event_invalid"

            elif not np.isfinite(anchor_time):
                invalid_reason = "anchor_time_nan"

            elif not np.isfinite(fs):
                invalid_reason = "fs_nan"

            elif not np.isfinite(n_samples):
                invalid_reason = "n_samples_nan"

            else:
                window_start_s = anchor_time - half_window
                window_end_s = anchor_time + half_window

                start_idx_tmp = int(round(window_start_s * fs))
                end_idx_tmp = int(round(window_end_s * fs))

                if start_idx_tmp < 0:
                    invalid_reason = "window_starts_before_recording"

                elif end_idx_tmp >= int(n_samples):
                    invalid_reason = "window_ends_after_recording"

                elif end_idx_tmp <= start_idx_tmp:
                    invalid_reason = "invalid_window_indices"

                else:
                    start_idx = start_idx_tmp
                    end_idx = end_idx_tmp
                    valid_window = True
                    invalid_reason = ""

            rows.append({
                "animal": animal,
                "date": date,
                "phase": phase,
                "event_number": event_number,

                "epoch_name": epoch_name,
                "window_s": window_s,

                "anchor_time_s": anchor_time,
                "window_start_s": window_start_s,
                "window_end_s": window_end_s,

                "start_idx": start_idx,
                "end_idx": end_idx,

                "valid_window": valid_window,
                "invalid_reason": invalid_reason,

                "parent_event_valid": base_valid,
            })

    windows_df = pd.DataFrame(rows)

    if not windows_df.empty:
        windows_df = windows_df.sort_values(
            ["phase", "animal", "date", "event_number", "epoch_name"]
        ).reset_index(drop=True)

    return windows_df


def save_behavior_qc_tables(
    events_df,
    session_summary_df,
    windows_df,
    pdata_root,
    filename="behavior_QC.h5",
):
    """
    Save QC and epoch-window tables into one H5 file under _cache.
    """

    cache_dir = Path(pdata_root) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    qc_file = cache_dir / filename

    events_df.to_hdf(
        qc_file,
        key="event_qc/events",
        mode="w",
        format="table"
    )

    session_summary_df.to_hdf(
        qc_file,
        key="event_qc/session_summary",
        mode="a",
        format="table"
    )

    windows_df.to_hdf(
        qc_file,
        key="epoch_windows/windows_1s",
        mode="a",
        format="table"
    )

    print(f"[SAVED] Behavior QC tables: {qc_file}")

    return qc_file


def load_behavior_qc_tables(
    pdata_root,
    filename="behavior_QC.h5",
):
    """
    Load behavior QC tables from _cache/behavior_QC.h5.
    """

    qc_file = Path(pdata_root) / "_cache" / filename

    events_df = pd.read_hdf(qc_file, key="event_qc/events")
    session_summary_df = pd.read_hdf(qc_file, key="event_qc/session_summary")
    windows_df = pd.read_hdf(qc_file, key="epoch_windows/windows_1s")

    print(f"[LOADED] Behavior QC tables: {qc_file}")

    return events_df, session_summary_df, windows_df