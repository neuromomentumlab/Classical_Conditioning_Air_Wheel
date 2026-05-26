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
    min_recording_duration_s=None,
    exclude_short_sessions_from_epochs=False,
):
    """
    Build one event-QC table across repeated wheel exposure / habituation,
    air training, and tone-air training.

    Uses stored Air_r and Air_f from behavior H5 files.

    For habituation / repeated wheel exposure:
        Air_r/Air_f are LED pseudo-events, not physical air.

    For air_training:
        Air_r/Air_f are physical air-on and air-off events.

    For tone_air_training:
        Air_r/Air_f are physical air-on and air-off events.
        Tone onset is inferred as air_on_time - 3 s.
        Tone offset is inferred as air_on_time + 2 s.

    Added day/session variables:
        rig_day_from_date:
            Calendar day from first recording day for each animal.

        rig_session_number:
            Sequential session number from first recording day for each animal.

        phase_day_from_date:
            Calendar day from first day of the current phase for each animal.

        phase_session_number:
            Sequential session number within the current phase for each animal.

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

                if np.isfinite(fs) and fs > 0 and np.isfinite(n_samples):
                    recording_duration_s = n_samples / fs
                else:
                    recording_duration_s = np.nan

                short_recording = False
                if min_recording_duration_s is not None:
                    if np.isfinite(recording_duration_s):
                        short_recording = recording_duration_s < min_recording_duration_s
                    else:
                        short_recording = True

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
                n_short_recording_excluded = 0
                n_valid_for_epoch_extraction = 0

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

                    short_session_excluded = (
                        short_recording and exclude_short_sessions_from_epochs
                    )

                    if short_session_excluded:
                        discard_reasons.append("short_recording_excluded")
                        n_short_recording_excluded += 1

                    valid_for_epoch_extraction = (
                        valid_pair
                        and not edge_excluded
                        and not short_led
                        and not tone_structure_bad
                        and not short_session_excluded
                    )

                    if valid_for_epoch_extraction:
                        n_valid_for_epoch_extraction += 1

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
                        "short_recording": short_recording,
                        "short_session_excluded": short_session_excluded,

                        "valid_for_epoch_extraction": valid_for_epoch_extraction,
                        "discard_reason": "; ".join(discard_reasons),

                        "fs": fs,
                        "n_samples": n_samples,
                        "recording_duration_s": recording_duration_s,
                    })

                session_rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,

                    "fs": fs,
                    "n_samples": n_samples,
                    "recording_duration_s": recording_duration_s,
                    "short_recording": short_recording,

                    "n_onsets": len(Air_r),
                    "n_offsets": len(Air_f),
                    "n_pairs": n_pairs,

                    "n_invalid_pair": n_invalid_pair,
                    "n_edge_excluded": n_edge_excluded,
                    "n_short_led": n_short_led,
                    "n_tone_structure_bad": n_tone_structure_bad,
                    "n_short_recording_excluded": n_short_recording_excluded,
                    "n_valid_for_epoch_extraction": n_valid_for_epoch_extraction,

                    "status": "ok",
                })

            except Exception as e:
                session_rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,

                    "fs": np.nan,
                    "n_samples": np.nan,
                    "recording_duration_s": np.nan,
                    "short_recording": np.nan,

                    "n_onsets": np.nan,
                    "n_offsets": np.nan,
                    "n_pairs": np.nan,

                    "n_invalid_pair": np.nan,
                    "n_edge_excluded": np.nan,
                    "n_short_led": np.nan,
                    "n_tone_structure_bad": np.nan,
                    "n_short_recording_excluded": np.nan,
                    "n_valid_for_epoch_extraction": np.nan,

                    "status": f"error: {e}",
                })

    events_df = pd.DataFrame(event_rows)
    session_summary_df = pd.DataFrame(session_rows)

    # --------------------------------------------------
    # Add absolute rig-day and phase-day variables
    # --------------------------------------------------
    if not session_summary_df.empty:

        session_summary_df["date_dt"] = pd.to_datetime(
            session_summary_df["date"].astype(str).str.replace("_", "-"),
            errors="coerce"
        )

        session_summary_df = session_summary_df.sort_values(
            ["animal", "date_dt", "date"]
        ).reset_index(drop=True)

        # Calendar day from first rig exposure
        first_rig_date = (
            session_summary_df
            .groupby("animal")["date_dt"]
            .transform("min")
        )

        session_summary_df["rig_day_from_date"] = (
            session_summary_df["date_dt"] - first_rig_date
        ).dt.days + 1

        # Sequential session number from first rig exposure
        session_summary_df["rig_session_number"] = (
            session_summary_df
            .groupby("animal")
            .cumcount() + 1
        )

        # Calendar day from first day of current phase
        first_phase_date = (
            session_summary_df
            .groupby(["animal", "phase"])["date_dt"]
            .transform("min")
        )

        session_summary_df["phase_day_from_date"] = (
            session_summary_df["date_dt"] - first_phase_date
        ).dt.days + 1

        # Sequential session number within current phase
        session_summary_df["phase_session_number"] = (
            session_summary_df
            .groupby(["animal", "phase"])
            .cumcount() + 1
        )

        # Merge day/session columns back into event table
        if not events_df.empty:

            day_cols = [
                "animal",
                "date",
                "phase",
                "date_dt",
                "rig_day_from_date",
                "rig_session_number",
                "phase_day_from_date",
                "phase_session_number",
            ]

            events_df = events_df.merge(
                session_summary_df[day_cols],
                on=["animal", "date", "phase"],
                how="left",
                validate="many_to_one"
            )

    # --------------------------------------------------
    # Final sorting
    # --------------------------------------------------
    if not events_df.empty:
        events_df = events_df.sort_values(
            ["phase", "animal", "date_dt", "date", "event_number"]
        ).reset_index(drop=True)

    if not session_summary_df.empty:
        session_summary_df = session_summary_df.sort_values(
            ["phase", "animal", "date_dt", "date"]
        ).reset_index(drop=True)

    return events_df, session_summary_df


def save_behavior_qc_tables(
    events_df,
    session_summary_df,
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

    # windows_df.to_hdf(
    #     qc_file,
    #     key="epoch_windows/windows_1s",
    #     mode="a",
    #     format="table"
    # )

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