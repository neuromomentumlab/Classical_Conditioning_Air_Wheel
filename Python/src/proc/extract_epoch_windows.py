import numpy as np
import pandas as pd
from pathlib import Path


def build_prepost_epoch_windows(
    events_df,
    window_s=1.0,
):
    """
    Build pre-event and post-event windows from validated event QC table.

    This function does not perform QC itself.
    It uses valid events from events_df and creates analysis windows.

    Required columns in events_df:
        animal
        date
        phase
        event_number
        on_time_s
        off_time_s
        fs
        n_samples
        valid_for_epoch_extraction

    Returns
    -------
    windows_df : pandas.DataFrame
        One row per animal/date/event/anchor/pre-post window.
    """

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

        parent_valid = bool(ev.get("valid_for_epoch_extraction", False))

        # --------------------------------------------------
        # Define anchor events by phase
        # --------------------------------------------------
        anchors = []

        if phase == "habituation":
            anchors = [
                ("LED_on", on_time),
                ("LED_off", off_time),
            ]

        elif phase == "air_training":
            anchors = [
                ("air_on", on_time),
                ("air_off", off_time),
            ]

        elif phase == "tone_air_training":
            tone_on_time = on_time - 3.0 if np.isfinite(on_time) else np.nan
            tone_off_time = on_time + 2.0 if np.isfinite(on_time) else np.nan

            anchors = [
                ("tone_on", tone_on_time),
                ("air_on", on_time),
                ("tone_off", tone_off_time),
                ("air_off", off_time),
            ]

        else:
            continue

        # --------------------------------------------------
        # Make pre/post windows for each anchor
        # --------------------------------------------------
        for anchor_name, anchor_time in anchors:

            window_defs = [
                ("pre", anchor_time - window_s, anchor_time),
                ("post", anchor_time, anchor_time + window_s),
            ]

            for window_position, window_start_s, window_end_s in window_defs:

                epoch_name = f"{anchor_name}_{window_position}_{window_s:g}s"

                valid_window = False
                invalid_reason = ""

                start_idx = np.nan
                end_idx = np.nan

                if not parent_valid:
                    invalid_reason = "parent_event_invalid"

                elif not np.isfinite(anchor_time):
                    invalid_reason = "anchor_time_nan"

                elif not np.isfinite(fs):
                    invalid_reason = "fs_nan"

                elif not np.isfinite(n_samples):
                    invalid_reason = "n_samples_nan"

                elif window_start_s < 0:
                    invalid_reason = "window_starts_before_recording"

                else:
                    start_idx_tmp = int(round(window_start_s * fs))
                    end_idx_tmp = int(round(window_end_s * fs))

                    if start_idx_tmp < 0:
                        invalid_reason = "window_starts_before_recording"

                    elif end_idx_tmp > int(n_samples):
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

                    "anchor_name": anchor_name,
                    "anchor_time_s": anchor_time,

                    "window_position": window_position,
                    "window_s": window_s,
                    "epoch_name": epoch_name,

                    "window_start_s": window_start_s,
                    "window_end_s": window_end_s,
                    "start_idx": start_idx,
                    "end_idx": end_idx,

                    "parent_event_valid": parent_valid,
                    "valid_window": valid_window,
                    "invalid_reason": invalid_reason,
                })

    windows_df = pd.DataFrame(rows)

    if not windows_df.empty:
        windows_df = windows_df.sort_values(
            ["phase", "animal", "date", "event_number", "anchor_name", "window_position"]
        ).reset_index(drop=True)

    return windows_df


def save_epoch_windows(
    windows_df,
    pdata_root,
    filename="behavior_epoch_windows.h5",
    key="windows/prepost_1s",
):
    """
    Save epoch/window table separately from QC.
    """

    cache_dir = Path(pdata_root) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_file = cache_dir / filename

    windows_df.to_hdf(
        out_file,
        key=key,
        mode="w",
        format="table"
    )

    print(f"[SAVED] Epoch windows: {out_file}")
    print(f"[KEY] {key}")

    return out_file


def load_epoch_windows(
    pdata_root,
    filename="behavior_epoch_windows.h5",
    key="windows/prepost_1s",
):
    """
    Load saved epoch/window table.
    """

    in_file = Path(pdata_root) / "_cache" / filename

    windows_df = pd.read_hdf(
        in_file,
        key=key
    )

    print(f"[LOADED] Epoch windows: {in_file}")
    print(f"[KEY] {key}")

    return windows_df