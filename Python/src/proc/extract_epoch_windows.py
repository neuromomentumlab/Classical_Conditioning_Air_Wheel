import numpy as np
import pandas as pd
from pathlib import Path


import numpy as np
import pandas as pd


def _append_reason(old_reason, new_reason):
    """
    Append invalid reason without deleting previous reason.
    """
    if old_reason is None or old_reason == "" or pd.isna(old_reason):
        return new_reason
    return f"{old_reason};{new_reason}"


def _assign_session_time_bin(
    anchor_time_s,
    session_duration_s,
    method="thirds",
    fixed_window_s=360.0,
):
    """
    Assign anchor to early/middle/late session time bin.

    method="thirds":
        Divide whole recording into equal thirds.

    method="fixed_windows":
        early  = first fixed_window_s seconds
        middle = fixed_window_s seconds centered on recording midpoint
        late   = last fixed_window_s seconds
        anything else = gap
    """

    if not np.isfinite(anchor_time_s) or not np.isfinite(session_duration_s):
        return ""

    if session_duration_s <= 0:
        return ""

    if method == "thirds":

        one_third = session_duration_s / 3.0
        two_third = 2.0 * session_duration_s / 3.0

        if anchor_time_s < one_third:
            return "early"
        elif anchor_time_s < two_third:
            return "middle"
        else:
            return "late"

    elif method == "fixed_windows":

        mid_time = session_duration_s / 2.0

        early_start = 0.0
        early_end = fixed_window_s

        middle_start = mid_time - fixed_window_s / 2.0
        middle_end = mid_time + fixed_window_s / 2.0

        late_start = session_duration_s - fixed_window_s
        late_end = session_duration_s

        if early_start <= anchor_time_s < early_end:
            return "early"
        elif middle_start <= anchor_time_s < middle_end:
            return "middle"
        elif late_start <= anchor_time_s <= late_end:
            return "late"
        else:
            return "gap"

    else:
        raise ValueError("method must be 'thirds' or 'fixed_windows'")


def build_prepost_epoch_windows(
    events_df,
    window_s=1.0,

    # pseudo tone timing
    add_pseudo_tone=True,
    pseudo_tone_on_offset_s=-3.0,
    pseudo_tone_off_offset_s=2.0,

    # middle anchors
    add_mid_phase_anchors=True,
    expected_off_duration_s=15.0,

    # overlap control
    check_overlap=True,
    invalidate_overlaps=True,
    max_allowed_overlap_s=0.020,

    # session time bins
    add_session_time_bin=True,
    session_time_bin_method="thirds",
    fixed_time_bin_window_s=360.0,
):
    """
    Build pre/post epoch windows from validated event QC table.

    Required columns in events_df:
        animal
        date
        phase
        event_number
        on_time_s
        off_time_s
        fs
        n_samples or number_of_samples
        valid_for_epoch_extraction

    Main anchors created:

    Habituation:
        LED_on
        LED_off
        pseudo_tone_on
        pseudo_tone_off
        LED_on_mid
        LED_off_mid

    Air training:
        pseudo_tone_on
        pseudo_tone_off
        air_on
        air_off
        air_on_mid
        air_off_mid

    Tone-air training:
        tone_on
        air_on
        tone_off
        air_off
        air_on_mid
        air_off_mid

    Notes:
        - pseudo_tone_on = on_time_s - 3 s
        - pseudo_tone_off = on_time_s + 2 s
        - ON middle = halfway between on_time_s and off_time_s
        - OFF middle = halfway between off_time_s and next on_time_s
        - If next on_time_s is missing, OFF middle uses expected_off_duration_s as fallback.
    """

    events_df = events_df.copy()

    # --------------------------------------------------
    # Sort events and add next_on_time_s for OFF middle
    # --------------------------------------------------
    events_df = events_df.sort_values(
        ["animal", "date", "event_number"]
    ).reset_index(drop=True)

    events_df["next_on_time_s"] = (
        events_df.groupby(["animal", "date"])["on_time_s"].shift(-1)
    )

    rows = []

    for _, ev in events_df.iterrows():

        animal = ev["animal"]
        date = ev["date"]
        phase = ev["phase"]
        event_number = ev["event_number"]

        fs = ev.get("fs", np.nan)
        # pull from event row
        rig_session_start_min = ev.get("rig_session_start_min", np.nan)
        phase_session_start_min = ev.get("phase_session_start_min", np.nan)
        recording_duration_min = ev.get("recording_duration_min", np.nan)
        good_session_basic = ev.get("good_session_basic", np.nan)
        rig_session_number = ev.get("rig_session_number", np.nan)
        phase_session_number = ev.get("phase_session_number", np.nan)
        rig_day_from_date = ev.get("rig_day_from_date", np.nan)
        phase_day_from_date = ev.get("phase_day_from_date", np.nan)

        if "n_samples" in ev:
            n_samples = ev.get("n_samples", np.nan)
        else:
            n_samples = ev.get("number_of_samples", np.nan)

        on_time = ev.get("on_time_s", np.nan)
        off_time = ev.get("off_time_s", np.nan)
        next_on_time = ev.get("next_on_time_s", np.nan)

        parent_valid = bool(ev.get("valid_for_epoch_extraction", False))

        if np.isfinite(fs) and np.isfinite(n_samples) and fs > 0:
            session_duration_s = float(n_samples) / float(fs)
        else:
            session_duration_s = np.nan

        anchors = []

        # --------------------------------------------------
        # Common derived anchor times
        # --------------------------------------------------
        pseudo_tone_on_time = (
            on_time + pseudo_tone_on_offset_s
            if np.isfinite(on_time)
            else np.nan
        )

        pseudo_tone_off_time = (
            on_time + pseudo_tone_off_offset_s
            if np.isfinite(on_time)
            else np.nan
        )

        on_mid_time = (
            (on_time + off_time) / 2.0
            if np.isfinite(on_time) and np.isfinite(off_time)
            else np.nan
        )

        if np.isfinite(off_time) and np.isfinite(next_on_time):
            off_mid_time = (off_time + next_on_time) / 2.0
            off_duration_s = next_on_time - off_time

        elif np.isfinite(off_time):
            off_mid_time = off_time + expected_off_duration_s / 2.0
            off_duration_s = expected_off_duration_s

        else:
            off_mid_time = np.nan
            off_duration_s = np.nan

        on_duration_s = (
            off_time - on_time
            if np.isfinite(on_time) and np.isfinite(off_time)
            else np.nan
        )

        # --------------------------------------------------
        # Define anchors by phase
        # --------------------------------------------------
        if phase == "habituation":

            anchors.extend([
                ("LED_on", on_time, "main"),
                ("LED_off", off_time, "main"),
            ])

            if add_pseudo_tone:
                anchors.extend([
                    ("pseudo_tone_on", pseudo_tone_on_time, "pseudo"),
                    ("pseudo_tone_off", pseudo_tone_off_time, "pseudo"),
                ])

            if add_mid_phase_anchors:
                anchors.extend([
                    ("LED_on_mid", on_mid_time, "middle"),
                    ("LED_off_mid", off_mid_time, "middle"),
                ])

        elif phase == "air_training":

            if add_pseudo_tone:
                anchors.extend([
                    ("pseudo_tone_on", pseudo_tone_on_time, "pseudo"),
                    ("pseudo_tone_off", pseudo_tone_off_time, "pseudo"),
                ])

            anchors.extend([
                ("air_on", on_time, "main"),
                ("air_off", off_time, "main"),
            ])

            if add_mid_phase_anchors:
                anchors.extend([
                    ("air_on_mid", on_mid_time, "middle"),
                    ("air_off_mid", off_mid_time, "middle"),
                ])

        elif phase == "tone_air_training":

            tone_on_time = (
                on_time - 3.0
                if np.isfinite(on_time)
                else np.nan
            )

            tone_off_time = (
                on_time + 2.0
                if np.isfinite(on_time)
                else np.nan
            )

            anchors.extend([
                ("tone_on", tone_on_time, "main"),
                ("air_on", on_time, "main"),
                ("tone_off", tone_off_time, "main"),
                ("air_off", off_time, "main"),
            ])

            if add_mid_phase_anchors:
                anchors.extend([
                    ("air_on_mid", on_mid_time, "middle"),
                    ("air_off_mid", off_mid_time, "middle"),
                ])

        else:
            continue

        # --------------------------------------------------
        # Make pre/post windows around each anchor
        # --------------------------------------------------
        for anchor_name, anchor_time, anchor_type in anchors:

            window_defs = [
                ("pre", anchor_time - window_s, anchor_time),
                ("post", anchor_time, anchor_time + window_s),
            ]

            for window_position, window_start_s, window_end_s in window_defs:

                epoch_name = f"{anchor_name}_{window_position}_{window_s:g}s"

                valid_window_basic = False
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
                        valid_window_basic = True

                # --------------------------------------------------
                # Optional anchor warnings
                # These do not automatically invalidate the window.
                # --------------------------------------------------
                anchor_warning = ""

                if anchor_name in ["air_on_mid", "LED_on_mid"]:

                    if np.isfinite(on_duration_s) and on_duration_s < 2.0 * window_s:
                        anchor_warning = _append_reason(
                            anchor_warning,
                            "on_duration_shorter_than_prepost_window"
                        )

                if anchor_name in ["air_off_mid", "LED_off_mid"]:

                    if np.isfinite(off_duration_s) and off_duration_s < 2.0 * window_s:
                        anchor_warning = _append_reason(
                            anchor_warning,
                            "off_duration_shorter_than_prepost_window"
                        )

                if anchor_name == "pseudo_tone_off":

                    if np.isfinite(pseudo_tone_off_time) and np.isfinite(off_time):
                        if pseudo_tone_off_time > off_time:
                            anchor_warning = _append_reason(
                                anchor_warning,
                                "pseudo_tone_off_after_air_or_LED_off"
                            )

                if add_session_time_bin:
                    session_time_bin = _assign_session_time_bin(
                        anchor_time,
                        session_duration_s,
                        method=session_time_bin_method,
                        fixed_window_s=fixed_time_bin_window_s,
                    )
                else:
                    session_time_bin = ""

                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "event_number": event_number,

                    "anchor_name": anchor_name,
                    "anchor_type": anchor_type,
                    "anchor_time_s": anchor_time,

                    "window_position": window_position,
                    "window_s": window_s,
                    "epoch_name": epoch_name,

                    "window_start_s": window_start_s,
                    "window_end_s": window_end_s,
                    "start_idx": start_idx,
                    "end_idx": end_idx,

                    "on_time_s": on_time,
                    "off_time_s": off_time,
                    "next_on_time_s": next_on_time,
                    "on_duration_s": on_duration_s,
                    "off_duration_s": off_duration_s,
                    "session_duration_s": session_duration_s,
                    "session_time_bin": session_time_bin,

                    "parent_event_valid": parent_valid,
                    "valid_window_basic": valid_window_basic,
                    "valid_window": valid_window_basic,

                    "overlap_flag": False,
                    "overlap_s": 0.0,
                    "overlap_with": "",

                    "anchor_warning": anchor_warning,
                    "invalid_reason": invalid_reason,

                    "good_session_basic": good_session_basic,

                    "rig_day_from_date": rig_day_from_date,
                    "rig_session_number": rig_session_number,
                    "phase_day_from_date": phase_day_from_date,
                    "phase_session_number": phase_session_number,

                    "rig_session_start_min": rig_session_start_min,
                    "phase_session_start_min": phase_session_start_min,
                    "recording_duration_min": recording_duration_min,

                    "anchor_session_time_min": anchor_time / 60.0 if np.isfinite(anchor_time) else np.nan,

                    "anchor_rig_time_min": (
                        rig_session_start_min + anchor_time / 60.0
                        if np.isfinite(rig_session_start_min) and np.isfinite(anchor_time)
                        else np.nan
                    ),

                    "anchor_phase_time_min": (
                        phase_session_start_min + anchor_time / 60.0
                        if np.isfinite(phase_session_start_min) and np.isfinite(anchor_time)
                        else np.nan
                    ),
                })

    windows_df = pd.DataFrame(rows)

    if windows_df.empty:
        return windows_df

    # --------------------------------------------------
    # Overlap checking
    # --------------------------------------------------
    if check_overlap:

        windows_df = windows_df.sort_values(
            ["animal", "date", "window_start_s", "window_end_s"]
        ).reset_index(drop=True)

        for (animal, date), g in windows_df.groupby(["animal", "date"]):

            valid_idx = g.index[g["valid_window_basic"] == True].tolist()

            valid_idx = sorted(
                valid_idx,
                key=lambda idx: windows_df.loc[idx, "window_start_s"]
            )

            # Check all windows that overlap in time.
            for i in range(len(valid_idx)):

                idx1 = valid_idx[i]

                start1 = windows_df.loc[idx1, "window_start_s"]
                end1 = windows_df.loc[idx1, "window_end_s"]
                name1 = windows_df.loc[idx1, "epoch_name"]

                for j in range(i + 1, len(valid_idx)):

                    idx2 = valid_idx[j]

                    start2 = windows_df.loc[idx2, "window_start_s"]
                    end2 = windows_df.loc[idx2, "window_end_s"]
                    name2 = windows_df.loc[idx2, "epoch_name"]

                    # Because windows are sorted, no later windows can overlap.
                    if start2 >= end1:
                        break

                    overlap_s = min(end1, end2) - max(start1, start2)

                    if overlap_s > max_allowed_overlap_s:

                        windows_df.loc[idx1, "overlap_flag"] = True
                        windows_df.loc[idx2, "overlap_flag"] = True

                        windows_df.loc[idx1, "overlap_s"] = max(
                            windows_df.loc[idx1, "overlap_s"],
                            overlap_s
                        )

                        windows_df.loc[idx2, "overlap_s"] = max(
                            windows_df.loc[idx2, "overlap_s"],
                            overlap_s
                        )

                        old1 = windows_df.loc[idx1, "overlap_with"]
                        old2 = windows_df.loc[idx2, "overlap_with"]

                        windows_df.loc[idx1, "overlap_with"] = (
                            name2 if old1 == "" else old1 + ";" + name2
                        )

                        windows_df.loc[idx2, "overlap_with"] = (
                            name1 if old2 == "" else old2 + ";" + name1
                        )

        if invalidate_overlaps:

            overlap_idx = windows_df.index[
                (windows_df["valid_window_basic"] == True)
                & (windows_df["overlap_flag"] == True)
            ]

            for idx in overlap_idx:
                windows_df.loc[idx, "valid_window"] = False
                windows_df.loc[idx, "invalid_reason"] = _append_reason(
                    windows_df.loc[idx, "invalid_reason"],
                    "overlaps_other_epoch"
                )

    # --------------------------------------------------
    # Final sorting
    # --------------------------------------------------
    windows_df = windows_df.sort_values(
        [
            "phase",
            "animal",
            "date",
            "event_number",
            "anchor_time_s",
            "anchor_name",
            "window_position",
        ]
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