import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.utils.pdata_io as pdio


def check_recording_durations(
    cc_data,
    phase_filter=None,
    target_duration_s=1200,
    tolerance_s=30,
    verbose=True
):
    """
    Check total recording duration for all sessions listed in cc_data.

    Loads only lightweight timing variables from the H5 behavior files.

    Parameters
    ----------
    cc_data : dict
        cc_data[animal][date] = info

    phase_filter : None, str, or list
        If None, include all phases.
        If str/list, include only selected phase(s).

    target_duration_s : float
        Expected recording duration, e.g. 1200 s.

    tolerance_s : float
        Allowed deviation from target duration.

    Returns
    -------
    df : pandas.DataFrame
        Summary table of recording durations.
    """

    if phase_filter is not None:
        if isinstance(phase_filter, str):
            phase_filter = {phase_filter}
        else:
            phase_filter = set(phase_filter)

    rows = []

    for animal, days in cc_data.items():
        for date, info in days.items():

            phase = info.get("phase", "unknown")

            if phase_filter is not None and phase not in phase_filter:
                continue

            try:
                b = pdio.load_behavior_h5(
                    animal,
                    date,
                    keys=["t", "fs", "number_of_samples"]
                )

                t = np.asarray(b["t"])
                fs = float(b["fs"])
                n_samples = int(b["number_of_samples"])

                duration_from_samples = n_samples / fs
                duration_from_t = t[-1] - t[0]

                duration_error_s = duration_from_samples - target_duration_s

                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "fs": fs,
                    "n_samples": n_samples,
                    "duration_from_samples_s": duration_from_samples,
                    "duration_from_t_s": duration_from_t,
                    "duration_error_s": duration_error_s,
                    "abs_error_s": abs(duration_error_s),
                    "near_target": abs(duration_error_s) <= tolerance_s,
                    "status": "ok"
                })

            except Exception as e:
                rows.append({
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "fs": np.nan,
                    "n_samples": np.nan,
                    "duration_from_samples_s": np.nan,
                    "duration_from_t_s": np.nan,
                    "duration_error_s": np.nan,
                    "abs_error_s": np.nan,
                    "near_target": False,
                    "status": f"error: {e}"
                })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["animal", "phase", "date"]).reset_index(drop=True)

    if verbose:
        print("=" * 70)
        print("Recording duration summary")
        print("=" * 70)
        print(f"Target duration: {target_duration_s} s")
        print(f"Tolerance: ±{tolerance_s} s")
        print()
        print(f"Total sessions checked: {len(df)}")
        print(f"Sessions near target: {df['near_target'].sum()}")
        print(f"Sessions outside target: {(~df['near_target']).sum()}")
        print()

        if "phase" in df.columns:
            display(
                df.groupby("phase")
                .agg(
                    n_sessions=("date", "count"),
                    mean_duration_s=("duration_from_samples_s", "mean"),
                    min_duration_s=("duration_from_samples_s", "min"),
                    max_duration_s=("duration_from_samples_s", "max"),
                    n_near_target=("near_target", "sum")
                )
                .reset_index()
            )

    return df