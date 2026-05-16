import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def view_cc_data(cc_data, print_summary=True, show_dates=False):
    """
    Summarize cc_data structure.

    Expected structure:
        cc_data[animal][date] = info_dict

    where info_dict may contain:
        phase
        recording
        video file paths
        other metadata
    """

    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y_%m_%d")
        except Exception:
            return None

    def path_exists(x):
        try:
            return Path(x).exists()
        except Exception:
            return False

    def is_video_file(x):
        try:
            suffix = Path(x).suffix.lower()
            return suffix in [".h264", ".mp4", ".avi", ".mov"]
        except Exception:
            return False

    rows = []

    for animal, days in cc_data.items():
        for date, info in days.items():

            phase = info.get("phase", "unknown")

            recording = info.get("recording", None)
            recording_exists = path_exists(recording) if recording is not None else False

            # Count possible video files from all values in info
            video_files = []

            for key, value in info.items():
                if isinstance(value, (str, Path)) and is_video_file(value):
                    video_files.append(str(value))

                elif isinstance(value, (list, tuple)):
                    for v in value:
                        if isinstance(v, (str, Path)) and is_video_file(v):
                            video_files.append(str(v))

            rows.append({
                "animal": animal,
                "date": date,
                "date_dt": parse_date(date),
                "phase": phase,
                "has_recording": recording is not None,
                "recording_exists": recording_exists,
                "n_video_files": len(video_files),
                "available_keys": ", ".join(sorted(info.keys()))
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("cc_data is empty.")
        return df, pd.DataFrame(), pd.DataFrame()

    df = df.sort_values(["animal", "date"]).reset_index(drop=True)

    # --------------------------------------------------
    # Phase-level summary
    # --------------------------------------------------
    phase_summary = (
        df.groupby("phase")
        .agg(
            n_animals=("animal", "nunique"),
            n_sessions=("date", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_recording_paths=("has_recording", "sum"),
            n_existing_recordings=("recording_exists", "sum"),
            total_video_files=("n_video_files", "sum"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Animal x phase summary
    # --------------------------------------------------
    animal_phase_summary = (
        df.groupby(["animal", "phase"])
        .agg(
            n_sessions=("date", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_recording_paths=("has_recording", "sum"),
            n_existing_recordings=("recording_exists", "sum"),
            total_video_files=("n_video_files", "sum"),
        )
        .reset_index()
    )

    # Add missing calendar dates within each animal/phase span
    missing_rows = []

    for _, row in animal_phase_summary.iterrows():
        animal = row["animal"]
        phase = row["phase"]

        sub = df[(df["animal"] == animal) & (df["phase"] == phase)].copy()
        dates_dt = sorted([d for d in sub["date_dt"].tolist() if d is not None])

        if len(dates_dt) == 0:
            missing_dates = []
            calendar_span_days = 0
        else:
            first_dt = dates_dt[0]
            last_dt = dates_dt[-1]

            all_dates = []
            d = first_dt
            while d <= last_dt:
                all_dates.append(d)
                d += timedelta(days=1)

            recorded_set = set(dates_dt)
            missing_dates = [d for d in all_dates if d not in recorded_set]
            calendar_span_days = len(all_dates)

        missing_rows.append({
            "animal": animal,
            "phase": phase,
            "calendar_span_days": calendar_span_days,
            "n_missing_calendar_days": len(missing_dates),
            "missing_calendar_dates": ", ".join(d.strftime("%Y_%m_%d") for d in missing_dates)
        })

    missing_df = pd.DataFrame(missing_rows)

    animal_phase_summary = animal_phase_summary.merge(
        missing_df,
        on=["animal", "phase"],
        how="left"
    )

    # --------------------------------------------------
    # Print simple summary
    # --------------------------------------------------
    if print_summary:
        print("=" * 60)
        print("CC DATA SUMMARY")
        print("=" * 60)
        print(f"Number of animals: {df['animal'].nunique()}")
        print(f"Number of total sessions: {len(df)}")
        print(f"Number of phases: {df['phase'].nunique()}")
        print()
        print("Phases present:")
        for phase in sorted(df["phase"].unique()):
            n_sess = (df["phase"] == phase).sum()
            n_animals = df.loc[df["phase"] == phase, "animal"].nunique()
            print(f"  - {phase}: {n_sess} sessions across {n_animals} animals")

        print()
        print("Animal x phase summary:")
        display(animal_phase_summary)

        if show_dates:
            print()
            print("Full session table:")
            display(df.drop(columns=["date_dt"]))

    return df, phase_summary, animal_phase_summary