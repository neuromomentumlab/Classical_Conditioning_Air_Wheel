import numpy as np
import pandas as pd


def make_session_availability_summary(
    events_df,
    windows_df=None,
    min_session_duration_s=1100,
    min_valid_events=3,
):
    """
    Build one-row-per-session availability/QC summary.

    One row = animal/date/phase.
    """

    ev = events_df.copy()

    # --------------------------------------------------
    # Estimate session duration if not already present
    # --------------------------------------------------
    if "session_duration_s" not in ev.columns:
        if "n_samples" in ev.columns:
            n_samples_col = "n_samples"
        elif "number_of_samples" in ev.columns:
            n_samples_col = "number_of_samples"
        else:
            n_samples_col = None

        if n_samples_col is not None and "fs" in ev.columns:
            ev["session_duration_s"] = ev[n_samples_col] / ev["fs"]
        else:
            ev["session_duration_s"] = np.nan

    # --------------------------------------------------
    # Event-level session summary
    # --------------------------------------------------
    session_df = (
        ev
        .groupby(["animal", "date", "phase"], dropna=False)
        .agg(
            session_duration_s=("session_duration_s", "max"),
            n_events_total=("event_number", "count"),
            n_events_valid=("valid_for_epoch_extraction", "sum"),
        )
        .reset_index()
    )

    session_df["short_recording"] = (
        session_df["session_duration_s"] < min_session_duration_s
    )

    session_df["enough_valid_events"] = (
        session_df["n_events_valid"] >= min_valid_events
    )

    session_df["good_session_basic"] = (
        (~session_df["short_recording"]) &
        (session_df["enough_valid_events"])
    )

    # --------------------------------------------------
    # Add valid-window availability if windows_df is provided
    # --------------------------------------------------
    if windows_df is not None and not windows_df.empty:

        w = windows_df.copy()

        valid_w = w[w["valid_window"] == True].copy()

        anchor_counts = (
            valid_w
            .groupby(
                ["animal", "date", "phase", "anchor_name", "window_position"],
                dropna=False
            )
            .size()
            .reset_index(name="n_windows")
        )

        anchor_pivot = anchor_counts.pivot_table(
            index=["animal", "date", "phase"],
            columns=["anchor_name", "window_position"],
            values="n_windows",
            fill_value=0,
            aggfunc="sum"
        )

        anchor_pivot.columns = [
            f"nwin_{anchor}_{pos}"
            for anchor, pos in anchor_pivot.columns
        ]

        anchor_pivot = anchor_pivot.reset_index()

        session_df = session_df.merge(
            anchor_pivot,
            on=["animal", "date", "phase"],
            how="left"
        )

        nwin_cols = [c for c in session_df.columns if c.startswith("nwin_")]
        session_df[nwin_cols] = session_df[nwin_cols].fillna(0).astype(int)

        session_df["n_valid_windows_total"] = session_df[nwin_cols].sum(axis=1)

    else:
        session_df["n_valid_windows_total"] = np.nan

    return session_df

import numpy as np
import pandas as pd


def add_day_bins_to_sessions(
    session_availability_df,
    use_good_sessions_only=True,
):
    """
    Add phase-specific day numbers, normalized day values, and early/middle/late bins.

    Day binning is done separately for each animal and phase.
    """

    df = session_availability_df.copy()

    # Convert date if needed
    df["date_dt"] = pd.to_datetime(
        df["date"].astype(str).str.replace("_", "-"),
        errors="coerce"
    )

    # Use only good sessions for assigning learning days if desired
    if use_good_sessions_only:
        df_for_day = df[df["good_session_basic"] == True].copy()
    else:
        df_for_day = df.copy()

    df_for_day = df_for_day.sort_values(
        ["phase", "animal", "date_dt", "date"]
    ).reset_index(drop=True)

    # Session number within each animal × phase
    df_for_day["phase_day_number_good"] = (
        df_for_day
        .groupby(["phase", "animal"])
        .cumcount() + 1
    )

    df_for_day["n_good_sessions_in_phase"] = (
        df_for_day
        .groupby(["phase", "animal"])["date"]
        .transform("count")
    )

    # Normalized day: 0 = first good session, 1 = last good session
    df_for_day["normalized_phase_day_good"] = np.where(
        df_for_day["n_good_sessions_in_phase"] > 1,
        (df_for_day["phase_day_number_good"] - 1)
        / (df_for_day["n_good_sessions_in_phase"] - 1),
        0.0
    )

    # Equal-count early/middle/late bins
    # This divides available good sessions into approximate thirds.
    bin_id = np.floor(
        (df_for_day["phase_day_number_good"] - 1) * 3
        / df_for_day["n_good_sessions_in_phase"]
    ).astype(int)

    bin_id = np.clip(bin_id, 0, 2)

    bin_map = {
        0: "early",
        1: "middle",
        2: "late",
    }

    df_for_day["phase_day_bin"] = bin_id.map(bin_map)

    # Merge day info back into the full session table
    merge_cols = [
        "animal",
        "date",
        "phase",
        "phase_day_number_good",
        "n_good_sessions_in_phase",
        "normalized_phase_day_good",
        "phase_day_bin",
    ]

    df = df.merge(
        df_for_day[merge_cols],
        on=["animal", "date", "phase"],
        how="left"
    )

    return df

def build_validated_epoch_matrix(
    encoder_epoch_df,
    session_day_df,
    keep_good_sessions_only=True,
    keep_valid_windows_only=True,
):
    """
    Attach validated session/day information to the event/window-level encoder metrics.

    One row = one valid animal/date/event/anchor/pre-post window.
    """

    df = encoder_epoch_df.copy()

    key_cols = ["animal", "date", "phase"]

    session_cols_wanted = [
        "good_session_basic",
        "short_recording",
        "enough_valid_events",
        "session_duration_s",
        "n_events_total",
        "n_events_valid",
        "n_valid_windows_total",
        "phase_day_number_good",
        "n_good_sessions_in_phase",
        "normalized_phase_day_good",
        "phase_day_bin",
    ]

    session_cols = [
        c for c in session_cols_wanted
        if c in session_day_df.columns
    ]

    sess = (
        session_day_df[key_cols + session_cols]
        .drop_duplicates(subset=key_cols)
        .copy()
    )

    # Remove older versions of these columns to avoid _x/_y merge problems
    cols_to_drop = [
        c for c in session_cols
        if c in df.columns
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    df = df.merge(
        sess,
        on=key_cols,
        how="left",
        validate="many_to_one"
    )

    if keep_valid_windows_only and "valid_window" in df.columns:
        df = df[df["valid_window"] == True].copy()

    if keep_good_sessions_only and "good_session_basic" in df.columns:
        df = df[df["good_session_basic"] == True].copy()

    df = df.sort_values(
        [
            "phase",
            "animal",
            "phase_day_number_good",
            "date",
            "event_number",
            "anchor_time_s",
            "anchor_name",
            "window_position",
        ],
        na_position="last"
    ).reset_index(drop=True)

    return df

def summarize_epoch_matrix(
    epoch_matrix_df,
    group_cols,
):
    """
    Average encoder metrics using user-defined grouping columns.

    This can create:
        session-level summaries
        animal-level summaries
        day-bin summaries
        session-time-bin summaries
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

    available_metrics = [
        m for m in metrics
        if m in epoch_matrix_df.columns
    ]

    group_cols = [
        c for c in group_cols
        if c in epoch_matrix_df.columns
    ]

    summary_df = (
        epoch_matrix_df
        .groupby(group_cols, dropna=False)
        .agg(
            **{m: (m, "mean") for m in available_metrics},
            n_windows=("epoch_name", "count"),
            n_events=("event_number", "nunique"),
            n_dates=("date", "nunique"),
        )
        .reset_index()
    )

    return summary_df

import numpy as np
import pandas as pd


def make_anchor_window_table(
    encoder_epoch_df,
    phase=None,
    anchor_name=None,
    metrics=None,
    return_type="wide",   # "wide", "delta", "pre", "post", or "long"
    keep_overlap=True,
    require_good_session=True,
    require_valid_window=True,
):
    """
    General function to extract pre, post, or post-pre delta values
    for any anchor and phase.

    One row in wide/delta output =
        animal × date × event_number × anchor_name

    Parameters
    ----------
    encoder_epoch_df : pandas.DataFrame
        Epoch-level metrics table.

    phase : str, list, or None
        Phase(s) to include.
        Example: "tone_air_training" or ["air_training", "tone_air_training"].
        If None, use all phases.

    anchor_name : str, list, or None
        Anchor(s) to include.
        Example: "tone_on" or ["tone_on", "air_on"].
        If None, use all anchors.

    metrics : list or None
        Metrics to extract. If None, uses common encoder metrics.

    return_type : str
        "wide"  = includes pre, post, and delta columns
        "delta" = includes only delta columns
        "pre"   = returns only pre rows in long format
        "post"  = returns only post rows in long format
        "long"  = returns filtered long table without pivoting

    keep_overlap : bool
        If False, removes rows with overlap_flag == True.

    require_good_session : bool
        If True, keeps only good_session_basic == True.

    require_valid_window : bool
        If True, keeps only valid_window == True.

    Returns
    -------
    pandas.DataFrame
    """

    df = encoder_epoch_df.copy()

    # --------------------------------------------------
    # Phase filter
    # --------------------------------------------------
    if phase is not None:
        if isinstance(phase, str):
            phase = [phase]
        df = df[df["phase"].isin(phase)].copy()

    # --------------------------------------------------
    # Anchor filter
    # --------------------------------------------------
    if anchor_name is not None:
        if isinstance(anchor_name, str):
            anchor_name = [anchor_name]
        df = df[df["anchor_name"].isin(anchor_name)].copy()

    # --------------------------------------------------
    # Window filter
    # --------------------------------------------------
    df = df[df["window_position"].isin(["pre", "post"])].copy()

    if require_good_session and "good_session_basic" in df.columns:
        df = df[df["good_session_basic"] == True].copy()

    if require_valid_window and "valid_window" in df.columns:
        df = df[df["valid_window"] == True].copy()

    if not keep_overlap and "overlap_flag" in df.columns:
        df = df[df["overlap_flag"] == False].copy()

    # --------------------------------------------------
    # Default metrics
    # --------------------------------------------------
    if metrics is None:
        metrics = [
            "mean_speed_path_cms",
            "median_speed_path_cms",
            "peak_speed_path_cms",
            "mean_speed_net_cms",
            "median_speed_net_cms",
            "distance_path_cm",
            "distance_net_cm",
            "frac_stationary",
            "frac_moving",
            "frac_forward",
            "frac_backward",
            "frac_low_net_movement",
            "net_direction_bias",
        ]

    metrics = [m for m in metrics if m in df.columns]

    if len(metrics) == 0:
        raise ValueError("No requested metrics were found in encoder_epoch_df.")

    # --------------------------------------------------
    # Return long/pre/post directly if requested
    # --------------------------------------------------
    if return_type == "long":
        return df.reset_index(drop=True)

    if return_type in ["pre", "post"]:
        return (
            df[df["window_position"] == return_type]
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # ID columns for wide/delta table
    # --------------------------------------------------
    candidate_id_cols = [
        "animal",
        "date",
        "phase",
        "event_number",
        "anchor_name",
        "anchor_type",

        # calendar/session indices
        "rig_day_from_date",
        "rig_session_number",
        "phase_day_from_date",
        "phase_session_number",

        # session and exposure timing
        "rig_session_start_min",
        "phase_session_start_min",
        "recording_duration_min",
        "anchor_time_s",
        "anchor_session_time_min",
        "anchor_phase_time_min",
        "anchor_rig_time_min",

        # QC
        "good_session_basic",
        "overlap_flag",
    ]

    id_cols = [c for c in candidate_id_cols if c in df.columns]

    # --------------------------------------------------
    # Pivot pre/post into columns
    # --------------------------------------------------
    wide = df.pivot_table(
        index=id_cols,
        columns="window_position",
        values=metrics,
        aggfunc="mean",
    )

    wide.columns = [
        f"{metric}_{position}"
        for metric, position in wide.columns
    ]

    wide = wide.reset_index()

    # --------------------------------------------------
    # Add deltas
    # --------------------------------------------------
    for metric in metrics:
        pre_col = f"{metric}_pre"
        post_col = f"{metric}_post"

        if pre_col in wide.columns and post_col in wide.columns:
            wide[f"delta_{metric}"] = wide[post_col] - wide[pre_col]
            # wide[f"delta_{metric"] = wide[post_col] - wide[pre_col]

    # --------------------------------------------------
    # Add useful LME time variables
    # --------------------------------------------------
    if "phase_session_start_min" in wide.columns:
        wide["phase_prior_exposure_min"] = wide["phase_session_start_min"]

    if "rig_session_start_min" in wide.columns:
        wide["rig_prior_exposure_min"] = wide["rig_session_start_min"]

    if "anchor_session_time_min" in wide.columns:
        wide["session_time_min"] = wide["anchor_session_time_min"]

    # Scaled/centered variables for LME
    if "phase_prior_exposure_min" in wide.columns:
        wide["phase_prior_100m_c"] = (
            wide["phase_prior_exposure_min"]
            - wide["phase_prior_exposure_min"].mean()
        ) / 100.0

    if "rig_prior_exposure_min" in wide.columns:
        wide["rig_prior_100m_c"] = (
            wide["rig_prior_exposure_min"]
            - wide["rig_prior_exposure_min"].mean()
        ) / 100.0

    if "session_time_min" in wide.columns:
        wide["session_10m_c"] = (
            wide["session_time_min"]
            - wide["session_time_min"].mean()
        ) / 10.0

    # Animal-day/session ID for LME
    wide["animal_day"] = (
        wide["animal"].astype(str) + ":" + wide["date"].astype(str)
    )

    # --------------------------------------------------
    # Return requested format
    # --------------------------------------------------
    if return_type == "wide":
        return wide.reset_index(drop=True)

    if return_type == "delta":
        delta_cols = [c for c in wide.columns if c.startswith("delta_")]

        keep_cols = [
            c for c in wide.columns
            if (
                c in id_cols
                or c in [
                    "animal_day",
                    "phase_prior_exposure_min",
                    "rig_prior_exposure_min",
                    "session_time_min",
                    "phase_prior_100m_c",
                    "rig_prior_100m_c",
                    "session_10m_c",
                ]
            )
        ]

        return wide[keep_cols + delta_cols].reset_index(drop=True)

    raise ValueError(
        "return_type must be one of: 'wide', 'delta', 'pre', 'post', 'long'"
    )