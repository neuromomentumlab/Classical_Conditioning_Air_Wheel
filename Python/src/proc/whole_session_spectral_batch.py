from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, savgol_filter, sosfiltfilt, welch

import src.utils.pdata_io as pdio


@dataclass
class WholeSessionSpectralConfig:
    """Settings for batch whole-session locomotor spectral analysis."""

    analysis_hz: float = 100.0

    derivative_window_ms: float = 20.0
    derivative_polyorder: int = 3

    speed_lowpass_hz: float = 20.0
    filter_order: int = 4

    welch_window_s: float = 10.0
    welch_overlap_fraction: float = 0.50

    spectral_fmin_hz: float = 0.10
    spectral_fmax_hz: float = 20.0

    behavior_window_s: float = 2.0
    movement_threshold_cms: float = 0.20

    spectral_slope_fmin_hz: float = 1.0
    spectral_slope_fmax_hz: float = 10.0


PHASE_ALIASES = {
    "habituation": "repeated_exposure",
    "repeated_exposure": "repeated_exposure",
    "repeated exposure": "repeated_exposure",
    "repeated-exposure": "repeated_exposure",
    "air_training": "air_training",
    "air training": "air_training",
    "air-training": "air_training",
    "tone_air_training": "tone_air_training",
    "tone air training": "tone_air_training",
    "tone-air training": "tone_air_training",
}

PHASE_ORDER = {
    "repeated_exposure": 0,
    "air_training": 1,
    "tone_air_training": 2,
}

BANDS_HZ = {
    "very_slow_0p1_0p5": (0.10, 0.50),
    "slow_0p5_2": (0.50, 2.00),
    "intermediate_2_5": (2.00, 5.00),
    "high_5_10": (5.00, 10.00),
    "very_high_10_20": (10.00, 20.00),
}


def normalize_phase(phase: object) -> str:
    """Convert phase labels to a consistent internal representation."""

    text = str(phase).strip().lower()
    text = text.replace("__", "_")

    return PHASE_ALIASES.get(
        text,
        text.replace("-", "_").replace(" ", "_"),
    )


def _valid_odd_window(
    window_ms: float,
    fs: float,
    polyorder: int,
    signal_length: int,
) -> int:
    window = int(round(window_ms / 1000.0 * fs))

    if window % 2 == 0:
        window += 1

    minimum_window = polyorder + 2
    if minimum_window % 2 == 0:
        minimum_window += 1

    window = max(window, minimum_window)

    maximum_window = (
        signal_length
        if signal_length % 2 == 1
        else signal_length - 1
    )
    window = min(window, maximum_window)

    if window <= polyorder:
        raise ValueError(
            "Signal is too short for the requested Savitzky-Golay derivative."
        )

    return window


def _resample_continuous(
    signal: np.ndarray,
    original_hz: float,
    target_hz: float,
) -> np.ndarray:
    ratio = Fraction(target_hz / original_hz).limit_denominator(100000)

    return resample_poly(
        np.asarray(signal, dtype=np.float64),
        up=ratio.numerator,
        down=ratio.denominator,
    )


def _downsample_last_sample(
    signal: np.ndarray,
    original_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Downsample cumulative/count signals by keeping the last sample per block."""

    ratio = original_hz / target_hz
    block_size = int(round(ratio))

    if np.isclose(ratio, block_size):
        number_of_blocks = len(signal) // block_size
        trimmed = np.asarray(signal)[: number_of_blocks * block_size]

        return trimmed.reshape(number_of_blocks, block_size)[:, -1]

    original_time = np.arange(len(signal)) / original_hz
    target_time = np.arange(
        0.0,
        original_time[-1],
        1.0 / target_hz,
    )

    return np.interp(target_time, original_time, signal)


def _downsample_binary(
    signal: np.ndarray,
    original_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Downsample a binary signal by majority occupancy within each block."""

    ratio = original_hz / target_hz
    block_size = int(round(ratio))

    if np.isclose(ratio, block_size):
        number_of_blocks = len(signal) // block_size
        trimmed = np.asarray(signal)[: number_of_blocks * block_size]
        occupancy = trimmed.reshape(number_of_blocks, block_size).mean(axis=1)

        return (occupancy >= 0.5).astype(np.int8)

    resampled = _resample_continuous(
        signal,
        original_hz=original_hz,
        target_hz=target_hz,
    )

    return (resampled >= 0.5).astype(np.int8)


def _derive_native_speed(
    position_cm: np.ndarray,
    fs: float,
    config: WholeSessionSpectralConfig,
) -> tuple[np.ndarray, dict]:
    position_cm = np.asarray(position_cm, dtype=np.float64)

    derivative_window = _valid_odd_window(
        window_ms=config.derivative_window_ms,
        fs=fs,
        polyorder=config.derivative_polyorder,
        signal_length=len(position_cm),
    )

    speed_native = savgol_filter(
        position_cm,
        window_length=derivative_window,
        polyorder=config.derivative_polyorder,
        deriv=1,
        delta=1.0 / fs,
        mode="interp",
    )

    sos = butter(
        N=config.filter_order,
        Wn=config.speed_lowpass_hz,
        btype="lowpass",
        fs=fs,
        output="sos",
    )

    speed_native = sosfiltfilt(sos, speed_native)

    metadata = {
        "derivative_window_samples": derivative_window,
        "derivative_window_ms_actual": derivative_window / fs * 1000.0,
    }

    return speed_native, metadata


def _calculate_welch_psd(
    signal: np.ndarray,
    fs: float,
    config: WholeSessionSpectralConfig,
) -> pd.DataFrame:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 8:
        raise ValueError("Not enough finite samples for Welch PSD.")

    nperseg = min(
        len(signal),
        max(8, int(round(config.welch_window_s * fs))),
    )

    noverlap = int(round(nperseg * config.welch_overlap_fraction))
    noverlap = min(noverlap, nperseg - 1)

    frequency_hz, power = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    keep = (
        (frequency_hz >= config.spectral_fmin_hz)
        & (frequency_hz <= config.spectral_fmax_hz)
    )

    return pd.DataFrame(
        {
            "frequency_hz": frequency_hz[keep],
            "power": power[keep],
        }
    )


def _band_power(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    lower_hz: float,
    upper_hz: float,
    include_upper: bool = False,
) -> float:
    if include_upper:
        keep = (frequency_hz >= lower_hz) & (frequency_hz <= upper_hz)
    else:
        keep = (frequency_hz >= lower_hz) & (frequency_hz < upper_hz)

    if np.count_nonzero(keep) < 2:
        return np.nan

    return float(np.trapz(power[keep], frequency_hz[keep]))


def _summarize_psd(
    psd_df: pd.DataFrame,
    prefix: str,
    config: WholeSessionSpectralConfig,
) -> dict:
    frequency_hz = psd_df["frequency_hz"].to_numpy(dtype=np.float64)
    power = psd_df["power"].to_numpy(dtype=np.float64)

    finite = (
        np.isfinite(frequency_hz)
        & np.isfinite(power)
        & (power >= 0)
    )

    frequency_hz = frequency_hz[finite]
    power = power[finite]

    total_power = float(np.trapz(power, frequency_hz))

    if total_power <= 0:
        return {
            f"{prefix}_total_power_0p1_20": np.nan,
            f"{prefix}_dominant_frequency_hz": np.nan,
            f"{prefix}_spectral_centroid_hz": np.nan,
            f"{prefix}_spectral_entropy": np.nan,
            f"{prefix}_spectral_slope_1_10": np.nan,
        }

    discrete_probability = power / power.sum()

    centroid = float(
        np.trapz(frequency_hz * power, frequency_hz) / total_power
    )

    entropy = float(
        -np.sum(
            discrete_probability
            * np.log(discrete_probability + np.finfo(float).eps)
        )
        / np.log(len(discrete_probability))
    )

    dominant_frequency = float(
        frequency_hz[np.argmax(power)]
    )

    slope_mask = (
        (frequency_hz >= config.spectral_slope_fmin_hz)
        & (frequency_hz <= config.spectral_slope_fmax_hz)
        & (power > 0)
    )

    if np.count_nonzero(slope_mask) >= 3:
        slope = float(
            np.polyfit(
                np.log10(frequency_hz[slope_mask]),
                np.log10(power[slope_mask]),
                deg=1,
            )[0]
        )
    else:
        slope = np.nan

    summary = {
        f"{prefix}_total_power_0p1_20": total_power,
        f"{prefix}_dominant_frequency_hz": dominant_frequency,
        f"{prefix}_spectral_centroid_hz": centroid,
        f"{prefix}_spectral_entropy": entropy,
        f"{prefix}_spectral_slope_1_10": slope,
    }

    for index, (band_name, (lower_hz, upper_hz)) in enumerate(
        BANDS_HZ.items()
    ):
        absolute_power = _band_power(
            frequency_hz,
            power,
            lower_hz=lower_hz,
            upper_hz=upper_hz,
            include_upper=(index == len(BANDS_HZ) - 1),
        )

        summary[f"{prefix}_power_{band_name}"] = absolute_power
        summary[f"{prefix}_relative_power_{band_name}"] = (
            absolute_power / total_power
            if np.isfinite(absolute_power)
            else np.nan
        )

    return summary


def _window_behavior_summary(
    signed_speed_cms: np.ndarray,
    encoder_count_100hz: Optional[np.ndarray],
    config: WholeSessionSpectralConfig,
) -> dict:
    window_samples = int(
        round(config.behavior_window_s * config.analysis_hz)
    )

    number_of_windows = len(signed_speed_cms) // window_samples

    if number_of_windows == 0:
        return {
            "number_of_behavior_windows": 0,
            "fraction_moving_2s_windows": np.nan,
            "median_2s_window_mean_abs_speed_cms": np.nan,
            "median_moving_2s_window_mean_abs_speed_cms": np.nan,
            "near_immobile_2s_windows": np.nan,
            "near_immobile_2s_mean_abs_speed_p99_cms": np.nan,
        }

    trimmed = signed_speed_cms[
        : number_of_windows * window_samples
    ]

    speed_windows = trimmed.reshape(
        number_of_windows,
        window_samples,
    )

    mean_abs_speed = np.mean(
        np.abs(speed_windows),
        axis=1,
    )

    moving = (
        mean_abs_speed
        >= config.movement_threshold_cms
    )

    summary = {
        "number_of_behavior_windows": number_of_windows,
        "fraction_moving_2s_windows": float(np.mean(moving)),
        "median_2s_window_mean_abs_speed_cms": float(
            np.median(mean_abs_speed)
        ),
        "median_moving_2s_window_mean_abs_speed_cms": (
            float(np.median(mean_abs_speed[moving]))
            if np.any(moving)
            else np.nan
        ),
    }

    if encoder_count_100hz is None:
        summary.update(
            {
                "near_immobile_2s_windows": np.nan,
                "near_immobile_2s_mean_abs_speed_p99_cms": np.nan,
            }
        )
        return summary

    encoder_trimmed = encoder_count_100hz[
        : number_of_windows * window_samples
    ]

    encoder_windows = encoder_trimmed.reshape(
        number_of_windows,
        window_samples,
    )

    count_range = np.ptp(
        encoder_windows,
        axis=1,
    )

    near_immobile = count_range <= 1

    summary.update(
        {
            "near_immobile_2s_windows": int(np.sum(near_immobile)),
            "near_immobile_2s_mean_abs_speed_p99_cms": (
                float(np.quantile(mean_abs_speed[near_immobile], 0.99))
                if np.any(near_immobile)
                else np.nan
            ),
        }
    )

    return summary


def _iter_cc_sessions(
    cc_data: dict,
    animals: Optional[Iterable[str]] = None,
    phases: Optional[Iterable[str]] = None,
):
    animal_filter = set(animals) if animals is not None else None
    phase_filter = (
        {normalize_phase(phase) for phase in phases}
        if phases is not None
        else None
    )

    sessions = []

    for animal, animal_data in cc_data.items():
        if animal_filter is not None and animal not in animal_filter:
            continue

        if not isinstance(animal_data, dict):
            continue

        for date, session_info in animal_data.items():
            if not isinstance(session_info, dict):
                continue

            phase_raw = session_info.get("phase")
            if phase_raw is None:
                continue

            phase = normalize_phase(phase_raw)

            if phase_filter is not None and phase not in phase_filter:
                continue

            parsed_date = pd.to_datetime(
                str(date),
                errors="coerce",
            )

            sessions.append(
                {
                    "animal": str(animal),
                    "date": str(date),
                    "date_parsed": parsed_date,
                    "phase_raw": str(phase_raw),
                    "phase": phase,
                    "session_info": session_info,
                }
            )

    sessions.sort(
        key=lambda item: (
            item["animal"],
            (
                item["date_parsed"]
                if pd.notna(item["date_parsed"])
                else pd.Timestamp.max
            ),
            item["date"],
        )
    )

    yield from sessions


def build_whole_session_spectral_tables(
    cc_data: dict,
    config: Optional[WholeSessionSpectralConfig] = None,
    root=None,
    animals: Optional[Iterable[str]] = None,
    phases: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process all behavior_v1.h5 sessions represented in cc_data.

    Returns
    -------
    whole_session_df
        One row per animal-session.

    whole_session_psd_df
        Long-form PSD table. Each session has rows for signed speed and
        absolute/path speed.

    errors_df
        Sessions that could not be processed, with error messages.
    """

    if config is None:
        config = WholeSessionSpectralConfig()

    load_kwargs = {}
    if root is not None:
        load_kwargs["root"] = root

    sessions = list(
        _iter_cc_sessions(
            cc_data,
            animals=animals,
            phases=phases,
        )
    )

    rows = []
    psd_tables = []
    errors = []

    for session_number, session in enumerate(sessions, start=1):
        animal = session["animal"]
        date = session["date"]
        phase = session["phase"]

        if verbose:
            print(
                f"[{session_number}/{len(sessions)}] "
                f"{animal} | {date} | {phase}"
            )

        try:
            available_keys = pdio.list_behavior_h5_keys(
                animal,
                date,
                **load_kwargs,
            )

            if available_keys is None:
                raise FileNotFoundError(
                    f"No behavior_v1.h5 found for {animal}, {date}."
                )

            required_keys = [
                "fs",
                "dist_net_cm",
            ]

            missing = [
                key
                for key in required_keys
                if key not in available_keys
            ]

            if missing:
                raise KeyError(
                    f"Missing required H5 variables: {missing}"
                )

            optional_keys = [
                key
                for key in [
                    "speed_net_cms",
                    "dist_path_cm",
                    "encoderCount",
                    "air_bin",
                    "trial_forward_cm",
                ]
                if key in available_keys
            ]

            behavior = pdio.load_behavior_h5(
                animal,
                date,
                keys=required_keys + optional_keys,
                **load_kwargs,
            )

            fs = float(
                np.asarray(behavior["fs"]).squeeze()
            )

            position_native = np.asarray(
                behavior["dist_net_cm"],
                dtype=np.float64,
            )

            speed_native, derivative_metadata = _derive_native_speed(
                position_native,
                fs=fs,
                config=config,
            )

            signed_speed = _resample_continuous(
                speed_native,
                original_hz=fs,
                target_hz=config.analysis_hz,
            )

            position_100hz = _downsample_last_sample(
                position_native,
                original_hz=fs,
                target_hz=config.analysis_hz,
            )

            encoder_100hz = None
            if "encoderCount" in behavior:
                encoder_100hz = _downsample_last_sample(
                    np.asarray(behavior["encoderCount"]),
                    original_hz=fs,
                    target_hz=config.analysis_hz,
                )

            control_100hz = None
            if "air_bin" in behavior:
                control_100hz = _downsample_binary(
                    np.asarray(behavior["air_bin"]),
                    original_hz=fs,
                    target_hz=config.analysis_hz,
                )

            lengths = [
                len(signed_speed),
                len(position_100hz),
            ]

            if encoder_100hz is not None:
                lengths.append(len(encoder_100hz))

            if control_100hz is not None:
                lengths.append(len(control_100hz))

            number_of_samples = min(lengths)

            signed_speed = signed_speed[:number_of_samples]
            path_speed = np.abs(signed_speed)
            position_100hz = position_100hz[:number_of_samples]

            if encoder_100hz is not None:
                encoder_100hz = encoder_100hz[:number_of_samples]

            if control_100hz is not None:
                control_100hz = control_100hz[:number_of_samples]

            signed_psd = _calculate_welch_psd(
                signed_speed,
                fs=config.analysis_hz,
                config=config,
            )

            path_psd = _calculate_welch_psd(
                path_speed,
                fs=config.analysis_hz,
                config=config,
            )

            duration_s = number_of_samples / config.analysis_hz

            row = {
                "animal": animal,
                "date": date,
                "phase_raw": session["phase_raw"],
                "phase": phase,
                "phase_order": PHASE_ORDER.get(phase, np.nan),
                "duration_s": duration_s,
                "duration_min": duration_s / 60.0,
                "original_sampling_hz": fs,
                "analysis_sampling_hz": config.analysis_hz,
                "number_of_samples_100hz": number_of_samples,
                "net_displacement_cm": float(
                    position_100hz[-1] - position_100hz[0]
                ),
                "path_distance_from_native_speed_cm": float(
                    np.sum(path_speed) / config.analysis_hz
                ),
                "mean_signed_speed_cms": float(np.mean(signed_speed)),
                "median_signed_speed_cms": float(np.median(signed_speed)),
                "std_signed_speed_cms": float(np.std(signed_speed)),
                "rms_signed_speed_cms": float(
                    np.sqrt(np.mean(signed_speed ** 2))
                ),
                "mean_abs_speed_cms": float(np.mean(path_speed)),
                "median_abs_speed_cms": float(np.median(path_speed)),
                "p95_abs_speed_cms": float(np.quantile(path_speed, 0.95)),
                "fraction_forward_samples_gt_threshold": float(
                    np.mean(signed_speed >= config.movement_threshold_cms)
                ),
                "fraction_backward_samples_lt_minus_threshold": float(
                    np.mean(signed_speed <= -config.movement_threshold_cms)
                ),
                "fraction_moving_samples_abs_gt_threshold": float(
                    np.mean(path_speed >= config.movement_threshold_cms)
                ),
                "control_on_fraction": (
                    float(np.mean(control_100hz))
                    if control_100hz is not None
                    else np.nan
                ),
                "trial_forward_cm_median": (
                    float(
                        np.median(
                            np.asarray(
                                behavior["trial_forward_cm"],
                                dtype=np.float64,
                            )
                        )
                    )
                    if "trial_forward_cm" in behavior
                    else np.nan
                ),
                **derivative_metadata,
                **_window_behavior_summary(
                    signed_speed,
                    encoder_count_100hz=encoder_100hz,
                    config=config,
                ),
                **_summarize_psd(
                    signed_psd,
                    prefix="signed",
                    config=config,
                ),
                **_summarize_psd(
                    path_psd,
                    prefix="path",
                    config=config,
                ),
            }

            if "speed_net_cms" in behavior:
                stored_speed = _resample_continuous(
                    np.asarray(
                        behavior["speed_net_cms"],
                        dtype=np.float64,
                    ),
                    original_hz=fs,
                    target_hz=config.analysis_hz,
                )

                comparison_length = min(
                    len(stored_speed),
                    len(signed_speed),
                )

                stored_speed = stored_speed[:comparison_length]
                native_comparison = signed_speed[:comparison_length]

                finite = (
                    np.isfinite(stored_speed)
                    & np.isfinite(native_comparison)
                )

                if np.count_nonzero(finite) >= 3:
                    row["native_vs_stored_speed_r"] = float(
                        np.corrcoef(
                            native_comparison[finite],
                            stored_speed[finite],
                        )[0, 1]
                    )
                    row["native_vs_stored_speed_rmse_cms"] = float(
                        np.sqrt(
                            np.mean(
                                (
                                    native_comparison[finite]
                                    - stored_speed[finite]
                                )
                                ** 2
                            )
                        )
                    )
                else:
                    row["native_vs_stored_speed_r"] = np.nan
                    row["native_vs_stored_speed_rmse_cms"] = np.nan

            rows.append(row)

            for signal_name, psd_df in [
                ("signed_speed", signed_psd),
                ("path_speed", path_psd),
            ]:
                psd_copy = psd_df.copy()
                psd_copy.insert(0, "signal", signal_name)
                psd_copy.insert(0, "phase", phase)
                psd_copy.insert(0, "date", date)
                psd_copy.insert(0, "animal", animal)

                total_power = np.trapz(
                    psd_copy["power"],
                    psd_copy["frequency_hz"],
                )

                psd_copy["relative_power_density"] = (
                    psd_copy["power"] / total_power
                    if total_power > 0
                    else np.nan
                )

                psd_tables.append(psd_copy)

        except Exception as exc:
            errors.append(
                {
                    "animal": animal,
                    "date": date,
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            if verbose:
                print(
                    f"  SKIPPED: {type(exc).__name__}: {exc}"
                )

    whole_session_df = pd.DataFrame(rows)
    whole_session_psd_df = (
        pd.concat(psd_tables, ignore_index=True)
        if psd_tables
        else pd.DataFrame()
    )
    errors_df = pd.DataFrame(errors)

    if not whole_session_df.empty:
        whole_session_df["_date_sort"] = pd.to_datetime(
            whole_session_df["date"],
            errors="coerce",
        )

        whole_session_df = whole_session_df.sort_values(
            ["animal", "_date_sort", "date"]
        ).reset_index(drop=True)

        whole_session_df["animal_session"] = (
            whole_session_df.groupby("animal").cumcount() + 1
        )

        whole_session_df["phase_session"] = (
            whole_session_df.groupby(["animal", "phase"]).cumcount() + 1
        )

        whole_session_df["phase_n_sessions"] = (
            whole_session_df.groupby(["animal", "phase"])["date"]
            .transform("size")
        )

        denominator = whole_session_df["phase_n_sessions"] - 1

        whole_session_df["phase_progress_0_1"] = np.where(
            denominator > 0,
            (
                whole_session_df["phase_session"] - 1
            )
            / denominator,
            0.0,
        )

        whole_session_df = whole_session_df.drop(columns="_date_sort")

        if not whole_session_psd_df.empty:
            index_columns = whole_session_df[
                [
                    "animal",
                    "date",
                    "animal_session",
                    "phase_session",
                    "phase_n_sessions",
                    "phase_progress_0_1",
                ]
            ]

            whole_session_psd_df = whole_session_psd_df.merge(
                index_columns,
                on=["animal", "date"],
                how="left",
                validate="many_to_one",
            )

    whole_session_df.attrs["config"] = asdict(config)

    return (
        whole_session_df,
        whole_session_psd_df,
        errors_df,
    )


def save_whole_session_spectral_tables(
    whole_session_df: pd.DataFrame,
    whole_session_psd_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_dir,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "whole_session": output_dir / "whole_session_spectral_features.csv",
        "psd_long": output_dir / "whole_session_psd_long.csv",
        "errors": output_dir / "whole_session_spectral_errors.csv",
    }

    whole_session_df.to_csv(paths["whole_session"], index=False)
    whole_session_psd_df.to_csv(paths["psd_long"], index=False)
    errors_df.to_csv(paths["errors"], index=False)

    return paths
