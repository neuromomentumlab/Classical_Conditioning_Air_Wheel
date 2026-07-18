from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, savgol_filter, sosfiltfilt, welch

import src.utils.pdata_io as pdio


@dataclass
class WholeSessionSpectralConfig:
    """Configuration for corrected whole-session locomotor spectral analysis."""

    analysis_hz: float = 100.0
    derivative_window_ms: float = 20.0
    derivative_polyorder: int = 3
    speed_lowpass_hz: float = 20.0
    filter_order: int = 4

    # Fast PSD: within-locomotion fluctuations and high-frequency QC.
    fast_welch_window_s: float = 10.0
    fast_welch_overlap_fraction: float = 0.50
    fast_fmin_hz: float = 0.10
    fast_fmax_hz: float = 20.0

    # Slow PSD: bout, trial-cycle, and LED-cycle organization.
    slow_welch_window_s: float = 60.0
    slow_welch_overlap_fraction: float = 0.50
    slow_fmin_hz: float = 0.02
    slow_fmax_hz: float = 5.0

    behavior_window_s: float = 2.0
    movement_threshold_cms: float = 0.20

    fast_spectral_slope_fmin_hz: float = 1.0
    fast_spectral_slope_fmax_hz: float = 10.0

    # Transparent session-level artifact rules.
    qc_min_native_stored_r: float = 0.90
    qc_max_native_stored_rmse_cms: float = 1.0
    qc_max_abs_speed_cms: float = 100.0
    qc_max_native_position_step_cm: float = 2.0
    qc_large_encoder_jump_counts: float = 5.0


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

FAST_BANDS_HZ = {
    "very_slow_0p1_0p5": (0.10, 0.50),
    "slow_0p5_2": (0.50, 2.00),
    "intermediate_2_5": (2.00, 5.00),
    "high_5_10": (5.00, 10.00),
    "very_high_10_20": (10.00, 20.0000001),
}

SLOW_BANDS_HZ = {
    "cycle_0p02_0p05": (0.02, 0.05),
    "cycle_0p05_0p1": (0.05, 0.10),
    "very_slow_0p1_0p5": (0.10, 0.50),
    "slow_0p5_2": (0.50, 2.00),
    "intermediate_2_5": (2.00, 5.0000001),
}


def normalize_phase(phase: object) -> str:
    text = str(phase).strip().lower()
    return PHASE_ALIASES.get(text, text.replace("-", "_").replace(" ", "_"))


def _valid_odd_window(window_ms: float, fs: float, polyorder: int, signal_length: int) -> int:
    window = int(round(window_ms / 1000.0 * fs))
    if window % 2 == 0:
        window += 1

    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1

    maximum = signal_length if signal_length % 2 == 1 else signal_length - 1
    window = min(max(window, minimum), maximum)

    if window <= polyorder:
        raise ValueError("Signal is too short for the Savitzky-Golay derivative.")

    return window


def _resample_continuous(signal: np.ndarray, original_hz: float, target_hz: float) -> np.ndarray:
    ratio = Fraction(target_hz / original_hz).limit_denominator(100000)
    return resample_poly(
        np.asarray(signal, dtype=np.float64),
        up=ratio.numerator,
        down=ratio.denominator,
    )


def _downsample_last_sample(signal: np.ndarray, original_hz: float, target_hz: float) -> np.ndarray:
    ratio = original_hz / target_hz
    block_size = int(round(ratio))

    if np.isclose(ratio, block_size):
        n_blocks = len(signal) // block_size
        trimmed = np.asarray(signal)[: n_blocks * block_size]
        return trimmed.reshape(n_blocks, block_size)[:, -1]

    original_time = np.arange(len(signal)) / original_hz
    target_time = np.arange(0.0, original_time[-1], 1.0 / target_hz)
    return np.interp(target_time, original_time, signal)


def _downsample_binary(signal: np.ndarray, original_hz: float, target_hz: float) -> np.ndarray:
    ratio = original_hz / target_hz
    block_size = int(round(ratio))

    if np.isclose(ratio, block_size):
        n_blocks = len(signal) // block_size
        trimmed = np.asarray(signal)[: n_blocks * block_size]
        occupancy = trimmed.reshape(n_blocks, block_size).mean(axis=1)
        return (occupancy >= 0.5).astype(np.int8)

    return (
        _resample_continuous(signal, original_hz, target_hz) >= 0.5
    ).astype(np.int8)


def _derive_native_rate(
    cumulative_position_cm: np.ndarray,
    fs: float,
    config: WholeSessionSpectralConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    cumulative_position_cm = np.asarray(cumulative_position_cm, dtype=np.float64)

    derivative_window = _valid_odd_window(
        config.derivative_window_ms,
        fs,
        config.derivative_polyorder,
        len(cumulative_position_cm),
    )

    rate_native = savgol_filter(
        cumulative_position_cm,
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
    rate_native = sosfiltfilt(sos, rate_native)

    return rate_native, {
        "derivative_window_samples": derivative_window,
        "derivative_window_ms_actual": derivative_window / fs * 1000.0,
    }


def _calculate_welch_psd(
    signal: np.ndarray,
    fs: float,
    window_s: float,
    overlap_fraction: float,
    fmin_hz: float,
    fmax_hz: float,
) -> pd.DataFrame:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 8:
        raise ValueError("Not enough finite samples for Welch PSD.")

    nperseg = min(len(signal), max(8, int(round(window_s * fs))))
    noverlap = min(int(round(nperseg * overlap_fraction)), nperseg - 1)

    frequency_hz, power = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    keep = (frequency_hz >= fmin_hz) & (frequency_hz <= fmax_hz)
    return pd.DataFrame({"frequency_hz": frequency_hz[keep], "power": power[keep]})


def _bin_integrated_power(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> float:
    """Integrate nonoverlapping equally-spaced bins using sum(power) * df."""

    if len(frequency_hz) < 2:
        return np.nan

    df_hz = float(np.median(np.diff(frequency_hz)))
    keep = (frequency_hz >= lower_hz) & (frequency_hz < upper_hz)
    return float(np.sum(power[keep]) * df_hz) if np.any(keep) else np.nan


def _summarize_psd(
    psd_df: pd.DataFrame,
    prefix: str,
    bands_hz: Dict[str, Tuple[float, float]],
    slope_range_hz: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    frequency_hz = psd_df["frequency_hz"].to_numpy(dtype=np.float64)
    power = psd_df["power"].to_numpy(dtype=np.float64)

    finite = np.isfinite(frequency_hz) & np.isfinite(power) & (power >= 0)
    frequency_hz = frequency_hz[finite]
    power = power[finite]

    if len(frequency_hz) < 2:
        return {}

    df_hz = float(np.median(np.diff(frequency_hz)))
    total_power = float(np.sum(power) * df_hz)

    if total_power <= 0:
        return {
            f"{prefix}_total_power": np.nan,
            f"{prefix}_dominant_frequency_hz": np.nan,
            f"{prefix}_spectral_centroid_hz": np.nan,
            f"{prefix}_spectral_entropy": np.nan,
            f"{prefix}_relative_band_power_sum": np.nan,
        }

    probability = power / np.sum(power)
    summary = {
        f"{prefix}_total_power": total_power,
        f"{prefix}_dominant_frequency_hz": float(frequency_hz[np.argmax(power)]),
        f"{prefix}_spectral_centroid_hz": float(
            np.sum(frequency_hz * power) / np.sum(power)
        ),
        f"{prefix}_spectral_entropy": float(
            -np.sum(probability * np.log(probability + np.finfo(float).eps))
            / np.log(len(probability))
        ),
    }

    if slope_range_hz is not None:
        slope_mask = (
            (frequency_hz >= slope_range_hz[0])
            & (frequency_hz <= slope_range_hz[1])
            & (power > 0)
        )
        summary[f"{prefix}_spectral_slope"] = (
            float(
                np.polyfit(
                    np.log10(frequency_hz[slope_mask]),
                    np.log10(power[slope_mask]),
                    deg=1,
                )[0]
            )
            if np.count_nonzero(slope_mask) >= 3
            else np.nan
        )

    relative_sum = 0.0
    for band_name, (lower_hz, upper_hz) in bands_hz.items():
        absolute = _bin_integrated_power(frequency_hz, power, lower_hz, upper_hz)
        relative = absolute / total_power if np.isfinite(absolute) else np.nan
        summary[f"{prefix}_power_{band_name}"] = absolute
        summary[f"{prefix}_relative_power_{band_name}"] = relative
        if np.isfinite(relative):
            relative_sum += relative

    summary[f"{prefix}_relative_band_power_sum"] = relative_sum
    return summary


def _window_behavior_summary(
    signed_speed_cms: np.ndarray,
    encoder_count_100hz: Optional[np.ndarray],
    config: WholeSessionSpectralConfig,
) -> Dict[str, float]:
    """Use total encoder-count change, not count range, to define immobility."""

    window_samples = int(round(config.behavior_window_s * config.analysis_hz))
    n_windows = len(signed_speed_cms) // window_samples

    base = {
        "number_of_behavior_windows": n_windows,
        "fraction_moving_2s_windows": np.nan,
        "median_2s_window_mean_abs_speed_cms": np.nan,
        "median_moving_2s_window_mean_abs_speed_cms": np.nan,
        "strict_immobile_2s_windows": np.nan,
        "strict_immobile_2s_mean_abs_speed_p99_cms": np.nan,
        "near_immobile_2s_windows": np.nan,
        "near_immobile_2s_mean_abs_speed_p99_cms": np.nan,
    }

    if n_windows == 0:
        return base

    speed_windows = signed_speed_cms[: n_windows * window_samples].reshape(
        n_windows, window_samples
    )
    mean_abs_speed = np.mean(np.abs(speed_windows), axis=1)
    moving = mean_abs_speed >= config.movement_threshold_cms

    base.update(
        {
            "fraction_moving_2s_windows": float(np.mean(moving)),
            "median_2s_window_mean_abs_speed_cms": float(np.median(mean_abs_speed)),
            "median_moving_2s_window_mean_abs_speed_cms": (
                float(np.median(mean_abs_speed[moving])) if np.any(moving) else np.nan
            ),
        }
    )

    if encoder_count_100hz is None:
        return base

    encoder_windows = encoder_count_100hz[: n_windows * window_samples].reshape(
        n_windows, window_samples
    )
    total_count_change = np.sum(np.abs(np.diff(encoder_windows, axis=1)), axis=1)
    strict = total_count_change == 0
    near = total_count_change <= 1

    base.update(
        {
            "strict_immobile_2s_windows": int(np.sum(strict)),
            "strict_immobile_2s_mean_abs_speed_p99_cms": (
                float(np.quantile(mean_abs_speed[strict], 0.99)) if np.any(strict) else np.nan
            ),
            "near_immobile_2s_windows": int(np.sum(near)),
            "near_immobile_2s_mean_abs_speed_p99_cms": (
                float(np.quantile(mean_abs_speed[near], 0.99)) if np.any(near) else np.nan
            ),
        }
    )
    return base


def _artifact_metrics(
    position_native: np.ndarray,
    encoder_native: Optional[np.ndarray],
    signed_speed_100hz: np.ndarray,
    stored_speed_100hz: Optional[np.ndarray],
    config: WholeSessionSpectralConfig,
) -> Dict[str, object]:
    position_steps = np.diff(np.asarray(position_native, dtype=np.float64))
    max_position_step = float(np.max(np.abs(position_steps))) if len(position_steps) else np.nan

    if encoder_native is not None:
        encoder_steps = np.diff(np.asarray(encoder_native, dtype=np.float64))
        max_encoder_step = float(np.max(np.abs(encoder_steps))) if len(encoder_steps) else np.nan
        n_large_jumps = int(
            np.sum(np.abs(encoder_steps) > config.qc_large_encoder_jump_counts)
        )
    else:
        max_encoder_step = np.nan
        n_large_jumps = np.nan

    absolute_speed = np.abs(signed_speed_100hz)
    metrics = {
        "maximum_abs_speed_cms": float(np.max(absolute_speed)),
        "p999_abs_speed_cms": float(np.quantile(absolute_speed, 0.999)),
        "maximum_native_position_step_cm": max_position_step,
        "maximum_encoder_count_step": max_encoder_step,
        "number_of_large_count_jumps": n_large_jumps,
        "native_vs_stored_speed_r": np.nan,
        "native_vs_stored_speed_rmse_cms": np.nan,
    }

    if stored_speed_100hz is not None:
        n = min(len(stored_speed_100hz), len(signed_speed_100hz))
        stored = stored_speed_100hz[:n]
        native = signed_speed_100hz[:n]
        finite = np.isfinite(stored) & np.isfinite(native)
        if np.count_nonzero(finite) >= 3:
            metrics["native_vs_stored_speed_r"] = float(
                np.corrcoef(native[finite], stored[finite])[0, 1]
            )
            metrics["native_vs_stored_speed_rmse_cms"] = float(
                np.sqrt(np.mean((native[finite] - stored[finite]) ** 2))
            )

    reasons = []
    r_value = metrics["native_vs_stored_speed_r"]
    rmse = metrics["native_vs_stored_speed_rmse_cms"]

    if np.isfinite(r_value) and r_value < config.qc_min_native_stored_r:
        reasons.append(f"native_vs_stored_r<{config.qc_min_native_stored_r:g}")
    if np.isfinite(rmse) and rmse > config.qc_max_native_stored_rmse_cms:
        reasons.append(f"native_vs_stored_rmse>{config.qc_max_native_stored_rmse_cms:g}")
    if metrics["maximum_abs_speed_cms"] > config.qc_max_abs_speed_cms:
        reasons.append(f"maximum_abs_speed>{config.qc_max_abs_speed_cms:g}")
    if np.isfinite(max_position_step) and max_position_step > config.qc_max_native_position_step_cm:
        reasons.append("large_native_position_step")

    metrics["qc_pass"] = len(reasons) == 0
    metrics["qc_reason"] = "; ".join(reasons)
    return metrics


def _iter_cc_sessions(
    cc_data: dict,
    animals: Optional[Iterable[str]] = None,
    phases: Optional[Iterable[str]] = None,
):
    animal_filter = set(animals) if animals is not None else None
    phase_filter = {normalize_phase(x) for x in phases} if phases is not None else None
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

            sessions.append(
                {
                    "animal": str(animal),
                    "date": str(date),
                    "date_parsed": pd.to_datetime(str(date), errors="coerce"),
                    "phase_raw": str(phase_raw),
                    "phase": phase,
                }
            )

    sessions.sort(
        key=lambda item: (
            item["animal"],
            item["date_parsed"] if pd.notna(item["date_parsed"]) else pd.Timestamp.max,
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return session features, long-form PSDs, and processing errors."""

    config = WholeSessionSpectralConfig() if config is None else config
    load_kwargs = {} if root is None else {"root": root}
    sessions = list(_iter_cc_sessions(cc_data, animals=animals, phases=phases))

    rows = []
    psd_tables = []
    errors = []

    for index, session in enumerate(sessions, start=1):
        animal, date, phase = session["animal"], session["date"], session["phase"]

        if verbose:
            print(f"[{index}/{len(sessions)}] {animal} | {date} | {phase}")

        try:
            available = pdio.list_behavior_h5_keys(animal, date, **load_kwargs)
            if available is None:
                raise FileNotFoundError(f"No behavior_v1.h5 found for {animal}, {date}.")

            required = ["fs", "dist_net_cm"]
            missing = [key for key in required if key not in available]
            if missing:
                raise KeyError(f"Missing required H5 variables: {missing}")

            optional = [
                key
                for key in [
                    "dist_path_cm",
                    "speed_net_cms",
                    "speed_path_cms",
                    "encoderCount",
                    "air_bin",
                    "trial_forward_cm",
                ]
                if key in available
            ]
            behavior = pdio.load_behavior_h5(
                animal,
                date,
                keys=required + optional,
                **load_kwargs,
            )

            fs = float(np.asarray(behavior["fs"]).squeeze())
            net_position_native = np.asarray(behavior["dist_net_cm"], dtype=np.float64)
            signed_native, derivative_metadata = _derive_native_rate(
                net_position_native, fs, config
            )
            signed_speed = _resample_continuous(signed_native, fs, config.analysis_hz)

            if "dist_path_cm" in behavior:
                path_position_native = np.asarray(behavior["dist_path_cm"], dtype=np.float64)
                path_native, _ = _derive_native_rate(path_position_native, fs, config)
                path_speed_raw = _resample_continuous(path_native, fs, config.analysis_hz)
                path_negative_fraction = float(np.mean(path_speed_raw < 0))
                path_speed = np.maximum(path_speed_raw, 0.0)
                path_source = "derived_from_dist_path_cm"
            else:
                path_speed = np.abs(signed_speed)
                path_negative_fraction = np.nan
                path_source = "fallback_abs_signed_speed"

            absolute_signed_speed = np.abs(signed_speed)
            net_position_100hz = _downsample_last_sample(
                net_position_native, fs, config.analysis_hz
            )

            encoder_native = None
            encoder_100hz = None
            if "encoderCount" in behavior:
                encoder_native = np.asarray(behavior["encoderCount"], dtype=np.float64)
                encoder_100hz = _downsample_last_sample(
                    encoder_native, fs, config.analysis_hz
                )

            control_100hz = None
            if "air_bin" in behavior:
                control_100hz = _downsample_binary(
                    np.asarray(behavior["air_bin"]), fs, config.analysis_hz
                )

            stored_speed_100hz = None
            if "speed_net_cms" in behavior:
                stored_speed_100hz = _resample_continuous(
                    np.asarray(behavior["speed_net_cms"], dtype=np.float64),
                    fs,
                    config.analysis_hz,
                )

            lengths = [
                len(signed_speed),
                len(path_speed),
                len(net_position_100hz),
            ]
            for optional_signal in (encoder_100hz, control_100hz, stored_speed_100hz):
                if optional_signal is not None:
                    lengths.append(len(optional_signal))
            n_samples = min(lengths)

            signed_speed = signed_speed[:n_samples]
            path_speed = path_speed[:n_samples]
            absolute_signed_speed = absolute_signed_speed[:n_samples]
            net_position_100hz = net_position_100hz[:n_samples]
            if encoder_100hz is not None:
                encoder_100hz = encoder_100hz[:n_samples]
            if control_100hz is not None:
                control_100hz = control_100hz[:n_samples]
            if stored_speed_100hz is not None:
                stored_speed_100hz = stored_speed_100hz[:n_samples]

            signal_map = {
                "signed_speed": signed_speed,
                "path_speed": path_speed,
                "absolute_signed_speed": absolute_signed_speed,
            }
            psd_lookup = {}

            for signal_name, signal in signal_map.items():
                psd_lookup[(signal_name, "fast")] = _calculate_welch_psd(
                    signal,
                    config.analysis_hz,
                    config.fast_welch_window_s,
                    config.fast_welch_overlap_fraction,
                    config.fast_fmin_hz,
                    config.fast_fmax_hz,
                )
                psd_lookup[(signal_name, "slow")] = _calculate_welch_psd(
                    signal,
                    config.analysis_hz,
                    config.slow_welch_window_s,
                    config.slow_welch_overlap_fraction,
                    config.slow_fmin_hz,
                    config.slow_fmax_hz,
                )

            artifacts = _artifact_metrics(
                net_position_native,
                encoder_native,
                signed_speed,
                stored_speed_100hz,
                config,
            )
            duration_s = n_samples / config.analysis_hz

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
                "number_of_samples_100hz": n_samples,
                "path_speed_source": path_source,
                "path_speed_negative_fraction_preclip": path_negative_fraction,
                "net_displacement_cm": float(
                    net_position_100hz[-1] - net_position_100hz[0]
                ),
                "path_distance_from_true_path_speed_cm": float(
                    np.sum(path_speed) / config.analysis_hz
                ),
                "path_distance_from_abs_signed_speed_cm": float(
                    np.sum(absolute_signed_speed) / config.analysis_hz
                ),
                "mean_signed_speed_cms": float(np.mean(signed_speed)),
                "median_signed_speed_cms": float(np.median(signed_speed)),
                "std_signed_speed_cms": float(np.std(signed_speed)),
                "rms_signed_speed_cms": float(np.sqrt(np.mean(signed_speed ** 2))),
                "mean_abs_signed_speed_cms": float(np.mean(absolute_signed_speed)),
                "median_abs_signed_speed_cms": float(np.median(absolute_signed_speed)),
                "mean_true_path_speed_cms": float(np.mean(path_speed)),
                "median_true_path_speed_cms": float(np.median(path_speed)),
                "p95_abs_signed_speed_cms": float(
                    np.quantile(absolute_signed_speed, 0.95)
                ),
                "fraction_forward_samples_gt_threshold": float(
                    np.mean(signed_speed >= config.movement_threshold_cms)
                ),
                "fraction_backward_samples_lt_minus_threshold": float(
                    np.mean(signed_speed <= -config.movement_threshold_cms)
                ),
                "fraction_moving_samples_abs_gt_threshold": float(
                    np.mean(absolute_signed_speed >= config.movement_threshold_cms)
                ),
                "control_on_fraction": (
                    float(np.mean(control_100hz)) if control_100hz is not None else np.nan
                ),
                "trial_forward_cm_median": (
                    float(np.median(np.asarray(behavior["trial_forward_cm"], dtype=np.float64)))
                    if "trial_forward_cm" in behavior
                    else np.nan
                ),
                **derivative_metadata,
                **_window_behavior_summary(signed_speed, encoder_100hz, config),
                **artifacts,
            }

            for signal_name in signal_map:
                row.update(
                    _summarize_psd(
                        psd_lookup[(signal_name, "fast")],
                        prefix=f"{signal_name}_fast",
                        bands_hz=FAST_BANDS_HZ,
                        slope_range_hz=(
                            config.fast_spectral_slope_fmin_hz,
                            config.fast_spectral_slope_fmax_hz,
                        ),
                    )
                )
                row.update(
                    _summarize_psd(
                        psd_lookup[(signal_name, "slow")],
                        prefix=f"{signal_name}_slow",
                        bands_hz=SLOW_BANDS_HZ,
                    )
                )

            rows.append(row)

            for (signal_name, resolution), psd_df in psd_lookup.items():
                psd_copy = psd_df.copy()
                psd_copy.insert(0, "resolution", resolution)
                psd_copy.insert(0, "signal", signal_name)
                psd_copy.insert(0, "phase", phase)
                psd_copy.insert(0, "date", date)
                psd_copy.insert(0, "animal", animal)

                if len(psd_copy) >= 2:
                    df_hz = float(np.median(np.diff(psd_copy["frequency_hz"])))
                    total = float(np.sum(psd_copy["power"]) * df_hz)
                else:
                    total = np.nan

                psd_copy["relative_power_density"] = (
                    psd_copy["power"] / total
                    if np.isfinite(total) and total > 0
                    else np.nan
                )
                psd_copy["qc_pass"] = artifacts["qc_pass"]
                psd_copy["qc_reason"] = artifacts["qc_reason"]
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
                print(f"  SKIPPED: {type(exc).__name__}: {exc}")

    whole_session_df = pd.DataFrame(rows)
    whole_session_psd_df = (
        pd.concat(psd_tables, ignore_index=True) if psd_tables else pd.DataFrame()
    )
    errors_df = pd.DataFrame(errors)

    if not whole_session_df.empty:
        whole_session_df["_date_sort"] = pd.to_datetime(
            whole_session_df["date"], errors="coerce"
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
        whole_session_df["phase_n_sessions"] = whole_session_df.groupby(
            ["animal", "phase"]
        )["date"].transform("size")
        denominator = whole_session_df["phase_n_sessions"] - 1
        whole_session_df["phase_progress_0_1"] = np.where(
            denominator > 0,
            (whole_session_df["phase_session"] - 1) / denominator,
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
    return whole_session_df, whole_session_psd_df, errors_df


def save_whole_session_spectral_tables(
    whole_session_df: pd.DataFrame,
    whole_session_psd_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_dir,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "whole_session": output_dir / "whole_session_spectral_features_corrected.csv",
        "psd_long": output_dir / "whole_session_psd_long_corrected.csv",
        "errors": output_dir / "whole_session_spectral_errors_corrected.csv",
        "qc_failed": output_dir / "whole_session_qc_failed_sessions.csv",
    }

    whole_session_df.to_csv(paths["whole_session"], index=False)
    whole_session_psd_df.to_csv(paths["psd_long"], index=False)
    errors_df.to_csv(paths["errors"], index=False)

    qc_failed = (
        whole_session_df.loc[~whole_session_df["qc_pass"].fillna(False)].copy()
        if "qc_pass" in whole_session_df.columns
        else pd.DataFrame()
    )
    qc_failed.to_csv(paths["qc_failed"], index=False)
    return paths
