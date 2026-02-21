import numpy as np


def compute_trial_ami(b, baseline_sec=2.0):
    """
    Compute Air Modulation Index per trial.

    Parameters
    ----------
    b : dict-like behavior struct
    baseline_sec : float
        seconds before air onset used for baseline

    Returns
    -------
    ami : np.ndarray
    speed_on : np.ndarray
    speed_off : np.ndarray
    """

    speed = np.asarray(b["speed_net_cms"])
    fs = float(b["fs"])
    Air_r = np.asarray(b["Air_r"], dtype=int)
    Air_f = np.asarray(b["Air_f"], dtype=int)
    print(f'DEBUG: {len(Air_r)}')
    baseline_samples = int(baseline_sec * fs)

    ami_list = []
    on_list = []
    off_list = []

    for r, f in zip(Air_r, Air_f):

        # --- bounds check ---
        if r <= baseline_samples:
            continue
        if f <= r:
            continue

        # --- windows ---
        off_slice = slice(r - baseline_samples, r)
        on_slice = slice(r, f)

        speed_off = np.nanmean(speed[off_slice])
        speed_on = np.nanmean(speed[on_slice])

        # avoid divide-by-zero
        denom = speed_on + speed_off
        if np.abs(denom) < 1e-6:
            continue

        ami = (speed_on - speed_off) / (denom + 1e-9)

        ami_list.append(ami)
        on_list.append(speed_on)
        off_list.append(speed_off)

    return (
        np.asarray(ami_list),
        np.asarray(on_list),
        np.asarray(off_list),
    )

def summarize_session_ami(b, baseline_sec=2.0):

    ami, s_on, s_off = compute_trial_ami(b, baseline_sec)

    if len(ami) == 0:
        return None

    return {
        "ami_mean": np.nanmean(ami),
        "ami_median": np.nanmedian(ami),
        "ami_std": np.nanstd(ami),
        "n_trials": len(ami),
        "speed_on_mean": np.nanmean(s_on),
        "speed_off_mean": np.nanmean(s_off),
    }