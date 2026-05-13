# src/utils/proc_paths.py

import numpy as np
import h5py
from pathlib import Path
from src.utils.config import PROC_BASE
import json


# =====================================================
# Session directory (mirror RAW → PROC)
# =====================================================

def get_proc_session_dir(animal, date, root=PROC_BASE):
    """
    Returns:
        /proc_base/NML_04/2026_01_12
    """
    root = Path(root).expanduser()
    out = root / animal / date
    out.mkdir(parents=True, exist_ok=True)
    return out


# =====================================================
# Subfolders
# =====================================================

def get_mp4_dir(animal, date, root=PROC_BASE):
    d = get_proc_session_dir(animal, date, root) / "mp4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_behavior_dir(animal, date, root=PROC_BASE):
    d = get_proc_session_dir(animal, date, root) / "behavior"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =====================================================
# Behavior save/load
# =====================================================

def behavior_npz_path(animal, date, root=PROC_BASE):
    """
    Full path to behavior NPZ.
    """
    return get_behavior_dir(animal, date, root) / "behavior_v1.npz"


def save_behavior_npz(b, animal, date, root=PROC_BASE):
    """
    Save behavior struct to compressed NPZ.
    """
    out_file = behavior_npz_path(animal, date, root)
    np.savez_compressed(out_file, **b)
    return str(out_file)


def load_behavior_npz(animal, date, root=PROC_BASE):
    """
    Load behavior struct if it exists.
    """
    file_path = behavior_npz_path(animal, date, root)

    if not file_path.exists():
        return None

    data = np.load(file_path, allow_pickle=True)
    return {k: data[k] for k in data.files}



# =====================================================
# Behavior H5 path
# =====================================================

def behavior_h5_path(animal, date, root=PROC_BASE):
    """
    Full path to behavior H5 file.
    """

    return get_behavior_dir(animal, date, root) / "behavior_v1.h5"


# =====================================================
# Save behavior dictionary to H5
# =====================================================

def save_behavior_h5(b, animal, date, root=PROC_BASE):
    """
    Save behavior dictionary to HDF5.
    """

    out_file = behavior_h5_path(animal, date, root)

    with h5py.File(out_file, "w") as hf:

        for key, value in b.items():

            # ---------- dictionaries ----------
            if isinstance(value, dict):
                hf.attrs[key] = json.dumps(value)

            # ---------- strings ----------
            elif isinstance(value, str):
                hf.attrs[key] = value

            # ---------- scalars ----------
            elif np.isscalar(value):
                hf.create_dataset(key, data=value)

            # ---------- arrays ----------
            else:
                arr = np.asarray(value)

                # skip object arrays safely
                if arr.dtype == object:
                    hf.attrs[key] = json.dumps(value)
                else:
                    hf.create_dataset(
                        key,
                        data=arr,
                        compression="gzip"
                    )

    return str(out_file)


# =====================================================
# Load behavior dictionary from H5
# =====================================================

def load_behavior_h5(animal, date, root=PROC_BASE):
    """
    Load behavior dictionary from HDF5.

    Returns
    -------
    dict or None
    """

    file_path = behavior_h5_path(animal, date, root)

    if not file_path.exists():
        return None

    out = {}

    with h5py.File(file_path, "r") as hf:

        for key in hf.keys():

            value = hf[key][()]

            # ---------- decode bytes ----------
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            out[key] = value

    return out


def list_behavior_h5_keys(animal, date, root=PROC_BASE):
    """
    List variables stored inside behavior_v1.h5.
    """

    file_path = behavior_h5_path(animal, date, root)

    if not file_path.exists():
        return None

    with h5py.File(file_path, "r") as hf:
        return list(hf.keys())
    

def load_behavior_h5(animal, date, keys=None, root=PROC_BASE):
    """
    Load behavior data from HDF5.

    Parameters
    ----------
    keys : None, str, or list of str
        None = load all variables
        str = load one variable
        list = load selected variables
    """

    file_path = behavior_h5_path(animal, date, root)

    if not file_path.exists():
        return None

    out = {}

    with h5py.File(file_path, "r") as hf:

        if keys is None:
            keys_to_load = list(hf.keys())

        elif isinstance(keys, str):
            keys_to_load = [keys]

        else:
            keys_to_load = keys

        for key in keys_to_load:

            if key not in hf:
                print(f"Variable not found in H5 file: {key}")
                continue

            value = hf[key][()]

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            out[key] = value

    return out

