# src/utils/proc_paths.py

import numpy as np
from pathlib import Path
from src.utils.config import PROC_BASE


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