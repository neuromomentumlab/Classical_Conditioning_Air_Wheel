import os
import numpy as np
from src.utils.config import PROC_BASE


""" LOADING Processed DAta """

def load_behavior_npz(animal, date, root="data/processed_behavior"):
    """
    Load behavior struct if it exists.
    """

    from pathlib import Path
    import numpy as np

    file_path = Path(root) / animal / f"{date}_behavior.npz"

    if not file_path.exists():
        return None

    data = np.load(file_path, allow_pickle=True)

    return {k: data[k] for k in data.files}


""" SAVING Processed DAta """

def save_behavior_npz(b, animal, date, root="data/processed_behavior"):
    """
    Save behavior struct to compressed NPZ.
    """

    out_dir = Path(root) / animal
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{date}_behavior.npz"

    np.savez_compressed(out_file, **b)

    return str(out_file)




