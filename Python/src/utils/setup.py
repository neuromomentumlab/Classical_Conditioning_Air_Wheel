import os
from src.utils.config import PROC_BASE

def create_base_tree(verbose=True):
    """
    Create the standard processing directory tree.
    Safe to run multiple times.
    """

    dirs_to_create = [
        # "tsnr/maps",
        # "tsnr/figures",
        # "tsnr/stats",
        # "masks",
        # "fc/matrices",
        # "fc/figures",
        # "qc/motion",
        # "qc/reports",
        "logs",
        "figures",
    ]

    for d in dirs_to_create:
        full_path = os.path.join(PROC_BASE, d)
        os.makedirs(full_path, exist_ok=True)
        if verbose:
            print("Created:", full_path)

    if verbose:
        print("\n✅ Directory tree ready.")
