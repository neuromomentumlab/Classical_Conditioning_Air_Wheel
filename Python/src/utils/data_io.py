from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any


def build_classical_conditioning_dict(raw_root: str | Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build hierarchical dictionary for Classical Conditioning dataset.

    Structure
    ---------
    data_dict[animal][date] = {
        'face': str | None,
        'pupi': str | None,
        'video': str | None,
        'recording': str | None,
        'phase': str,
        'path': str
    }

    Notes
    -----
    - Missing files remain None (safe for pipelines)
    - Phase classification is based on date string
    - Only folders matching 'NML*' are treated as animals
    """
    # ---------- check mount ----------
    raw_root = Path(raw_root)
    if not raw_root.exists():
        print(f"\nDATA DIRECTORY NOT MOUNTED:\n{raw_root}\n")
        return {}
    data_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ---------- EDITABLE PHASE BOUNDARIES ----------
    DEFAULT_HABITUATION_END = "2026_01_11"
    DEFAULT_AIR_END = "2026_01_27"

    # ---------- ANIMAL-SPECIFIC PHASE BOUNDARIES ----------
    # Edit these dates for NML_07 and NML_08
    PHASE_BOUNDARIES = {
        "NML_07": {
            "habituation_end": "2026_02_20",
            "air_end": "2026_03_10",
        },
        "NML_08": {
            "habituation_end": "2026_02_20",
            "air_end": "2026_03_10",
        },
    }

    def classify_phase(animal: str, date_str: str) -> str:
        boundaries = PHASE_BOUNDARIES.get(
            animal,
            {
                "habituation_end": DEFAULT_HABITUATION_END,
                "air_end": DEFAULT_AIR_END,
            },
        )

        habituation_end = boundaries["habituation_end"]
        air_end = boundaries["air_end"]

        if date_str <= habituation_end:
            return "habituation"
        elif date_str <= air_end:
            return "air_training"
        else:
            return "unknown"

    # def classify_phase(date_str: str) -> str:
    #     if date_str <= HABITUATION_END:
    #         return "habituation"
    #     elif date_str <= AIR_END:
    #         return "air_training"
    #     else:
    #         return "unknown"

    # ---------- iterate animals ----------
    for animal_dir in sorted(raw_root.glob("NML*")):
        if not animal_dir.is_dir():
            continue

        animal = animal_dir.name
        data_dict[animal] = {}

        # ---------- iterate dates ----------
        for date_dir in sorted(animal_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            date_name = date_dir.name

            entry = {
                "face": None,
                "pupi": None,
                "video": None,
                "recording": None,
                "phase": classify_phase(animal, date_name),
                "path": str(date_dir),
            }

            # ---------- scan files ----------
            for f in date_dir.iterdir():
                fname = f.name.lower()

                if fname.startswith("face") and fname.endswith(".h264"):
                    entry["face"] = str(f)

                elif fname.startswith("pupi") and fname.endswith(".h264"):
                    entry["pupi"] = str(f)

                elif fname.startswith("video") and fname.endswith(".h264"):
                    entry["video"] = str(f)

                elif fname.startswith("recording") and fname.endswith(".mat"):
                    entry["recording"] = str(f)

            data_dict[animal][date_name] = entry

    return data_dict


