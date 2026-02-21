from pathlib import Path
import re
import shutil
import json
import os


# --------------------------------------------------
# Load config
# --------------------------------------------------
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    for k, v in cfg["paths"].items():
        cfg["paths"][k] = os.path.expanduser(v)

    return cfg


# --------------------------------------------------
# Decide if MP4 is a TRUE camera conversion
# --------------------------------------------------
def is_raw_camera_mp4(fname: str) -> bool:
    """
    Keep only original camera MP4 conversions.
    Reject DLC/labeled/etc.
    """

    name = fname.lower()

    # reject known derived videos
    reject_patterns = [
        "dlc",
        "labeled",
        "resnet",
    ]
    if any(p in name for p in reject_patterns):
        return False

    # accept only known camera prefixes
    accept_patterns = [
        r"^face_.*\.mp4$",
        r"^pupi_.*\.mp4$",
        r"^video_.*\.mp4$",
    ]

    return any(re.match(p, name) for p in accept_patterns)


# --------------------------------------------------
# Main organizer
# --------------------------------------------------
def organize_mp4s(proc_base: str, dry_run: bool = True):
    """
    Walk proc_base and move valid MP4s into mp4/ folders.

    dry_run=True → preview only (SAFE)
    """

    proc_base = Path(proc_base)

    moved = 0
    skipped = 0

    print(f"Scanning: {proc_base}")
    print(f"Dry run: {dry_run}")
    print("-" * 50)

    # iterate animals
    for animal_dir in sorted(proc_base.glob("NML_*")):
        if not animal_dir.is_dir():
            continue

        # iterate dates
        for date_dir in sorted(animal_dir.glob("20*_**")):
            if not date_dir.is_dir():
                continue

            mp4_dir = date_dir / "mp4"

            # find loose mp4 files (ONLY in session root)
            for mp4_file in date_dir.glob("*.mp4"):

                if not is_raw_camera_mp4(mp4_file.name):
                    skipped += 1
                    continue

                target = mp4_dir / mp4_file.name

                print(f"MOVE: {mp4_file} → {target}")

                if not dry_run:
                    mp4_dir.mkdir(exist_ok=True)
                    shutil.move(str(mp4_file), str(target))

                moved += 1

    print("-" * 50)
    print(f"Moved:   {moved}")
    print(f"Skipped: {skipped}")