from pathlib import Path
import shutil


def _expected_mp4_stems(session_info: dict):
    """
    From cc_data session entry, compute expected MP4 stems
    corresponding to original H264 files.
    """
    stems = set()

    for key in ("face", "pupi", "video"):
        h264_path = session_info.get(key)
        if h264_path:
            stems.add(Path(h264_path).stem)

    return stems


def organize_mp4s_from_ccdata(cc_data: dict, proc_base: str, dry_run: bool = False):
    """
    Move MP4 files into mp4/ subfolder inside PROC tree.

    Only moves MP4s that correspond to original H264 files.

    Parameters
    ----------
    cc_data : dict
        Your classical conditioning dictionary

    proc_base : str
        Root of processed data tree

    dry_run : bool
        If True, only print actions (SAFE preview)
    """

    proc_base = Path(proc_base).expanduser()

    print("\n=== MP4 ORGANIZATION START ===\n")

    for animal, sessions in cc_data.items():

        for date, info in sessions.items():

            session_proc_dir = proc_base / animal / date

            if not session_proc_dir.exists():
                continue

            expected_stems = _expected_mp4_stems(info)
            if not expected_stems:
                continue

            mp4_dir = session_proc_dir / "mp4"

            # Find candidate MP4s ONLY in session root
            mp4_files = list(session_proc_dir.glob("*.mp4"))

            if not mp4_files:
                continue

            moved_any = False

            for mp4_path in mp4_files:
                stem = mp4_path.stem

                # --- KEY FILTER ---
                if stem not in expected_stems:
                    # skip DLC, labeled, etc.
                    continue

                target_path = mp4_dir / mp4_path.name

                if target_path.exists():
                    continue

                if not moved_any:
                    print(f"[SESSION] {animal} {date}")
                    moved_any = True

                print(f"  → move {mp4_path.name}")

                if not dry_run:
                    mp4_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(mp4_path), str(target_path))

    print("\n=== MP4 ORGANIZATION DONE ===\n")