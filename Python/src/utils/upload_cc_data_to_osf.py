import sys
import time
import shutil
import subprocess
import csv
from pathlib import Path
from datetime import datetime


# ============================================================
# OSF upload settings
# ============================================================

PROJECT_ID = "jrb2p"
REMOTE_ROOT = "data_mat_files"

INCLUDE_PHASES = {
    "habituation",
    "air_training",
    "tone_air_training",
}

UPLOAD_DELAY_SECONDS = 5
MAX_RETRIES = 3

# Set this to True only if you intentionally want to overwrite files on OSF.
FORCE_UPLOAD = False


# ============================================================
# Find repo root and import project utilities
# Expected structure:
# ~/ccaw/Python/src/utils/pdata_io.py
# ============================================================

start_dir = Path.cwd().resolve()
repo_root = start_dir

while not (repo_root / "Python" / "src" / "utils" / "pdata_io.py").exists():
    if repo_root.parent == repo_root:
        raise RuntimeError(
            "Could not find repo root containing Python/src/utils/pdata_io.py"
        )
    repo_root = repo_root.parent

python_root = repo_root / "Python"
sys.path.insert(0, str(python_root))

import src.utils.pdata_io as pdio


# Put the upload log in the repo root, not wherever Python happens to run.
LOG_PATH = repo_root / "osf_upload_log.csv"


# ============================================================
# Helper functions
# ============================================================

def check_osf_command():
    """Confirm that osfclient command-line tool is available."""
    if shutil.which("osf") is None:
        raise RuntimeError(
            "Could not find the 'osf' command. Install with: pip install osfclient"
        )


def load_done_files(log_path):
    """Read previous successful uploads so the script can resume safely."""
    done = set()

    if not log_path.exists():
        return done

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "uploaded":
                done.add(row.get("remote_path"))

    return done


def append_log(row):
    """Append one row to the upload log."""
    file_exists = LOG_PATH.exists()

    fieldnames = [
        "timestamp",
        "animal",
        "date",
        "phase",
        "source_path",
        "remote_path",
        "status",
        "message",
    ]

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def make_log_row(animal, date, phase, source_path, remote_path, status, message):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "animal": animal,
        "date": date,
        "phase": phase,
        "source_path": str(source_path) if source_path else "",
        "remote_path": remote_path,
        "status": status,
        "message": message,
    }


def upload_one_file(source_path, remote_path):
    """Upload one file to OSF with retry and delay."""

    cmd = [
        "osf",
        "-p",
        PROJECT_ID,
        "upload",
    ]

    if FORCE_UPLOAD:
        cmd.append("-f")

    cmd.extend([
        str(source_path),
        remote_path,
    ])

    last_message = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Upload attempt {attempt}/{MAX_RETRIES}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return True, "success"

        last_message = result.stderr.strip() or result.stdout.strip()

        print(f"Attempt {attempt} failed:")
        print(last_message)

        # Sometimes the file is actually already uploaded.
        # Treat this as success only if OSF clearly says it already exists.
        lower_msg = last_message.lower()
        if "already exists" in lower_msg or "file exists" in lower_msg:
            return True, "already exists on OSF; marked as uploaded"

        if attempt < MAX_RETRIES:
            wait_time = UPLOAD_DELAY_SECONDS * attempt
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)

    return False, last_message


# ============================================================
# Main script
# ============================================================

def main():
    check_osf_command()

    print(f"Repo root: {repo_root}")
    print(f"Python root: {python_root}")
    print(f"Upload log: {LOG_PATH}")

    data_root, pdata_root, cc_data = pdio.load_project_context()

    print(f"data_root:  {data_root}")
    print(f"pdata_root: {pdata_root}")
    print(f"Animals:    {list(cc_data.keys())}")

    already_uploaded = load_done_files(LOG_PATH)

    total_considered = 0
    total_attempted = 0
    total_uploaded = 0
    total_skipped = 0
    total_missing = 0
    total_failed = 0

    for animal in sorted(cc_data.keys()):
        sessions = cc_data[animal]

        for date in sorted(sessions.keys()):
            info = sessions[date]

            phase = info.get("phase")

            if phase not in INCLUDE_PHASES:
                continue

            total_considered += 1

            recording = info.get("recording")

            if recording is None:
                total_missing += 1
                remote_path = f"{REMOTE_ROOT}/{animal}/{date}/MISSING_RECORDING"

                append_log(make_log_row(
                    animal=animal,
                    date=date,
                    phase=phase,
                    source_path="",
                    remote_path=remote_path,
                    status="missing",
                    message="recording is None",
                ))

                print(f"Missing recording: {animal} {date} {phase}")
                continue

            source_path = Path(recording)

            remote_path = f"{REMOTE_ROOT}/{animal}/{date}/{source_path.name}"

            if not source_path.exists():
                total_missing += 1

                append_log(make_log_row(
                    animal=animal,
                    date=date,
                    phase=phase,
                    source_path=source_path,
                    remote_path=remote_path,
                    status="missing",
                    message="source file not found",
                ))

                print(f"File not found: {source_path}")
                continue

            if remote_path in already_uploaded:
                total_skipped += 1
                print(f"Skipping already uploaded: {remote_path}")
                continue

            print("\nUploading:")
            print(f"  animal: {animal}")
            print(f"  date:   {date}")
            print(f"  phase:  {phase}")
            print(f"  source: {source_path}")
            print(f"  remote: {remote_path}")

            total_attempted += 1

            success, message = upload_one_file(source_path, remote_path)

            if success:
                total_uploaded += 1
                already_uploaded.add(remote_path)

                append_log(make_log_row(
                    animal=animal,
                    date=date,
                    phase=phase,
                    source_path=source_path,
                    remote_path=remote_path,
                    status="uploaded",
                    message=message,
                ))

                print(f"Uploaded: {remote_path}")
                print(f"Waiting {UPLOAD_DELAY_SECONDS} seconds before next file...")
                time.sleep(UPLOAD_DELAY_SECONDS)

            else:
                total_failed += 1

                append_log(make_log_row(
                    animal=animal,
                    date=date,
                    phase=phase,
                    source_path=source_path,
                    remote_path=remote_path,
                    status="failed",
                    message=message,
                ))

                print(f"FAILED after {MAX_RETRIES} attempts: {remote_path}")
                print(message)

                raise RuntimeError(f"Upload failed at: {remote_path}")

    print("\nDone.")
    print(f"Considered sessions: {total_considered}")
    print(f"Attempted this run: {total_attempted}")
    print(f"Uploaded this run:  {total_uploaded}")
    print(f"Skipped previous:    {total_skipped}")
    print(f"Missing files:       {total_missing}")
    print(f"Failed this run:     {total_failed}")
    print(f"Log file:            {LOG_PATH}")


if __name__ == "__main__":
    main()