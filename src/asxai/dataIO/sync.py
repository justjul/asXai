import config
import subprocess

from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)


def push_to_drive(datatype: str | list, confirm: bool = True):
    """Push datasets to synced cloud storage with rsync."""
    if config.DRIVE_DATA_PATH:
        logger.error("No cloud storage mounted")
        pass

    valid_dirs = {
        "textdata": (config.TEXTDATA_PATH, config.DRIVE_TEXTDATA_PATH),
        "metadata": (config.METADATA_PATH, config.DRIVE_METADATA_PATH)}

    datatype = [datatype] if isinstance(datatype, str) else datatype
    invalid = [d for d in datatype if d not in valid_dirs.keys()]
    if invalid:
        raise ValueError(f"Invalid datatype(s) {invalid}. \
                         Choose from: {list(valid_dirs.keys())}")

    # Confirm sync operation
    if confirm:
        msg = "\n".join(
            [f"Sync {valid_dirs[d][0]} → {valid_dirs[d][1]}?" for d in datatype])
        proceed = input(f"{msg}\nProceed? (Y/n): ").strip().lower()
        if proceed not in ["y", "yes", ""]:
            print("Sync aborted.")
            return

    # Execute rsync for each dataset type
    for d in datatype:
        source, dest = valid_dirs[d]
        subprocess.run([
            "rsync",
            "-rv",
            "--size-only",  # synch recursively and don't check timestamps
            # ensures contents are copied, not the folder itself
            str(source) + "/",
            str(dest)])
        print(f"Synced {source} -> {dest} (rsync complete)")
