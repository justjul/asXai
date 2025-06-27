"""
asXai Drive Sync Utility
------------------------

Provides a helper function to synchronize local dataset directories (metadata, textdata)
to cloud-mounted storage using rsync. Ensures only changed files (by size) are copied.

Usage:
    from asxai.dataIO import push_to_drive
    push_to_drive("metadata", confirm=True)
    push_to_drive(["metadata", "textdata"], confirm=False)
"""

import config
import subprocess

from asxai.logger import get_logger

# Initialize logger with module-specific log level
logger = get_logger(__name__, level=config.LOG_LEVEL)


def push_to_drive(datatype: str | list, confirm: bool = True):
    """
    Synchronize one or more dataset directories to cloud storage via rsync.

    Args:
        datatype: Either a single string or list of strings indicating
                  which datasets to sync. Valid values are:
                  - "metadata": syncs config.METADATA_PATH → config.DRIVE_METADATA_PATH
                  - "textdata": syncs config.TEXTDATA_PATH → config.DRIVE_TEXTDATA_PATH
        confirm:  If True, prompt the user to confirm before syncing each directory.

    Raises:
        ValueError: If any entry in `datatype` is not one of the valid dataset keys.
    """
    # Ensure cloud drive base path is configured
    if config.DRIVE_DATA_PATH:
        logger.error("No cloud storage mounted")
        return

    # Map datatype keys to (source_path, destination_path) tuples
    valid_dirs = {
        "textdata": (config.TEXTDATA_PATH, config.DRIVE_TEXTDATA_PATH),
        "metadata": (config.METADATA_PATH, config.DRIVE_METADATA_PATH)}

    # Normalize to list
    datatype = [datatype] if isinstance(datatype, str) else datatype
    # Validate inputs
    invalid = [d for d in datatype if d not in valid_dirs.keys()]
    if invalid:
        raise ValueError(f"Invalid datatype(s) {invalid}. \
                         Choose from: {list(valid_dirs.keys())}")

    # Prompt the user for confirmation if requested
    if confirm:
        msg = "\n".join(
            [f"Sync {valid_dirs[d][0]} → {valid_dirs[d][1]}?" for d in datatype])
        proceed = input(f"{msg}\nProceed? (Y/n): ").strip().lower()
        if proceed not in ["y", "yes", ""]:
            print("Sync aborted.")
            return

    # Perform rsync for each requested dataset
    for d in datatype:
        source, dest = valid_dirs[d]
        # -r : recursive, -v : verbose, --size-only : compare by size, not timestamp
        # Trailing slash on source ensures contents are copied, not the directory itself
        subprocess.run([
            "rsync",
            "-rv",
            "--size-only",
            str(source) + "/",
            str(dest)])
        print(f"Synced {source} -> {dest} (rsync complete)")
