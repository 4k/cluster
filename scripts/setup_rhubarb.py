#!/usr/bin/env python3
"""
Setup script to download and install Rhubarb Lip Sync locally.

Downloads Rhubarb from GitHub releases and installs it in the vendor directory.
"""

import hashlib
import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Rhubarb release information
RHUBARB_VERSION = "1.13.0"
RHUBARB_RELEASES = {
    "Linux": {
        "url": f"https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v{RHUBARB_VERSION}/Rhubarb-Lip-Sync-{RHUBARB_VERSION}-Linux.zip",
        "archive_type": "zip",
        "executable": "rhubarb",
    },
    "Darwin": {  # macOS
        "url": f"https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v{RHUBARB_VERSION}/Rhubarb-Lip-Sync-{RHUBARB_VERSION}-macOS.zip",
        "archive_type": "zip",
        "executable": "rhubarb",
    },
    "Windows": {
        "url": f"https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v{RHUBARB_VERSION}/Rhubarb-Lip-Sync-{RHUBARB_VERSION}-Windows.zip",
        "archive_type": "zip",
        "executable": "rhubarb.exe",
    },
}

# Installation directory (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "rhubarb"


def get_platform_info():
    """Get platform-specific release information."""
    system = platform.system()
    if system not in RHUBARB_RELEASES:
        raise RuntimeError(f"Unsupported platform: {system}")
    return RHUBARB_RELEASES[system]


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"Downloading: {url}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urlretrieve(url, dest, progress_hook)
    print()  # New line after progress


def extract_archive(archive_path: Path, dest_dir: Path, archive_type: str) -> Path:
    """Extract archive and return the extracted directory."""
    print(f"Extracting to: {dest_dir}")

    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Get the top-level directory name
            names = zf.namelist()
            top_dir = names[0].split('/')[0] if '/' in names[0] else None
            zf.extractall(dest_dir)
            return dest_dir / top_dir if top_dir else dest_dir
    elif archive_type == "tar.gz":
        with tarfile.open(archive_path, 'r:gz') as tf:
            names = tf.getnames()
            top_dir = names[0].split('/')[0] if '/' in names[0] else None
            tf.extractall(dest_dir)
            return dest_dir / top_dir if top_dir else dest_dir
    else:
        raise ValueError(f"Unknown archive type: {archive_type}")


def setup_rhubarb(force: bool = False) -> Path:
    """
    Download and setup Rhubarb Lip Sync.

    Args:
        force: If True, re-download even if already installed

    Returns:
        Path to the rhubarb executable
    """
    platform_info = get_platform_info()
    executable_name = platform_info["executable"]
    executable_path = VENDOR_DIR / executable_name

    # Check if already installed
    if executable_path.exists() and not force:
        print(f"Rhubarb already installed at: {executable_path}")
        return executable_path

    # Create vendor directory
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    # Download to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        archive_name = platform_info["url"].split("/")[-1]
        archive_path = tmpdir / archive_name

        # Download
        download_with_progress(platform_info["url"], archive_path)

        # Extract
        extracted_dir = extract_archive(
            archive_path,
            tmpdir,
            platform_info["archive_type"]
        )

        # Find and copy executable and required files
        print(f"Installing to: {VENDOR_DIR}")

        # Copy all files from extracted directory to vendor
        for item in extracted_dir.iterdir():
            dest = VENDOR_DIR / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Make executable
        if executable_path.exists():
            executable_path.chmod(
                executable_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

    # Verify installation
    if not executable_path.exists():
        raise RuntimeError(f"Installation failed: {executable_path} not found")

    print(f"\nRhubarb {RHUBARB_VERSION} installed successfully!")
    print(f"Executable: {executable_path}")

    return executable_path


def get_rhubarb_path() -> Path:
    """Get the path to the Rhubarb executable."""
    platform_info = get_platform_info()
    return VENDOR_DIR / platform_info["executable"]


def is_rhubarb_installed() -> bool:
    """Check if Rhubarb is installed locally."""
    return get_rhubarb_path().exists()


def verify_installation() -> bool:
    """Verify that Rhubarb is working correctly."""
    import subprocess

    rhubarb_path = get_rhubarb_path()
    if not rhubarb_path.exists():
        return False

    try:
        result = subprocess.run(
            [str(rhubarb_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"Rhubarb version: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"Verification failed: {e}")

    return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Rhubarb Lip Sync")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if already installed"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify installation without downloading"
    )
    parser.add_argument(
        "--path",
        action="store_true",
        help="Print the path to rhubarb executable"
    )

    args = parser.parse_args()

    if args.path:
        print(get_rhubarb_path())
        return 0

    if args.verify:
        if verify_installation():
            print("Rhubarb is installed and working correctly")
            return 0
        else:
            print("Rhubarb is not installed or not working")
            return 1

    try:
        setup_rhubarb(force=args.force)

        if verify_installation():
            print("\nInstallation verified successfully!")
            return 0
        else:
            print("\nWarning: Installation completed but verification failed")
            return 1

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
