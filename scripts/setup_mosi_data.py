import argparse
import pickle
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AADfY2s7gD_MkR76m03KS0K1a/Archive.zip?dl=1"
TARGET_FILE = "mosi_data_noalign.pkl"
LOCAL_ARCHIVE_CANDIDATES = ("Archive.zip", "MOSI_MOSEI_IEMOCAP.zip")


def download_archive(url, destination, retries=3):
    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                resolved_url = response.geturl()
            if zipfile.is_zipfile(destination):
                return resolved_url
            last_error = zipfile.BadZipFile(f"Downloaded file is not a valid zip on attempt {attempt}.")
        except Exception as exc:
            last_error = exc
        if destination.exists():
            destination.unlink()
    raise last_error


def extract_target(archive_path, target_name, data_dir):
    with zipfile.ZipFile(archive_path, "r") as archive:
        matching = [name for name in archive.namelist() if name.endswith(target_name)]
        if not matching:
            raise FileNotFoundError(f"Could not find {target_name} in archive.")
        member_name = matching[0]
        with archive.open(member_name) as source, open(data_dir / target_name, "wb") as target:
            shutil.copyfileobj(source, target)
        return member_name


def summarize_pickle(data_file):
    with open(data_file, "rb") as handle:
        dataset = pickle.load(handle)
    for split in ("train", "valid", "test"):
        print(
            f"{split}: "
            f"text={dataset[split]['text'].shape}, "
            f"audio={dataset[split]['audio'].shape}, "
            f"vision={dataset[split]['vision'].shape}, "
            f"labels={dataset[split]['labels'].shape}"
        )


def find_local_archive(target_name):
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in LOCAL_ARCHIVE_CANDIDATES:
        path = repo_root / candidate
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path, "r") as archive:
                if any(name.endswith(target_name) for name in archive.namelist()):
                    return path
        except zipfile.BadZipFile:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Download and verify processed CMU-MOSI data")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Archive download URL")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Output data directory",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Use an existing local archive instead of downloading",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.data_dir / TARGET_FILE
    if target_path.exists() and not args.force:
        print(f"Using existing {target_path}")
        summarize_pickle(target_path)
        return

    archive_path = args.archive_path or find_local_archive(TARGET_FILE)
    if archive_path is not None:
        archive_path = archive_path.resolve()
        print(f"Using local archive {archive_path}")
        member_name = extract_target(archive_path, TARGET_FILE, args.data_dir)
        print(f"Extracted {member_name} -> {target_path}")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "Archive.zip"
            print(f"Downloading archive to {archive_path}")
            resolved_url = download_archive(args.url, archive_path)
            print(f"Resolved archive URL: {resolved_url}")
            member_name = extract_target(archive_path, TARGET_FILE, args.data_dir)
            print(f"Extracted {member_name} -> {target_path}")

    summarize_pickle(target_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"setup_mosi_data.py failed: {exc}", file=sys.stderr)
        raise
