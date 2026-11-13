#!/usr/bin/env python3
"""Download and extract the official MIND-large train, dev, and test archives."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


BASE_URL = "https://mind201910small.blob.core.windows.net/release"
DEFAULT_HF_REPO = "yjw1029/MIND"
ARCHIVES = {
    "MINDlarge_train": "MINDlarge_train.zip",
    "MINDlarge_dev": "MINDlarge_dev.zip",
    "MINDlarge_test": "MINDlarge_test.zip",
}
EXPECTED_FILES = {
    "news.tsv",
    "behaviors.tsv",
    "entity_embedding.vec",
    "relation_embedding.vec",
}


def download(url: str, destination: Path, token: str | None = None) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "MIND-large-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                received += len(block)
                if total:
                    print(
                        f"\r  {destination.name}: {received / total:.1%} "
                        f"({received / (1024**2):,.1f}/{total / (1024**2):,.1f} MiB)",
                        end="",
                        flush=True,
                    )
        if total:
            print()
        temporary.replace(destination)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        detail = exc.read(500).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Server trả HTTP {exc.code} khi tải {url}. {detail}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def inspect_archive(archive: Path) -> dict[str, zipfile.ZipInfo]:
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"File tải về không phải ZIP hợp lệ: {archive}")
    with zipfile.ZipFile(archive) as zipped:
        bad_member = zipped.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP bị lỗi CRC tại: {bad_member}")
        members: dict[str, zipfile.ZipInfo] = {}
        for info in zipped.infolist():
            if info.is_dir():
                continue
            basename = PurePosixPath(info.filename).name
            if basename in EXPECTED_FILES:
                if basename in members:
                    raise RuntimeError(f"ZIP có nhiều file cùng tên {basename}")
                members[basename] = info
        missing = {"news.tsv", "behaviors.tsv"} - members.keys()
        if missing:
            raise RuntimeError(f"{archive.name} thiếu file cần thiết: {', '.join(sorted(missing))}")
        return members


def extract_archive(archive: Path, destination: Path) -> None:
    members = inspect_archive(archive)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for basename, info in members.items():
            output = destination / basename
            with zipped.open(info) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
    print(f"  Đã giải nén {len(members)} file vào {destination}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", type=Path, default=project_root / "downloads")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--source", choices=("official", "huggingface"), default="official")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--splits", nargs="+", choices=tuple(ARCHIVES), default=list(ARCHIVES),
        help="Các split cần tải (mặc định gồm train, dev, test)",
    )
    args = parser.parse_args()
    downloads_dir = args.downloads_dir.resolve()
    data_dir = args.data_dir.resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if args.source == "huggingface" and not token:
        print("Nguồn Hugging Face yêu cầu HF_TOKEN đã được cấp quyền truy cập dataset.", file=sys.stderr)
        return 1

    try:
        archives: list[tuple[str, Path]] = []
        for split in args.splits:
            filename = ARCHIVES[split]
            archive = downloads_dir / filename
            if archive.exists():
                try:
                    inspect_archive(archive)
                    print(f"Dùng ZIP hợp lệ đã có: {archive}")
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    archive.unlink()
            if not archive.exists():
                if args.source == "huggingface":
                    url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{filename}"
                else:
                    url = f"{args.base_url.rstrip('/')}/{filename}"
                print(f"Đang tải {url}")
                download(url, archive, token if args.source == "huggingface" else None)
                inspect_archive(archive)
            archives.append((split, archive))
        for split, archive in archives:
            extract_archive(archive, data_dir / split)
    except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"\nTải MIND-large thất bại: {exc}", file=sys.stderr)
        if "409" in str(exc) or "Public access is not permitted" in str(exc):
            print("Endpoint Azure từ chối truy cập công khai. Có thể thử mirror Hugging Face nếu tài khoản đã được cấp quyền.", file=sys.stderr)
        elif "401" in str(exc) or "403" in str(exc):
            print("Hugging Face chưa cấp quyền cho token này; cần được chấp thuận điều kiện dataset.", file=sys.stderr)
        return 1

    print("\nHoàn tất. Dữ liệu được giải nén tại:")
    for split in args.splits:
        folder = data_dir / split
        total = sum(path.stat().st_size for path in folder.iterdir() if path.is_file())
        print(f"  {folder}: {total / (1024**3):,.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
