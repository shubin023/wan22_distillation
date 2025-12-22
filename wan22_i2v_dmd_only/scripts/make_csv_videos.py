#!/usr/bin/env python3
"""
Create a CSV with a single column 'path' listing all video files in a folder.

Usage:
  python make_video_csv.py /data/videos out.csv
  python make_video_csv.py /data/videos out.csv --recursive
  python make_video_csv.py /data/videos out.csv --extensions .mp4 .mov
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXTS = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"]


def iter_video_paths(root: Path, exts: List[str], recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="Folder containing videos")
    parser.add_argument("out_csv", type=Path, help="Output CSV path")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTS,
        help=f"Video extensions to include (default: {DEFAULT_EXTS})",
    )
    parser.add_argument("--absolute", action="store_true", help="Write absolute paths")
    parser.add_argument("--sort", action="store_true", help="Sort paths lexicographically")
    args = parser.parse_args()

    root = args.video_dir
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions]

    paths = list(iter_video_paths(root, exts=exts, recursive=args.recursive))
    if args.sort:
        paths.sort()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        for p in paths:
            writer.writerow([str(p.resolve() if args.absolute else p)])

    print(f"Wrote {len(paths)} paths to {args.out_csv}")


if __name__ == "__main__":
    main()
