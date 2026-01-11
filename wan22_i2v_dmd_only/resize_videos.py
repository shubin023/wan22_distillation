import argparse
import subprocess
from pathlib import Path


def resize_video_ffmpeg(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    overwrite: bool = False,
    codec: str = "mpeg4",
    bitrate: str | None = None,
    qscale: int = 2,
    max_frames: int | None = None,
):
    """
    Resize a single video using ffmpeg with bicubic scaling.
    Defaults to mpeg4 + qscale for maximum compatibility on older FFmpeg builds.
    """
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        f"scale={width}:{height}",
        "-c:v",
        codec,
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    if bitrate:
        cmd.extend(["-b:v", bitrate])
    else:
        cmd.extend(["-qscale:v", str(qscale)])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
    )

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Resize all videos in a folder to a fixed resolution.")
    parser.add_argument("--input_dir", required=True, help="Folder with input videos.")
    parser.add_argument("--output_dir", required=True, help="Folder to save resized videos.")
    parser.add_argument("--width", type=int, required=True, help="Target width.")
    parser.add_argument("--height", type=int, required=True, help="Target height.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--codec", default="mpeg4", help="Video codec to use (default: mpeg4).")
    parser.add_argument(
        "--bitrate", default=None, help="Target video bitrate (e.g., 5M). If not set, uses qscale."
    )
    parser.add_argument("--qscale", type=int, default=2, help="Constant quality for mpeg4 (lower is higher quality).")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames (e.g., 84).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]

    print(f"Found {len(videos)} videos to process.")
    for vid in videos:
        rel = vid.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            resize_video_ffmpeg(
                vid,
                out_path,
                args.width,
                args.height,
                overwrite=args.overwrite,
                codec=args.codec,
                bitrate=args.bitrate,
                qscale=args.qscale,
                max_frames=args.max_frames,
            )
            print(f"Resized {vid} -> {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to resize {vid}: {e}")


if __name__ == "__main__":
    main()
