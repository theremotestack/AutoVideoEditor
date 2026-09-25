#!/usr/bin/env python3
"""
watch_folder.py

Light automation glue for video_to_chapters.py.

Watches a folder. The moment a new video file is dropped in (and finishes
copying), it automatically runs video_to_chapters.py on it - no manual
command needed.

This is deliberately dependency-free (stdlib only) so there's nothing extra
to install for a demo. It polls rather than using OS-level file-system
events - simpler to reason about on camera, at the cost of a small delay
(POLL_INTERVAL) before a new file is noticed.

Built for The Remote Stack (youtube.com/@theremotestack) - Build Log Ep.1.

Usage:
    python watch_folder.py --folder ./inbox --out-root ./chapters_output \
        --chapters llm --llm-provider anthropic --api-key $ANTHROPIC_API_KEY

    # Ollama, local, no key
    python watch_folder.py --folder ./inbox --chapters llm \
        --llm-provider ollama --llm-model llama3.1
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
POLL_INTERVAL = 5          # seconds between folder checks
STABLE_CHECKS = 3          # how many consecutive stable size checks before treating a file as "done copying"


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def wait_until_stable(path: Path) -> None:
    """Block until a file's size stops changing, so we don't try to process
    a video that's still being copied/written into the folder."""
    last_size = -1
    stable_count = 0
    while stable_count < STABLE_CHECKS:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            # File vanished mid-copy (e.g. a temp file was renamed) - bail out
            return
        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size
        time.sleep(POLL_INTERVAL)


def process_video(video_path: Path, out_root: Path, pipeline_args: list[str]) -> None:
    out_dir = out_root / video_path.stem
    print(f"\n[watch] New video detected: {video_path.name}")
    print(f"[watch] Waiting for file to finish copying ...")
    wait_until_stable(video_path)
    if not video_path.exists():
        print(f"[watch] {video_path.name} disappeared before processing - skipping")
        return

    print(f"[watch] Running pipeline -> {out_dir}")
    cmd = [
        sys.executable, "video_to_chapters.py", str(video_path),
        "--out", str(out_dir),
    ] + pipeline_args

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[watch] Done: {video_path.name} -> {out_dir}")
    else:
        print(f"[watch] Pipeline failed for {video_path.name} (exit code {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True, help="Folder to watch for new video files")
    parser.add_argument("--out-root", type=Path, default=Path("chapters_output"),
                         help="Base output directory - each video gets its own subfolder here")
    parser.add_argument("--chapters", choices=["llm", "heuristic"], default="heuristic")
    parser.add_argument("--llm-provider", choices=["anthropic", "gemini", "groq", "ollama"], default="anthropic")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--whisper-model", default="small")
    args = parser.parse_args()

    if not args.folder.exists():
        sys.exit(f"Watch folder not found: {args.folder}")

    # Rebuild the flag list to hand straight through to video_to_chapters.py
    pipeline_args = [
        "--chapters", args.chapters,
        "--llm-provider", args.llm_provider,
        "--whisper-model", args.whisper_model,
        "--ollama-host", args.ollama_host,
    ]
    if args.api_key:
        pipeline_args += ["--api-key", args.api_key]
    if args.llm_model:
        pipeline_args += ["--llm-model", args.llm_model]

    seen = {p.name for p in args.folder.iterdir() if is_video(p)}
    print(f"[watch] Watching {args.folder} for new video files ... (Ctrl+C to stop)")
    print(f"[watch] Ignoring {len(seen)} file(s) already present at startup")

    try:
        while True:
            current = {p for p in args.folder.iterdir() if is_video(p)}
            new_files = [p for p in current if p.name not in seen]
            for video_path in new_files:
                seen.add(video_path.name)
                process_video(video_path, args.out_root, pipeline_args)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[watch] Stopped.")


if __name__ == "__main__":
    main()
