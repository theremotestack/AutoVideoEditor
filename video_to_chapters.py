#!/usr/bin/env python3
"""
video_to_chapters.py

End-to-end pipeline: take one long video recording and turn it into a set of
chapter clips, automatically, using a transcript.

Pipeline:
    1. Extract audio from the video (ffmpeg)
    2. Transcribe the audio with timestamps (faster-whisper)
    3. Detect chapter breaks from the transcript (LLM or heuristic)
    4. Cut the source video at those timestamps (ffmpeg, stream copy)
    5. Write out named chapter clips + chapters.json + a YouTube-style
       chapters.txt

Usage:
    # Claude (default provider)
    python video_to_chapters.py input.mp4 \
        --whisper-model small \
        --chapters llm \
        --api-key $ANTHROPIC_API_KEY

    # Google Gemini (free tier)
    python video_to_chapters.py input.mp4 --chapters llm \
        --llm-provider gemini --api-key $GEMINI_API_KEY

    # Groq (free tier, fast inference)
    python video_to_chapters.py input.mp4 --chapters llm \
        --llm-provider groq --api-key $GROQ_API_KEY

    # Ollama (local, free, no API key needed - requires `ollama serve` running)
    python video_to_chapters.py input.mp4 --chapters llm \
        --llm-provider ollama --llm-model llama3.1

    # No LLM at all, $0, no API key
    python video_to_chapters.py input.mp4 --chapters heuristic

Requires ffmpeg on PATH. See requirements.txt for Python deps.
Each LLM provider is only imported if you actually select it, so you don't
need every provider's package installed - just the one you're using.

Built for The Remote Stack (youtube.com/@theremotestack) - Build Log Ep.1.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chapter:
    title: str
    start: float
    end: float


# --------------------------------------------------------------------------
# Shell helper
# --------------------------------------------------------------------------

def run(cmd: List[str]) -> None:
    """Run a subprocess command, raising with stderr visible on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stderr.strip()}"
        )


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{proc.stderr.strip()}")
    return float(proc.stdout.strip())


# --------------------------------------------------------------------------
# Step 1: extract audio
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, audio_path: Path) -> None:
    print(f"[1/5] Extracting audio -> {audio_path.name}")
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ])


# --------------------------------------------------------------------------
# Step 2: transcribe with timestamps
# --------------------------------------------------------------------------

def transcribe(audio_path: Path, model_size: str) -> List[Segment]:
    print(f"[2/5] Transcribing with faster-whisper ({model_size}) ...")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from e

    model = WhisperModel(model_size, compute_type="int8")
    segments_iter, _info = model.transcribe(str(audio_path), beam_size=5)
    segments = [Segment(s.start, s.end, s.text.strip()) for s in segments_iter]
    print(f"      -> {len(segments)} segments")
    return segments


def transcript_for_prompt(segments: List[Segment]) -> str:
    """Compact, timestamped transcript suitable for an LLM prompt."""
    lines = [f"[{s.start:.0f}s] {s.text}" for s in segments]
    return "\n".join(lines)


def write_transcript(segments: List[Segment], out_dir: Path) -> None:
    """Persist the raw transcript so it can be reused later (e.g. for
    platform-specific content rewrites), independent of chapter detection."""
    (out_dir / "transcript.json").write_text(
        json.dumps([asdict(s) for s in segments], indent=2)
    )
    lines = [f"[{s.start:.0f}s] {s.text}" for s in segments]
    (out_dir / "transcript.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Step 3a: chapter detection via LLM (multiple providers)
# --------------------------------------------------------------------------

# Sensible default model per provider - override with --llm-model
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-flash-latest",
    "groq": "openai/gpt-oss-120b",
    "ollama": "llama3.1",
}

CHAPTER_PROMPT_TEMPLATE = """Here is a timestamped transcript of a spoken video (timestamps in seconds).
Identify natural chapter breaks based on topic shifts.

Rules:
- Return 4 to 12 chapters depending on how much the topic actually shifts.
- The first chapter must start at 0.
- Titles are under 8 words, in sentence case, no numbering.
- Return ONLY a JSON array, no prose, no markdown fences. Format:
[{{"title": "...", "start_seconds": 0}}, ...]

Transcript:
{transcript}
"""


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed. Run: pip install requests") from e

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, api_key: str, model: str) -> str:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed. Run: pip install requests") from e

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, model: str, host: str) -> str:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed. Run: pip install requests") from e

    url = f"{host.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Is `ollama serve` running, "
            f"and have you pulled the model? (`ollama pull {model}`)"
        ) from e
    return resp.json()["response"]


PROVIDER_LABELS = {
    "anthropic": "Claude",
    "gemini": "Gemini",
    "groq": "Groq",
    "ollama": "a local Ollama model",
}


def detect_chapters_llm(
    segments: List[Segment],
    duration: float,
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
) -> List[Chapter]:
    model = model or DEFAULT_MODELS[provider]
    print(f"[3/5] Detecting chapters with {PROVIDER_LABELS[provider]} ({model}) ...")

    transcript = transcript_for_prompt(segments)
    prompt = CHAPTER_PROMPT_TEMPLATE.format(transcript=transcript)

    if provider == "anthropic":
        raw = _call_anthropic(prompt, api_key, model)
    elif provider == "gemini":
        raw = _call_gemini(prompt, api_key, model)
    elif provider == "groq":
        raw = _call_groq(prompt, api_key, model)
    elif provider == "ollama":
        raw = _call_ollama(prompt, model, ollama_host)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse chapter JSON from model output:\n{raw}") from e

    return _build_chapters(parsed, duration)


# --------------------------------------------------------------------------
# Step 3b: chapter detection via heuristic (no API key required)
# --------------------------------------------------------------------------

def detect_chapters_heuristic(
    segments: List[Segment],
    duration: float,
    pause_threshold: float = 2.0,
    min_chapter_len: float = 45.0,
) -> List[Chapter]:
    """Break chapters on long pauses between segments, merging any
    resulting chapter shorter than min_chapter_len into the previous one."""
    print("[3/5] Detecting chapters with pause-based heuristic ...")
    if not segments:
        return [Chapter("Full video", 0.0, duration)]

    breakpoints = [0.0]
    for prev, nxt in zip(segments, segments[1:]):
        gap = nxt.start - prev.end
        if gap >= pause_threshold:
            breakpoints.append(nxt.start)

    breakpoints = sorted(set(breakpoints))
    starts = breakpoints + [duration]

    chapters: List[Chapter] = []
    for i in range(len(starts) - 1):
        start, end = starts[i], starts[i + 1]
        if chapters and (end - start) < min_chapter_len:
            chapters[-1] = Chapter(chapters[-1].title, chapters[-1].start, end)
            continue
        title = _guess_title(segments, start, end, index=len(chapters) + 1)
        chapters.append(Chapter(title, start, end))

    return chapters


def _guess_title(segments: List[Segment], start: float, end: float, index: int) -> str:
    """Very rough title: first few words spoken in this window."""
    for s in segments:
        if s.start >= start and s.start < end and s.text:
            words = s.text.split()
            snippet = " ".join(words[:6])
            return snippet.rstrip(",.") or f"Chapter {index}"
    return f"Chapter {index}"


def _build_chapters(parsed: list, duration: float) -> List[Chapter]:
    parsed = sorted(parsed, key=lambda c: c["start_seconds"])
    chapters = []
    for i, c in enumerate(parsed):
        start = float(c["start_seconds"])
        end = float(parsed[i + 1]["start_seconds"]) if i + 1 < len(parsed) else duration
        chapters.append(Chapter(c["title"].strip(), start, end))
    return chapters


# --------------------------------------------------------------------------
# Step 4: cut the video
# --------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:60] or "chapter"


def cut_chapters(video_path: Path, chapters: List[Chapter], out_dir: Path) -> None:
    print(f"[4/5] Cutting {len(chapters)} chapter clips ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ch in enumerate(chapters, start=1):
        filename = f"{i:02d}-{slugify(ch.title)}.mp4"
        out_path = out_dir / filename
        # -c copy = stream copy, no re-encode, near-instant. Cuts may land on
        # the nearest keyframe rather than the exact frame; drop -c copy and
        # re-encode if you need frame-accurate cuts.
        run([
            "ffmpeg", "-y",
            "-ss", str(ch.start), "-to", str(ch.end),
            "-i", str(video_path),
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(out_path),
        ])
        print(f"      -> {filename}  ({ch.start:.0f}s - {ch.end:.0f}s)")


# --------------------------------------------------------------------------
# Step 5: write metadata
# --------------------------------------------------------------------------

def fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def write_metadata(chapters: List[Chapter], out_dir: Path) -> None:
    print("[5/5] Writing chapters.json and chapters.txt")
    (out_dir / "chapters.json").write_text(
        json.dumps([asdict(c) for c in chapters], indent=2)
    )
    lines = [f"{fmt_timestamp(c.start)} {c.title}" for c in chapters]
    (out_dir / "chapters.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument("--out", type=Path, default=Path("chapters_output"), help="Output directory")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--chapters", choices=["llm", "heuristic"], default="heuristic", help="Chapter detection method")
    parser.add_argument("--llm-provider", choices=["anthropic", "gemini", "groq", "ollama"], default="anthropic",
                         help="Which LLM to use when --chapters llm (default: anthropic)")
    parser.add_argument("--api-key", default=None,
                         help="API key for the chosen --llm-provider (not needed for ollama)")
    parser.add_argument("--llm-model", default=None,
                         help="Override the default model for the chosen --llm-provider")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                         help="Ollama server URL (only used with --llm-provider ollama)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the extracted audio.wav after the run")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Input video not found: {args.video}")
    if args.chapters == "llm" and args.llm_provider != "ollama" and not args.api_key:
        sys.exit(
            f"--chapters llm --llm-provider {args.llm_provider} requires --api-key "
            f"(or use --llm-provider ollama, which needs no key)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    audio_path = args.out / "audio.wav"

    duration = ffprobe_duration(args.video)
    extract_audio(args.video, audio_path)
    segments = transcribe(audio_path, args.whisper_model)
    write_transcript(segments, args.out)

    if args.chapters == "llm":
        chapters = detect_chapters_llm(
            segments,
            duration,
            provider=args.llm_provider,
            api_key=args.api_key,
            model=args.llm_model,
            ollama_host=args.ollama_host,
        )
    else:
        chapters = detect_chapters_heuristic(segments, duration)

    cut_chapters(args.video, chapters, args.out)
    write_metadata(chapters, args.out)

    if not args.keep_audio:
        audio_path.unlink(missing_ok=True)

    print(f"\nDone. {len(chapters)} chapters written to {args.out}/")


if __name__ == "__main__":
    main()
