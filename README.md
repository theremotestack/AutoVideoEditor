# video-to-chapters

Turn a long video recording into auto-chaptered clips, using a transcript and (optionally) an LLM to write genuinely sensible chapter titles — not just guesses based on pauses.

Built and explained in full on **[The Remote Stack](https://www.youtube.com/@theremotestack)** — Episode 1 of the *Build Log* series walks through this entire build, including the AI-vs-heuristic comparison and the platform-content-rewrite step.

## What it does

1. Extracts audio from your video (ffmpeg)
2. Transcribes it with timestamps (faster-whisper)
3. Detects chapter breaks — either a free pause-based heuristic, or an LLM that actually reads and understands the transcript (Claude, Gemini, Groq, or a local Ollama model)
4. Cuts the video at those points (fast — stream copy, no re-encoding)
5. Writes out named chapter clips, plus `chapters.json`, `chapters.txt`, and the full `transcript.json`/`transcript.txt`

`watch_folder.py` adds a lightweight automation layer on top — drop a new video into a watched folder and the whole pipeline runs automatically.

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) on your PATH
- An NVIDIA GPU is optional but strongly recommended for faster-whisper — CPU works, just slower

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install faster-whisper requests
# Only install what you actually need:
pip install anthropic         # if using --llm-provider anthropic
```

## Usage

```bash
# Free, no API key, no LLM — pause-based chapters
python video_to_chapters.py input.mp4 --chapters heuristic

# Claude
python video_to_chapters.py input.mp4 --chapters llm --api-key $ANTHROPIC_API_KEY

# Free-tier alternatives
python video_to_chapters.py input.mp4 --chapters llm --llm-provider gemini --api-key $GEMINI_API_KEY
python video_to_chapters.py input.mp4 --chapters llm --llm-provider groq --api-key $GROQ_API_KEY

# Fully local and free (requires `ollama serve` running + model pulled)
python video_to_chapters.py input.mp4 --chapters llm --llm-provider ollama --llm-model llama3.1
```

**Automate it** — watch a folder and process new videos automatically:
```bash
python watch_folder.py --folder ./inbox --chapters llm --api-key $ANTHROPIC_API_KEY
```

## Troubleshooting

**`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`**
This comes from `faster-whisper`'s backend (ctranslate2), which doesn't automatically find pip-installed CUDA libraries the way PyTorch does. Fix:
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
export LD_LIBRARY_PATH=$(python3 -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```
Add the `export` line to your shell profile so it persists across terminal sessions.

**`You are sending unauthenticated requests to the HF Hub` warning**
Harmless — just means slower downloads. Set a free token from huggingface.co/settings/tokens as `HF_TOKEN` if you want faster downloads and higher rate limits.

**`anthropic.BadRequestError: anthropic-workspace-id is required...`**
Your Anthropic API key is identity-linked (spans multiple workspaces). Simplest fix: create a new key in the [Console](https://console.anthropic.com) scoped to a single workspace ("Default" rather than "Same as linked account").

## License

MIT — see [LICENSE](LICENSE). Use it, modify it, ship it in your own projects.

## Support this project

This tool is free and always will be. If it saved you time, the best way to support more free tools like it is subscribing to [The Remote Stack](https://www.youtube.com/@theremotestack) on YouTube.
