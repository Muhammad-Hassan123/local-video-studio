#!/usr/bin/env bash
# Setup for a Linux machine: Python packages and the voice model.
# Enough to validate screenplays (direct) and voice them (tts); the art stages need ComfyUI on a GPU PC.
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
mkdir -p models
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -f "models/$f" ] || curl -L -o "models/$f" "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f"
done
