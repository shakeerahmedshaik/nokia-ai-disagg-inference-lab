#!/usr/bin/env bash
# Download the model once on the host into ./models/<name>. The directory is bind-mounted read-only
# at /models in both workers, so neither worker needs internet access for weights.
# Usage: bash scripts/fetch-model.sh [hf-repo]   (default HuggingFaceTB/SmolLM2-135M, about 270 MB)
set -euo pipefail

REPO=${1:-HuggingFaceTB/SmolLM2-135M}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=$HERE/models/$(basename "$REPO")
mkdir -p "$DEST"
for f in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.json merges.txt; do
  [ -s "$DEST/$f" ] && { echo "have $f"; continue; }
  echo "fetch $f"
  curl -fsSL --retry 3 -o "$DEST/$f.part" "https://huggingface.co/$REPO/resolve/main/$f" || { rm -f "$DEST/$f.part"; echo "  (optional file missing: $f)"; continue; }
  mv "$DEST/$f.part" "$DEST/$f"
done
[ -s "$DEST/model.safetensors" ] || { echo "FAIL: model.safetensors not downloaded"; exit 1; }
du -sh "$DEST"
