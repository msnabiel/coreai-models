#!/usr/bin/env bash
set -euo pipefail

PLATFORM="${1:-macOS}"
OUTPUT_DIR="${2:-./exports/qwen3-4b-${PLATFORM}}"

if [[ "$PLATFORM" != "macOS" && "$PLATFORM" != "iOS" ]]; then
  echo "Usage: $0 [macOS|iOS] [output-dir]"
  exit 1
fi

uv run coreai.llm.export qwen3-4b \
  --platform "$PLATFORM" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite
