#!/usr/bin/env bash
set -Eeuo pipefail

# Keep the progress log focused on actionable output. These messages are known
# non-fatal warnings for this repository's pinned Torch 2.9.0 environment.
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS sed doesn't support -u; use -l (line buffered) instead
  exec > >(sed -l \
    -e '/Skipping import of cpp extensions due to incompatible torch version/d' \
    -e '/Torch version 2\.9\.0 has not been tested with coremltools/d' \
    -e '/Redirects are currently not supported in Windows or MacOs/d') 2>&1
else
  exec > >(sed -u \
    -e '/Skipping import of cpp extensions due to incompatible torch version/d' \
    -e '/Torch version 2\.9\.0 has not been tested with coremltools/d' \
    -e '/Redirects are currently not supported in Windows or MacOs/d') 2>&1
fi

# Export, upload, and remove one or more fixed-context iOS LLM assets.
# The asset is uploaded under <HF_REPO>/<model-name>-<context>/.
#
# Example:
#   ./export_upload_cleanup.sh \
#     --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
#     --repo Nabiel/FlowKit-Lyra \
#     --contexts 4096 8192 16384 24576 32768

HF_REPO="Nabiel/FlowKit-Lyra"
OUTPUT_DIR="exports"
PLATFORM="iOS"
CONTEXTS=(4096 8192 16384 24576 32768)
PYTHON=".venv/bin/python"
export PYTHONPATH="$(pwd)/python/src${PYTHONPATH:+:$PYTHONPATH}"
MODELS=(
  "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
)

usage() {
  sed -n '1,18p' "$0"
  echo ""
  echo "Options: --model ID --repo OWNER/REPO --platform macOS|iOS --contexts N [N ...] --output-dir DIR --python PATH"
}

while (($#)); do
  case "$1" in
    --model) MODELS=("$2"); shift 2 ;;
    --repo) HF_REPO="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --contexts) CONTEXTS=(); shift; while (($#)) && [[ "$1" != --* ]]; do CONTEXTS+=("$1"); shift; done ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ ${#CONTEXTS[@]} -gt 0 ]] || { echo "At least one context is required" >&2; exit 2; }
[[ "$PLATFORM" == "macOS" || "$PLATFORM" == "iOS" ]] || { echo "Invalid platform '$PLATFORM'; must be macOS or iOS" >&2; exit 2; }
command -v hf >/dev/null || { echo "hf CLI is required (install huggingface-hub)" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 1; }

for context in "${CONTEXTS[@]}"; do
  [[ "$context" =~ ^[0-9]+$ && "$context" -gt 0 && "$context" -le 32768 ]] || {
    echo "Invalid context '$context'; Qwen2.5-Coder native maximum is 32768" >&2; exit 2;
  }
done

for model_id in "${MODELS[@]}"; do
  model_slug="${model_id##*/}"
  model_slug="$(printf '%s' "$model_slug" | tr '[:upper:]' '[:lower:]')"
  platform_lower="$(printf '%s' "$PLATFORM" | tr '[:upper:]' '[:lower:]')"
  for context in "${CONTEXTS[@]}"; do
    name="${model_slug}-${context}-${platform_lower}"
    # The LLM exporter creates a directory named exactly like --output-name.
    bundle="$OUTPUT_DIR/$name"
    if [[ -d "$bundle" ]]; then
      echo "==> Reusing existing export $bundle"
    else
      echo "==> Exporting $name"
      # macOS uses 4bit quantization, iOS uses 4bit palettization
      if [[ "$PLATFORM" == "macOS" ]]; then
        compression="4bit"
      else
        compression="4bit_weight_palettized_group8"
      fi
      "$PYTHON" -m coreai_models.llm.export "$model_id" \
        --platform "$PLATFORM" --max-context-length "$context" \
        --compute-precision float16 --compression "$compression" \
        --experimental \
        --output-dir "$OUTPUT_DIR" --output-name "$name"
    fi

    [[ -d "$bundle" ]] || { echo "Expected export missing: $bundle" >&2; exit 1; }
    echo "==> Uploading $bundle to $HF_REPO/$name"
    "$PYTHON" -c 'from huggingface_hub import HfApi; import sys
api = HfApi()
api.upload_folder(folder_path=sys.argv[1], path_in_repo=sys.argv[2], repo_id=sys.argv[3], repo_type="model", commit_message=f"Add Core AI export {sys.argv[2]}")' \
      "$bundle" "$name" "$HF_REPO"

    echo "==> Removing local export $bundle"
    rm -rf -- "$bundle"
  done
done

echo "Completed all requested model and context exports in $HF_REPO"
