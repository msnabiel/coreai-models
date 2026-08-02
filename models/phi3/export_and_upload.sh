#!/bin/sh
set -eu

# Export and upload both validated 4096-token Phi-3.5 bundles.
# The temporary directory is removed only after both HF uploads succeed.

REPO_ID=${HF_REPO_ID:-Nabiel/FlowKit-Lyra}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
HF_CLI=${HF_CLI:-.venv/bin/hf}
MODEL_ID=phi-3.5-mini-instruct
# Phi-3.5 is stored in Xet-backed shards. Standard HTTP downloads are more
# reliable for this script and still use the normal Hugging Face cache.
export HF_HUB_DISABLE_XET=1
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/flowkit-phi3.XXXXXX")

cleanup_on_exit() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "Export/upload failed; preserving artifacts at: $WORK_DIR" >&2
    fi
    exit "$status"
}
trap cleanup_on_exit EXIT HUP INT TERM

echo "Exporting iOS Phi-3.5 bundle..."
"$PYTHON_BIN" -m coreai_models.llm.export "$MODEL_ID" \
    --platform iOS \
    --output-dir "$WORK_DIR" \
    --overwrite

echo "Exporting macOS Phi-3.5 bundle..."
"$PYTHON_BIN" -m coreai_models.llm.export "$MODEL_ID" \
    --platform macOS \
    --output-dir "$WORK_DIR" \
    --overwrite

IOS_BUNDLE="$WORK_DIR/phi_3_5_mini_instruct_static"
MACOS_BUNDLE="$WORK_DIR/phi_3_5_mini_instruct_4bit_dynamic"

test -d "$IOS_BUNDLE"
test -d "$MACOS_BUNDLE"
test -f "$IOS_BUNDLE/metadata.json"
test -f "$MACOS_BUNDLE/metadata.json"

echo "Uploading iOS bundle to $REPO_ID..."
"$HF_CLI" upload "$REPO_ID" "$IOS_BUNDLE" \
    phi_3_5_mini_instruct_static \
    --commit-message "Add Phi-3.5 iOS 4096-token model"

echo "Uploading macOS bundle to $REPO_ID..."
"$HF_CLI" upload "$REPO_ID" "$MACOS_BUNDLE" \
    phi_3_5_mini_instruct_4bit_dynamic \
    --commit-message "Add Phi-3.5 macOS 4096-token model"

echo "Both uploads succeeded; deleting local export artifacts: $WORK_DIR"
rm -rf "$WORK_DIR"
trap - EXIT HUP INT TERM
