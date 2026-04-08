#!/usr/bin/env bash
# Deploy a Modal vLLM app for Gemma 4 E2B on A10G.
#
# Usage:
#   bash scripts/deploy_modal_vllm.sh [standard|optimized]
#
# Profiles:
#   standard   — baseline vLLM flags, max_model_len=8192 (default)
#   optimized  — chunked prefill + prefix caching, tuned for Gemma 4 on A10G
#
# After deploy, copy the printed URL into config.yaml under the matching backend.

set -euo pipefail

PROFILE=${1:-standard}

case "$PROFILE" in
  standard|optimized)
    ;;
  *)
    echo "Unknown profile: $PROFILE. Use 'standard' or 'optimized'." >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$PROFILE" = "standard" ]; then
  MODAL_FILE="$REPO_ROOT/modal/vllm_gemma4.py"
else
  MODAL_FILE="$REPO_ROOT/modal/vllm_gemma4_optimized.py"
fi

echo "Deploying $MODAL_FILE ..."
modal deploy "$MODAL_FILE"

echo ""
echo "Done. Paste the URL above into config.yaml:"
echo "  - name: modal-gemma4-${PROFILE}"
echo "    type: http"
echo "    url: <URL from above>"
echo "    timeout: 120"
echo "    model: gemma-4-e2b-it"
echo ""
echo "Then verify: curl http://localhost:8080/v1/backends"
