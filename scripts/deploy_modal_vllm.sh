#!/usr/bin/env bash
# Deploy a Modal vLLM app for Gemma 4 E2B on A10G.
#
# Usage:
#   bash scripts/deploy_modal_vllm.sh [standard|optimized|hardcore]
#
# Profiles:
#   standard   — baseline vLLM flags, max_model_len=8192 (default)
#   optimized  — chunked prefill + prefix caching, tuned for Gemma 4 on A10G
#   hardcore   — optimized + fp8 KV cache, larger batch tokens, more seqs
#
# After deploy, copy the printed URL into config.yaml under the matching backend.

set -euo pipefail

PROFILE=${1:-standard}

case "$PROFILE" in
  standard|optimized|hardcore)
    ;;
  *)
    echo "Unknown profile: $PROFILE. Use 'standard', 'optimized', or 'hardcore'." >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

case "$PROFILE" in
  standard)  MODAL_FILE="$REPO_ROOT/modal/vllm_gemma4.py" ;;
  optimized) MODAL_FILE="$REPO_ROOT/modal/vllm_gemma4_optimized.py" ;;
  hardcore)  MODAL_FILE="$REPO_ROOT/modal/vllm_gemma4_hardcore.py" ;;
esac

echo "Deploying $MODAL_FILE ..."
# modal is in the 'deploy' dependency group — use 'uv run --group deploy'
uv run --group deploy modal deploy "$MODAL_FILE"

echo ""
echo "Done. Paste the URL above into config.yaml:"
echo "  - name: modal-gemma4-${PROFILE}"
echo "    type: http"
echo "    url: <URL from above>"
echo "    timeout: 120"
echo "    model: gemma-4-e2b-it"
echo ""
echo "Then verify: curl http://localhost:8080/v1/backends"
