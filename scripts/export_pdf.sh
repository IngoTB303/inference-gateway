#!/usr/bin/env bash
# Export submission.ipynb to submission.pdf using nbconvert webpdf.
#
# Requires: uv (project venv), playwright chromium (auto-downloaded on first run)
#
# Usage:
#   bash scripts/export_pdf.sh                  # export submission.ipynb
#   bash scripts/export_pdf.sh my_notebook.ipynb # export a different notebook

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NOTEBOOK="${1:-$REPO_ROOT/submission.ipynb}"
OUTPUT_DIR="$(dirname "$NOTEBOOK")"
NOTEBOOK_NAME="$(basename "$NOTEBOOK" .ipynb)"
PDF_OUT="$OUTPUT_DIR/${NOTEBOOK_NAME}.pdf"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Error: notebook not found: $NOTEBOOK" >&2
  exit 1
fi

echo "Exporting: $NOTEBOOK"
echo "Output:    $PDF_OUT"
echo ""

uv run --group notebook \
  --with jupyter,playwright \
  jupyter nbconvert \
    --to webpdf \
    --allow-chromium-download \
    --output "$PDF_OUT" \
    "$NOTEBOOK"

echo ""
echo "Done: $PDF_OUT ($(du -h "$PDF_OUT" | cut -f1))"
