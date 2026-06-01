#!/usr/bin/env bash
# Führt den NuExtract3-Extraktionstest gegen die Modal-API aus.
#
# Verwendung:
#   bash scripts/test_nuextract3_meldebescheinigung.sh [PDF_PFAD]
#
# Standard-PDF: scripts/Meldebescheinigung_Beispiel.pdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PDF="${1:-$SCRIPT_DIR/Meldebescheinigung_Beispiel.pdf}"

if [[ ! -f "$PDF" ]]; then
  echo "Fehler: Datei nicht gefunden: $PDF" >&2
  exit 1
fi

uv run --with "pymupdf,openai" \
  "$SCRIPT_DIR/test_nuextract3_meldebescheinigung.py" "$PDF"
