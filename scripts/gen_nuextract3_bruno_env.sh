#!/usr/bin/env bash
# Generiert die Bruno-Umgebungsdatei "modal-nuextract3.bru" mit dem Base64-kodierten
# Dokument als Umgebungsvariable document_base64.
#
# Verwendung:
#   bash scripts/gen_nuextract3_bruno_env.sh [PDF_PFAD]
#
# Standard-PDF: scripts/Meldebescheinigung_Beispiel.pdf
# Ziel-Env:     tests/bruno/environments/modal-nuextract3.bru

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PDF="${1:-$SCRIPT_DIR/Meldebescheinigung_Beispiel.pdf}"
ENV_FILE="$REPO_ROOT/tests/bruno/environments/modal-nuextract3.bru"
NUEXTRACT3_URL="https://ingo-villnow--vllm-nuextract3-a100-serve.modal.run/v1"

if [[ ! -f "$PDF" ]]; then
  echo "Fehler: Datei nicht gefunden: $PDF" >&2
  exit 1
fi

echo "PDF:       $PDF"
echo "Zieldatei: $ENV_FILE"
echo "Konvertiere PDF → PNG → Base64 …"

DOCUMENT_BASE64=$(uv run --with pymupdf python -c "
import base64, fitz, sys
with fitz.open('$PDF') as doc:
    # Erste Seite (für mehrseitige PDFs hier iterieren)
    pix = doc[0].get_pixmap(dpi=170, alpha=False)
    print(base64.b64encode(pix.tobytes('png')).decode(), end='')
")

cat > "$ENV_FILE" <<EOF
vars {
  nuextract3_url: ${NUEXTRACT3_URL}
  document_base64: ${DOCUMENT_BASE64}
}
EOF

echo "Fertig. Umgebung 'modal-nuextract3' in Bruno auswählen und Anfragen senden."
