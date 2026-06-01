"""Test-Skript: NuExtract3 strukturierte Extraktion aus einer Meldebescheinigung (PDF).

Verwendung:
    uv run --with pymupdf,openai scripts/test_nuextract3_meldebescheinigung.py [PDF_PFAD]

Standard-PDF: scripts/Meldebescheinigung_Beispiel.pdf
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import fitz  # pymupdf
from openai import OpenAI

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
API_URL = "https://ingo-villnow--vllm-nuextract3-a100-serve.modal.run/v1"
MODEL = "NuExtract3"

SCRIPT_DIR = Path(__file__).parent
DEFAULT_PDF = SCRIPT_DIR / "Meldebescheinigung_Beispiel.pdf"

# ---------------------------------------------------------------------------
# JSON-Template für eine deutsche Meldebescheinigung
# ---------------------------------------------------------------------------
TEMPLATE = {
    "familienname": "verbatim-string",
    "geburtsname": "verbatim-string",
    "vorname": "verbatim-string",
    "geburtsdatum": "date",
    "geburtsort": "verbatim-string",
    "adresse": {
        "strasse_hausnummer": "verbatim-string",
        "postleitzahl": "verbatim-string",
        "wohnort": "verbatim-string",
        "kreis": "verbatim-string",
    },
    "wohnsituation": [
        "allein",
        "mit Ehegatte/Lebenspartner und/oder Kind",
        "bei Erziehungsberechtigtem",
        "unbekannt",
    ],
    "ausstellungsort": "verbatim-string",
    "ausstellungsdatum": "date",
    "unterschrift_behoerde": "verbatim-string",
}

INSTRUCTIONS = (
    "Extrahiere alle Felder exakt so, wie sie im Dokument stehen. "
    "Felder, die nicht im Dokument vorhanden oder nicht lesbar sind, setze auf null. "
    "Das Datum soll im Format YYYY-MM-DD ausgegeben werden."
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def pdf_to_data_urls(pdf_path: Path, dpi: int = 170) -> list[str]:
    data_urls = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{b64}")
    return data_urls


def extract(pdf_path: Path) -> dict:
    t_start = time.perf_counter()

    print(f"PDF: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")

    t0 = time.perf_counter()
    data_urls = pdf_to_data_urls(pdf_path)
    t_render = time.perf_counter() - t0
    print(f"Seiten als Bild gerendert: {len(data_urls)}  ({t_render * 1000:.0f} ms)")

    client = OpenAI(api_key="EMPTY", base_url=API_URL)

    content = [
        {"type": "image_url", "image_url": {"url": url}} for url in data_urls
    ]

    print("Sende Anfrage an NuExtract3 …")
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
        extra_body={
            "chat_template_kwargs": {
                "template": json.dumps(TEMPLATE, ensure_ascii=False, indent=2),
                "instructions": INSTRUCTIONS,
                "enable_thinking": False,
            }
        },
    )
    t_api = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    tokens_per_sec = completion_tokens / t_api if t_api > 0 else 0

    raw = response.choices[0].message.content
    print(f"\nAntwort ({completion_tokens} Tokens):\n")
    print(raw)

    print("\n--- Metriken ---")
    print(f"  PDF rendern:       {t_render * 1000:7.0f} ms")
    print(f"  API-Latenz:        {t_api * 1000:7.0f} ms")
    print(f"  Gesamt:            {t_total * 1000:7.0f} ms")
    print(f"  Prompt-Tokens:     {prompt_tokens:7d}")
    print(f"  Completion-Tokens: {completion_tokens:7d}")
    print(f"  Throughput:        {tokens_per_sec:7.1f} tok/s")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF

    if not pdf_path.exists():
        print(f"Datei nicht gefunden: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    result = extract(pdf_path)

    print("\n--- Strukturiertes Ergebnis ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
