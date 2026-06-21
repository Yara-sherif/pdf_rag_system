"""
Stage 1 — Ingest: extract text from PDF with page-level metadata.
Run once: python preprocessing.py
Output: pages.json
"""
import json
from pathlib import Path
import fitz  # pymupdf

_ROOT = Path(__file__).parent.parent
PDF_PATH = _ROOT / "data" / "52_introduction-nlp.pdf"
OUTPUT_PATH = _ROOT / "data" / "pages.json"
MIN_CHARS = 50


def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if len(text) < MIN_CHARS:
            print(f"  Skipping page {page_num + 1} — only {len(text)} chars (blank/TOC)")
            continue
        pages.append({
            "text": text,
            "page_number": page_num + 1,
            "source": Path(pdf_path).name,
            "char_count": len(text),
        })
    doc.close()
    return pages


if __name__ == "__main__":
    print(f"Extracting text from: {PDF_PATH}")
    pages = extract_pages(PDF_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"\nExtracted {len(pages)} usable pages → {OUTPUT_PATH}")
    for p in pages:
        print(f"  Page {p['page_number']:>3}: {p['char_count']:>5} chars")