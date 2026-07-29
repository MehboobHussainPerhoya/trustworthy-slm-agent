"""
extract_text.py
Extracts text from the source PDF(s) in data/raw/ and saves clean .txt
files to data/raw/extracted/ for downstream chunking.

Usage:
    python src/extract_text.py
"""

import pdfplumber
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/raw/extracted")


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF, page by page, joined with page markers."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages_text.append(f"\n\n[PAGE {i}]\n{text}")
    return "".join(pages_text)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}. Copy your source PDF there first.")
        return

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        text = extract_pdf(pdf_path)
        out_path = OUT_DIR / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> saved {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()