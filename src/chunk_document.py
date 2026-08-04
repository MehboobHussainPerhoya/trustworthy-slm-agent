"""
chunk_document.py
Splits extracted paper text into semantic chunks (by section) and saves
them as a structured JSONL retrieval corpus.

Usage:
    python src/chunk_document.py
"""

import json
import re
from pathlib import Path

EXTRACTED_DIR = Path("data/raw/extracted")
OUT_PATH = Path("data/chunks.jsonl")

# Section headers as they appear in this specific paper.
# Adjust this list if you use a different source document.
SECTION_PATTERNS = [
    r"^1 Introduction",
    r"^1\.1 Errors caused by pretraining",
    r"^1\.2 Why hallucinations survive post-training",
    r"^2 Related work",
    r"^3 Pretraining Errors",
    r"^3\.1 The reduction without prompts",
    r"^3\.2 The reduction with prompts",
    r"^3\.3 Error factors for base models",
    r"^3\.4 Additional factors",
    r"^4 Post-training and hallucination",
    r"^4\.1 How evaluations reinforce hallucination",
    r"^4\.2 Explicit confidence targets",
    r"^5 Discussion and limitations",
    r"^6 Conclusions",
    r"^References",
    r"^A Proof of the main theorem",
    r"^B Arbitrary-facts analysis",
    r"^C Poor-model analysis",
    r"^D Computationally intractable hallucinations",
    r"^E Post-training analysis",
    r"^F Current grading of uncertain responses",
    r"^F\.1 HELM Capabilities Benchmark",
    r"^F\.2 Open LLM Leaderboard",
    r"^F\.3 SWE-bench and Humanity\u2019s Last Exam",
]

MAX_WORDS_PER_CHUNK = 350  # sub-split long sections further


def load_text() -> str:
    txt_files = list(EXTRACTED_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError("Run extract_text.py first.")
    return txt_files[0].read_text(encoding="utf-8")


def split_into_sections(text: str) -> list[dict]:
    # Remove page markers for cleaner section splitting
    clean = re.sub(r"\[PAGE \d+\]", "", text)

    # Build a combined regex to find section boundaries
    pattern = "|".join(SECTION_PATTERNS)
    matches = list(re.finditer(pattern, clean, flags=re.MULTILINE))

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
        title = m.group().strip()
        body = clean[start:end].strip()
        sections.append({"section": title, "text": body})
    return sections


def sub_split_long_section(section: dict) -> list[dict]:
    """Further split a section into ~MAX_WORDS_PER_CHUNK-word chunks on paragraph
    boundaries, so no single chunk is too long for embedding/retrieval."""
    words = section["text"].split()
    if len(words) <= MAX_WORDS_PER_CHUNK:
        return [section]

    paragraphs = section["text"].split("\n\n")
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        plen = len(para.split())
        if current_len + plen > MAX_WORDS_PER_CHUNK and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += plen
    if current:
        chunks.append(" ".join(current))

    return [
        {"section": f"{section['section']} (part {i+1})", "text": c}
        for i, c in enumerate(chunks)
    ]


def main():
    text = load_text()
    sections = split_into_sections(text)

    all_chunks = []
    for sec in sections:
        all_chunks.extend(sub_split_long_section(sec))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            record = {
                "chunk_id": f"chunk_{i:03d}",
                "section": chunk["section"],
                "text": chunk["text"],
                "source": "Kalai et al. 2025 - Why Language Models Hallucinate",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()