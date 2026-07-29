# Data Provenance

**Source document:** Kalai, A. T., Nachum, O., Vempala, S. S., & Zhang, E. (2025).
*Why Language Models Hallucinate*. arXiv:2509.04664v1 [cs.CL].
https://arxiv.org/abs/2509.04664

**License / usage note:** This paper is used for personal, non-commercial,
educational research purposes only (fine-tuning a small demonstration model
and building a retrieval-grounded Q&A agent). The original PDF is not
redistributed in this repository — only derived Q&A pairs and short excerpted
chunks used for retrieval are included, with full citation to the source.

**Processing pipeline:**
1. `src/extract_text.py` — extracts text via pdfplumber
2. `src/chunk_document.py` — splits into 48 semantic chunks by section
   (Sections 1-6; References and mathematical Appendices A-F excluded from
   Q&A generation, retained separately for retrieval completeness)
3. Q&A pairs in `qa_pairs.jsonl` are human-written/human-reviewed paraphrases
   of the paper's claims — not verbatim excerpts.
4. `src/split_dataset.py` — 70/15/15 train/val/test split, seed=42