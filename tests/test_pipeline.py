"""
test_pipeline.py
Smoke tests for the Trustworthy SLM Agent pipeline. These are sanity
checks, not full correctness tests — they confirm each piece still loads,
runs, and produces reasonably-shaped output, so you can catch a broken
dependency or accidental file corruption before spending time in Colab.

Most tests run entirely on CPU with no GPU required. A few (marked with
`@pytest.mark.gpu`) need a real fine-tuned model and are automatically
skipped if no CUDA GPU is available — run those specifically in Colab.

Usage (from project root, in the activated venv):
    pytest tests/test_pipeline.py -v

    # to also attempt GPU-dependent tests (only meaningful in Colab):
    pytest tests/test_pipeline.py -v -m gpu
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Data integrity tests
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"Missing expected file: {path}"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_chunks_file_valid():
    chunks = _load_jsonl(PROJECT_ROOT / "data" / "chunks.jsonl")
    assert len(chunks) >= 50, f"Expected at least 50 chunks, found {len(chunks)}"
    for c in chunks[:5]:
        for key in ("chunk_id", "section", "text", "source"):
            assert key in c, f"Chunk missing expected key: {key}"


def test_qa_pairs_file_valid():
    pairs = _load_jsonl(PROJECT_ROOT / "data" / "qa_pairs.jsonl")
    assert len(pairs) >= 150, f"Expected at least 150 Q&A pairs, found {len(pairs)}"
    for p in pairs[:5]:
        assert "question" in p and "answer" in p


def test_train_val_test_split_valid():
    train = _load_jsonl(PROJECT_ROOT / "data" / "qa_train.jsonl")
    val = _load_jsonl(PROJECT_ROOT / "data" / "qa_val.jsonl")
    test = _load_jsonl(PROJECT_ROOT / "data" / "qa_test.jsonl")
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # No leakage: no identical questions should appear in more than one split
    train_qs = {r["question"] for r in train}
    val_qs = {r["question"] for r in val}
    test_qs = {r["question"] for r in test}
    assert train_qs.isdisjoint(val_qs), "Found overlapping questions between train and val"
    assert train_qs.isdisjoint(test_qs), "Found overlapping questions between train and test"
    assert val_qs.isdisjoint(test_qs), "Found overlapping questions between val and test"


def test_bias_probes_file_valid():
    probes = _load_jsonl(PROJECT_ROOT / "data" / "bias_probes.jsonl")
    assert len(probes) >= 30
    axes = {p["axis"] for p in probes}
    assert "gender_occupation" in axes
    assert "nationality_sentiment" in axes


def test_hallucination_eval_file_valid():
    items = _load_jsonl(PROJECT_ROOT / "data" / "hallucination_eval.jsonl")
    assert len(items) >= 15
    in_scope_count = sum(1 for i in items if i["in_scope"])
    out_scope_count = sum(1 for i in items if not i["in_scope"])
    assert in_scope_count > 0 and out_scope_count > 0, \
        "Eval set should mix in-scope and out-of-scope questions"


# ---------------------------------------------------------------------------
# Config validity
# ---------------------------------------------------------------------------

def test_lora_config_valid():
    config_path = PROJECT_ROOT / "configs" / "lora_config.yaml"
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("base_model"), "Missing base_model in config"
    for key in ("r", "alpha", "dropout", "target_modules"):
        assert key in cfg.get("lora", {}), f"Missing lora.{key} in config"
    for key in ("num_train_epochs", "learning_rate"):
        assert key in cfg.get("training", {}), f"Missing training.{key} in config"
    assert cfg.get("hub", {}).get("repo_id"), "Missing hub.repo_id in config"


# ---------------------------------------------------------------------------
# Retrieval index tests (CPU only — small embedding model, no GPU needed)
# ---------------------------------------------------------------------------

def test_retrieval_index_files_exist():
    index_dir = PROJECT_ROOT / "data" / "index"
    assert (index_dir / "chunks.faiss").exists()
    assert (index_dir / "chunk_metadata.json").exists()
    assert (index_dir / "embedding_model.txt").exists()


def test_retriever_returns_results():
    """Loads the actual FAISS index and embedding model (small, CPU-friendly,
    ~90MB download on first run) and confirms retrieval returns sensible
    results for a known in-domain query."""
    from agent import Retriever

    retriever = Retriever()
    results = retriever.retrieve("What is the singleton rate?", k=3)

    assert len(results) == 3
    for r in results:
        assert "section" in r and "text" in r and "score" in r
        assert isinstance(r["score"], float)
    # top result should be a real similarity score, not zero/garbage
    assert results[0]["score"] > 0


# ---------------------------------------------------------------------------
# Pure-function unit tests (no model loading at all)
# ---------------------------------------------------------------------------

def test_token_f1_scoring():
    from evaluate import token_f1

    identical = token_f1("the singleton rate is a fraction", "the singleton rate is a fraction")
    assert identical == pytest.approx(1.0)

    unrelated = token_f1("completely different sentence here", "singleton rate fraction training")
    assert unrelated < 0.3

    empty = token_f1("", "something")
    assert empty == 0.0


def test_hallucination_check_sentence_splitting():
    from hallucination_check import split_sentences

    sentences = split_sentences("This is one sentence. This is another! And a third?")
    assert len(sentences) == 3

    empty_result = split_sentences("")
    assert empty_result == []


def test_chunk_document_section_patterns_cover_appendices():
    """Regression test for the section-mislabeling bug fixed in Phase 1 —
    confirms References and all Appendix headers are present in the
    patterns list, not just the main body sections."""
    import chunk_document

    patterns_text = " ".join(chunk_document.SECTION_PATTERNS)
    assert "References" in patterns_text
    assert "Arbitrary-facts analysis" in patterns_text
    assert "F\\.1" in patterns_text or "F.1" in patterns_text


# ---------------------------------------------------------------------------
# GPU-dependent tests — skipped automatically unless a CUDA GPU is present
# (run these specifically in Colab, not on the local Windows machine)
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _has_gpu(), reason="No CUDA GPU available — run this test in Colab")
def test_agent_end_to_end():
    from agent import Agent

    agent = Agent()
    result = agent.ask("What is the singleton rate?")

    assert result["verdict"] in ("supported", "partially_supported", "unsupported")
    assert len(result["answer"]) > 0
    assert len(result["sources"]) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not _has_gpu(), reason="No CUDA GPU available — run this test in Colab")
def test_hallucination_check_end_to_end():
    from hallucination_check import check_hallucination

    context = "The singleton rate is the fraction of prompts appearing exactly once."
    good_answer = "The singleton rate is the fraction of prompts that appear once."
    result = check_hallucination(good_answer, context)

    assert result["verdict"] in ("supported", "partially_supported", "unsupported")
    assert len(result["sentence_results"]) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))