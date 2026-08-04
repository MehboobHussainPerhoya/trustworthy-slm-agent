"""
hallucination_check.py
Given a generated answer and the retrieved context it was supposed to be
grounded in, uses an NLI (Natural Language Inference) model to check
whether each claim in the answer is actually supported ("entailed") by
the context, contradicted by it, or unsupported ("neutral" / not found).

Approach:
  1. Split the answer into individual sentences (claims).
  2. For each sentence, run NLI with the retrieved context as the premise
     and the sentence as the hypothesis.
  3. Aggregate per-sentence verdicts into an overall verdict:
       - "supported"          : every claim is entailed by the context
       - "partially_supported": some claims entailed, none contradicted,
                                 but at least one is neutral (not addressed
                                 by the context)
       - "unsupported"        : at least one claim is contradicted by the
                                 context (the strongest signal of hallucination)

Usage as a module:
    from hallucination_check import check_hallucination
    result = check_hallucination(answer, context)

Usage standalone (quick manual test):
    python src/hallucination_check.py
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import PreTrainedTokenizer, PreTrainedModel

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Thresholds for calling a sentence "entailed" or "contradicted" rather than
# just "neutral" — tuned to be conservative (favor flagging as unsupported
# over falsely claiming support), appropriate for a safety-oriented checker.
ENTAILMENT_THRESHOLD = 0.55
CONTRADICTION_THRESHOLD = 0.45

_tokenizer = None
_model = None


def _load_model():
    """Lazy-load the NLI model once, on first use."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading NLI model: {NLI_MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — good enough for model-generated answers,
    which tend to have clean punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def _nli_scores(premise: str, hypothesis: str) -> dict:
    """Runs NLI for a single (premise, hypothesis) pair. Returns a dict of
    label -> probability. Model label order: 0=entailment, 1=neutral,
    2=contradiction for this specific model (MoritzLaurer's mnli-fever-anli)."""
    tokenizer, model = _load_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {
        "entailment": float(probs[0]),
        "neutral": float(probs[1]),
        "contradiction": float(probs[2]),
    }


def check_hallucination(answer: str, context: str) -> dict:
    """
    Main entry point. Checks each sentence in `answer` against `context`.

    Returns:
        {
            "verdict": "supported" | "partially_supported" | "unsupported",
            "sentence_results": [
                {"sentence": str, "label": str, "scores": {...}}, ...
            ]
        }
    """
    sentences = split_sentences(answer)
    if not sentences:
        return {"verdict": "unsupported", "sentence_results": []}

    sentence_results = []
    for sentence in sentences:
        scores = _nli_scores(premise=context, hypothesis=sentence)
        if scores["contradiction"] >= CONTRADICTION_THRESHOLD:
            label = "contradicted"
        elif scores["entailment"] >= ENTAILMENT_THRESHOLD:
            label = "entailed"
        else:
            label = "neutral"
        sentence_results.append({"sentence": sentence, "label": label, "scores": scores})

    labels = [r["label"] for r in sentence_results]
    if "contradicted" in labels:
        verdict = "unsupported"
    elif all(l == "entailed" for l in labels):
        verdict = "supported"
    else:
        verdict = "partially_supported"

    return {"verdict": verdict, "sentence_results": sentence_results}


if __name__ == "__main__":
    # Quick manual sanity test
    context = (
        "The singleton rate is the fraction of prompts that appear exactly "
        "once in the training data with a real, non-abstaining answer."
    )

    print("=== Test 1: correct, supported answer ===")
    answer_good = "The singleton rate is the fraction of prompts appearing exactly once in training data."
    result = check_hallucination(answer_good, context)
    print(result["verdict"])
    for r in result["sentence_results"]:
        print(f"  [{r['label']}] {r['sentence']}")

    print("\n=== Test 2: fabricated, unsupported answer ===")
    answer_bad = "The singleton rate measures the fraction of token pairs the model predicts correctly."
    result = check_hallucination(answer_bad, context)
    print(result["verdict"])
    for r in result["sentence_results"]:
        print(f"  [{r['label']}] {r['sentence']}")