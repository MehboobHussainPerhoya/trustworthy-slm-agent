"""
evaluate_hallucination.py
Runs the full Agent (retrieval + generation + hallucination check) over
data/hallucination_eval.jsonl and computes summary metrics:
  - Overall verdict distribution (supported / partially_supported / unsupported)
  - Breakdown by whether the question was in-scope (answerable from the
    paper) or deliberately out-of-scope (testing appropriate abstention)

For in-scope questions: "supported" is the desired outcome.
For out-of-scope questions: since there's no real answer available, either
  an explicit refusal/hedge in the answer text, OR an "unsupported" /
  "partially_supported" verdict counts as correct behavior — a confident
  "supported" verdict on an out-of-scope question is the worst outcome,
  since it means the model fabricated an answer AND the checker missed it.

Usage (from project root, needs GPU realistically — run in Colab):
    python src/evaluate_hallucination.py
"""

import json
from pathlib import Path
from collections import Counter

from agent import Agent

EVAL_SET_PATH = Path("data/hallucination_eval.jsonl")
RESULTS_PATH = Path("results/hallucination_eval_results.jsonl")
SUMMARY_PATH = Path("results/hallucination_eval_summary.json")

REFUSAL_PHRASES = [
    "i don't know", "i do not know", "not covered", "does not cover",
    "not mentioned", "does not mention", "outside the scope",
    "cannot answer", "can't answer", "no information", "not addressed",
    "does not discuss", "does not address", "not able to answer",
]


def looks_like_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def load_eval_set() -> list[dict]:
    items = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    eval_items = load_eval_set()
    print(f"Loaded {len(eval_items)} eval questions "
          f"({sum(1 for i in eval_items if i['in_scope'])} in-scope, "
          f"{sum(1 for i in eval_items if not i['in_scope'])} out-of-scope)")

    print("Initializing agent...")
    agent = Agent()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for item in eval_items:
        question = item["question"]
        in_scope = item["in_scope"]
        print(f"\nAsking: {question}")

        result = agent.ask(question)
        refused = looks_like_refusal(result["answer"])

        # Determine "correct behavior" per the rules described in the docstring
        if in_scope:
            correct_behavior = result["verdict"] == "supported"
        else:
            correct_behavior = refused or result["verdict"] in ("unsupported", "partially_supported")

        record = {
            "question": question,
            "in_scope": in_scope,
            "answer": result["answer"],
            "verdict": result["verdict"],
            "looks_like_refusal": refused,
            "correct_behavior": correct_behavior,
        }
        results.append(record)
        print(f"  verdict={result['verdict']}  refusal={refused}  correct={correct_behavior}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved detailed results to {RESULTS_PATH}")

    # Summary metrics
    verdict_counts = Counter(r["verdict"] for r in results)
    in_scope_results = [r for r in results if r["in_scope"]]
    out_scope_results = [r for r in results if not r["in_scope"]]

    in_scope_correct = sum(1 for r in in_scope_results if r["correct_behavior"])
    out_scope_correct = sum(1 for r in out_scope_results if r["correct_behavior"])

    summary = {
        "total_questions": len(results),
        "verdict_distribution": dict(verdict_counts),
        "in_scope": {
            "count": len(in_scope_results),
            "correct_(supported)": in_scope_correct,
            "accuracy": round(in_scope_correct / len(in_scope_results), 3) if in_scope_results else None,
        },
        "out_of_scope": {
            "count": len(out_scope_results),
            "correct_(refused_or_flagged)": out_scope_correct,
            "appropriate_abstention_rate": round(out_scope_correct / len(out_scope_results), 3) if out_scope_results else None,
        },
        "overall_hallucination_rate": round(
            sum(1 for r in results if not r["correct_behavior"]) / len(results), 3
        ) if results else None,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()