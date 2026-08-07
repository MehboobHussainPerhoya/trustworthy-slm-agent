# Hallucination Evaluation Analysis

**Eval set:** 20 questions (10 in-scope, answerable from the paper; 10
deliberately out-of-scope, e.g. "capital of France," "price of Bitcoin").
**Checker:** MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (NLI entailment).

## Summary Metrics

| Metric | Result |
|---|---|
| In-scope accuracy (verdict = "supported") | 60% (6/10) |
| Out-of-scope appropriate flagging | 100% (10/10) |
| Overall hallucination rate | 20% |

## Key Finding: The checker caught a real hallucination on the most important question

Asked "Why do language models hallucinate according to this paper?", the
agent answered that models hallucinate because they "lack calibrated belief
estimates." This is inconsistent with the paper's actual finding — pretrained
base models tend to be well-calibrated; miscalibration emerges after RL
post-training, not as a root cause. The checker correctly flagged this as
**unsupported**. This is the system working as designed: catching a
plausible-sounding but incorrect explanation on the single most central
question the agent exists to answer.

## Manual review of the 3 "partially_supported" in-scope cases

- **IIV problem, GIGO definition:** both answers were accurate, reasonable
  paraphrases of the paper's definitions. Likely **false alarms** — the NLI
  checker appears conservative when answer phrasing diverges from the
  retrieved text's exact wording, even when the meaning is correct.
- **Trigram model example:** the answer conflated two separate examples
  (the pronoun/trigram example and the unrelated Figure 1 spelling
  illustration). This is a genuine factual error that arguably should have
  been flagged as "unsupported" rather than "partially_supported" — a case
  where the checker under-flagged a real hallucination.

## Interpretation

- The 100% out-of-scope abstention/flagging rate is the strongest result:
  the agent never let a fabricated answer to an unanswerable question pass
  through as "supported."
- The 60% in-scope "accuracy" understates true performance, since at least
  2 of the 4 non-"supported" cases were likely correct answers marked down
  by an overly conservative checker, not real hallucinations.
- The checker's conservatism is a defensible design choice for a
  trustworthy-AI system (under-claiming confidence is safer than
  over-claiming it), but it should be stated explicitly as a known
  limitation rather than presenting the 60% figure without this context.
- One real miss was identified (the trigram/Figure-1 conflation), showing
  the checker is not perfectly reliable and should not be treated as a
  ground-truth oracle.

## Known Limitations

1. NLI-based checking evaluates surface-level entailment, not deep factual
   verification — it can be fooled by paraphrase mismatches and may miss
   subtle factual conflations.
2. Eval set size (20 questions) is small; results are illustrative, not
   statistically definitive.
3. Sentence-level splitting may not perfectly isolate individual claims in
   longer, multi-clause answers.