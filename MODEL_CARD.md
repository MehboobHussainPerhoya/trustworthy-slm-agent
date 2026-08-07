# Model Card: Trustworthy SLM Agent (Qwen2.5-1.5B, LoRA fine-tuned)

*Auto-generated on 2026-08-06. Regenerate with `python src/generate_model_card.py`.*

## Model Details

- **Base model:** Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning method:** LoRA (rank 16, alpha 32, dropout 0.05), applied to: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Adapter repository:** Mehboobali512/trustworthy-slm-agent-qwen2.5-1.5b
- **Developed by:** [Mehboob Hussain]
- **Model type:** Causal language model, instruction-tuned for narrow-domain Q&A
- **Language:** English

## Intended Use

**Primary use case:** Answering questions about the paper *"Why Language
Models Hallucinate"* (Kalai, Nachum, Vempala, and Zhang, 2025), grounded via
retrieval over the paper's text, with automated hallucination and bias
checking as part of a research/portfolio project demonstrating responsible
AI deployment practices.

**Out of scope uses:** This model is not intended for general-purpose
question answering, use outside the paper's specific domain, or any
production/high-stakes decision-making context. It is a narrow-domain
research demonstration, not a general-purpose assistant.

## Training Data

- **Source:** Kalai, Nachum, Vempala, and Zhang (2025), "Why Language Models Hallucinate," arXiv:2509.04664
- **Dataset size:** 167 human-reviewed Q&A pairs
  (116 train / 25 val / 26 test, 70/15/15 split, seed 42)
- **Provenance:** Q&A pairs were LLM-drafted from source text chunks, then
  manually reviewed and corrected before use. See `data/README.md` for full
  provenance and licensing details.

## Training Procedure

- **Quantization:** 4-bit QLoRA (nf4, bfloat16 compute dtype) during training
- **Epochs:** 3
- **Effective batch size:** 16
- **Learning rate:** 0.0002 (cosine schedule)
- **Hardware:** Free-tier Google Colab T4 GPU

### Training Results

| Epoch | Eval Loss | Eval Token Accuracy |
|---|---|---|
| 1 | 1.534 | 0.712 |
| 2 | 1.244 | 0.7632 |
| 3 | 1.228 | 0.7653 |


- **Final train loss:** 1.171
- **Trainable parameters:** 18,464,768 / 1,562,179,072 (1.182%)

## Evaluation

### Qualitative: Base vs. Fine-Tuned

Fine-tuning measurably improved response conciseness, topic focus, and fixed
at least one clear factual/terminology error (a fabricated RAG acronym
expansion in the base model). It did not fully resolve factual accuracy on
sparse, technical concepts underrepresented in the small training set (e.g.
"singleton rate"). Full detail: `results/qualitative_comparison.md`.

### Hallucination Detection (retrieval-grounded agent)

Evaluated on a 20-question set (10 in-scope, 10 deliberately out-of-scope),
using an NLI-based entailment checker (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
against retrieved context:

| Metric | Result |
|---|---|
| In-scope accuracy (verdict = "supported") | 0.6 |
| Out-of-scope appropriate abstention/flagging rate | 1.0 |
| Overall hallucination rate | 0.2 |

**Key finding:** the checker correctly flagged a real hallucination on the
model's answer to its single most central question ("why do models
hallucinate"), and correctly flagged 100% of out-of-scope questions rather
than letting fabricated answers pass silently. Manual review also found the
checker to be conservative (some correct answers scored only "partially
supported" due to paraphrase mismatch) and to have missed at least one real
factual conflation. Full detail: `results/hallucination_eval_analysis.md`.

### Bias Audit

Tested on two axes: gender-occupation pronoun association (20
occupations) and nationality/group sentiment association (18 groups).

- **Gender-occupation:** strong, mostly stereotype-consistent skew found
  across 19/20 occupations tested (e.g. mechanic skewed strongly toward
  "he", babysitter strongly toward "she"). One clear counter-stereotypical
  result (hairdresser) and one neutral result (flight attendant) were also
  observed.
- **Nationality sentiment:** no meaningful variance detected across groups
  (all scored near-maximum positive sentiment) — likely reflects a
  limitation of coarse sentiment scoring rather than genuine absence of bias.

Full detail: `results/bias_audit_analysis.md`, `results/bias_audit_plot.png`.

## Known Limitations

1. **Small fine-tuning dataset** (167 pairs) — sparse,
   technical facts appearing in only 1-2 training examples are not reliably
   learned by fine-tuning alone; retrieval grounding mitigates but does not
   eliminate this.
2. **Hallucination checker is conservative and imperfect** — it can
   under-score correct paraphrased answers and has been observed to miss at
   least one genuine factual error. It should not be treated as a
   ground-truth oracle.
3. **Measurable gender bias present**, inherited primarily from the base
   model's pretraining, not introduced by this project's narrow fine-tuning.
4. **Nationality-sentiment bias axis is likely under-measured** by a coarse
   sentiment classifier; a content/stereotype-specific analysis would be a
   more rigorous follow-up.
5. **Narrow domain only** — this model is not evaluated and not intended
   for use outside questions about this specific paper.

## Ethical Considerations

This model was built as a demonstration of responsible AI deployment
practices — fine-tuning, retrieval grounding, automated hallucination
detection, and bias auditing — rather than as a production system. All
evaluation results, including negative/limitation findings, are reported
here rather than omitted, in line with the project's transparency goals.

## How to Reproduce

See the project README and `docs/SPEC.md` for full setup instructions.
Training: `src/finetune.py`. Evaluation: `src/evaluate_hallucination.py`,
`src/bias_audit.py`. This card: `src/generate_model_card.py`.