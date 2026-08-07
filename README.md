# Trustworthy SLM Agent

A small language model, fine-tuned on a single research paper, wrapped as a
retrieval-grounded agent with automated hallucination detection and bias
auditing — built end-to-end on free-tier compute as a demonstration of
responsible AI deployment practices.

**Domain:** The agent answers questions about *"Why Language Models
Hallucinate"* (Kalai, Nachum, Vempala, and Zhang, 2025), arXiv:2509.04664.
The project is intentionally self-referential: an AI-safety-audited agent
that explains the theory of why models hallucinate.

**Live demo:** temporary Colab-hosted public link (active while the Colab
session is running): see [Live Demo](#live-demo) below. Permanent
Hugging Face Spaces (ZeroGPU) deployment is planned once account
eligibility is confirmed.

---

## Why this project

Most SLM fine-tuning tutorials stop at "here's your fine-tuned model."
This project asks the next question: **is it safe and honest to deploy?**
It fine-tunes a 1.5B model, then builds three layers on top of it —
retrieval grounding, automated hallucination checking, and bias auditing —
and reports the *real*, sometimes imperfect, results of each layer rather
than only showcasing successes.

## Architecture

```
User question
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Retriever   │────▶│  Fine-tuned SLM   │────▶│ Hallucination Check│
│ (FAISS +     │     │ (Qwen2.5-1.5B +   │     │ (NLI entailment,   │
│  MiniLM      │     │  LoRA adapter)    │     │  DeBERTa-v3-mnli)  │
│  embeddings) │     │                   │     │                    │
└─────────────┘     └──────────────────┘     └───────────────────┘
                                                          │
                                                          ▼
                                    Verdict-gated response:
                                    supported → show answer
                                    partially supported → show with caution
                                    unsupported → withhold, explain why
```

## Repo Structure

```
trustworthy-slm-agent/
├── app/gradio_app.py          # Gradio UI (Phase 8)
├── configs/lora_config.yaml   # LoRA + training hyperparameters
├── data/                      # source text, chunks, Q&A pairs, indices
├── docs/SPEC.md               # full functional specification
├── notebooks/                 # Colab fine-tuning notebooks
├── results/                   # all evaluation outputs and analysis writeups
├── src/                       # all pipeline scripts (see below)
├── tests/test_pipeline.py     # smoke tests
└── MODEL_CARD.md              # auto-generated model card
```

## Setup (Windows / VS Code)

```powershell
git clone https://github.com/MehboobHussainPerhoya/trustworthy-slm-agent.git
cd trustworthy-slm-agent
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Heavy/GPU-dependent steps (fine-tuning, agent generation, evaluation) are
designed to run on free-tier Google Colab, not the local machine. See
`docs/SPEC.md` for the full phase-by-phase pipeline and exact Colab setup
commands.

## Pipeline / How to Reproduce

| Step | Script | What it does |
|---|---|---|
| 1 | `src/extract_text.py` | Extract text from the source PDF |
| 2 | `src/chunk_document.py` | Split into 55 labeled section chunks |
| 3 | `src/split_dataset.py` | 70/15/15 train/val/test split of Q&A pairs |
| 4 | `src/finetune.py` | LoRA fine-tune Qwen2.5-1.5B-Instruct |
| 5 | `src/build_index.py` | Build FAISS retrieval index |
| 6 | `src/agent.py` | Retrieval-grounded agent (CLI or import) |
| 7 | `src/evaluate_hallucination.py` | Hallucination eval harness |
| 8 | `src/bias_audit.py` | Gender + nationality bias audit |
| 9 | `src/evaluate.py` | Base vs. fine-tuned F1/latency benchmark |
| 10 | `src/generate_model_card.py` | Auto-generate `MODEL_CARD.md` |
| 11 | `app/gradio_app.py` | Web UI |

## Results

### Fine-Tuning (Phase 2)

LoRA fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` (r=16, alpha=32) on 116
human-reviewed Q&A pairs, 3 epochs, free Colab T4 GPU.

| Epoch | Eval Loss | Eval Token Accuracy |
|---|---|---|
| 1 | 1.534 | 71.2% |
| 2 | 1.244 | 76.3% |
| 3 | 1.228 | 76.5% |

Trainable parameters: 18,464,768 / 1,562,179,072 (1.18%).

### Base vs. Fine-Tuned Benchmark (Phase 7)

Evaluated on the full 26-question held-out test set, token-level F1 against
reference answers:

| Metric | Base Model | Fine-Tuned | Change |
|---|---|---|---|
| Mean token F1 | 0.135 | 0.185 | **+0.051 (≈38% relative)** |
| Mean latency | 10.34s | 6.86s | **33% faster** |
| Model size | ~1.5B params | +37MB LoRA adapter | — |

19 of 26 test questions improved with fine-tuning; a few regressed
(notably both RAG-related questions) — reported honestly rather than
cherry-picked.

### Hallucination Detection (Phase 4)

NLI-based entailment checking (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)
against retrieved context, evaluated on 20 questions (10 in-scope,
10 deliberately out-of-scope):

| Metric | Result |
|---|---|
| In-scope accuracy (verdict = "supported") | 60% |
| Out-of-scope appropriate abstention/flagging | **100%** |
| Overall hallucination rate | 20% |

**Key finding:** the checker correctly flagged a real hallucination in the
agent's answer to its own most central question ("why do models
hallucinate") — the model claimed models "lack calibrated belief
estimates," which actually contradicts the paper's finding that base
models tend to be well-calibrated. Manual review also found the checker to
be conservative (some correct paraphrased answers scored only "partially
supported") and to have missed at least one genuine factual conflation —
documented honestly in `results/hallucination_eval_analysis.md`.

### Bias Audit (Phase 5)

Two axes tested on the fine-tuned model:

- **Gender-occupation** (20 occupations): strong, stereotype-consistent
  pronoun skew in 19/20 occupations (e.g. mechanic skews strongly toward
  "he," babysitter strongly toward "she"). Two notable exceptions:
  hairdresser skews counter-stereotypically toward "he"; flight attendant
  shows exactly zero skew.
- **Nationality sentiment** (18 groups): no meaningful variance detected
  (all near-maximum positive) — likely reflects a coarse sentiment
  classifier's limitation rather than genuine absence of bias.

Full details, plot, and interpretation: `results/bias_audit_analysis.md`,
`results/bias_audit_plot.png`.

## Live Demo

The agent is deployed as a Gradio app. Answers are gated by hallucination
verdict — "unsupported" answers are withheld entirely rather than shown
with just a warning label, and "partially supported" answers are shown
with an explicit caution prefix.

Currently hosted via a temporary Colab `share=True` public link (see
project maintainer for current active link, or run
`app/gradio_app.py` yourself in Colab per `docs/SPEC.md`). Permanent
Hugging Face Spaces (ZeroGPU) deployment is pending account eligibility
confirmation.

## Known Limitations

1. Small fine-tuning dataset (167 Q&A pairs) — sparse technical facts
   appearing in only 1-2 training examples are not reliably learned by
   fine-tuning alone; retrieval grounding mitigates but does not
   eliminate this.
2. The hallucination checker is conservative and imperfect — it can
   under-score correct paraphrased answers, and has been observed to miss
   at least one genuine factual error. It is not a ground-truth oracle.
3. Measurable gender bias is present in the model, inherited primarily
   from the base model's pretraining, not introduced by this project's
   narrow fine-tuning.
4. The nationality-sentiment bias axis is likely under-measured by a
   coarse sentiment classifier; a content/stereotype-specific analysis
   would be a more rigorous follow-up.
5. Narrow domain only — this model is not evaluated and not intended for
   use outside questions about this specific paper.
6. Live demo hosting is currently temporary (Colab-hosted), pending
   permanent free-tier deployment eligibility.

## Full Model Card

See [`MODEL_CARD.md`](MODEL_CARD.md) for the complete auto-generated model
card, and `docs/SPEC.md` for the full functional specification this
project was built against.

## Future Work

- Expand the Q&A dataset to improve coverage of sparse/technical concepts.
- Content-level (not just sentiment-polarity) bias analysis.
- Permanent Hugging Face Spaces ZeroGPU deployment.
- Multi-document support beyond the single source paper.

## License / Data Provenance

Source paper: Kalai et al. (2025), CC BY 4.0, arXiv:2509.04664. See
`data/README.md` for full provenance of derived data. Code in this
repository is provided for educational/portfolio purposes.