# Trustworthy SLM Agent — Functional Specification & Execution Plan

**Author:** [Your Name]
**Status:** Draft v1.0
**Target:** GitHub portfolio project, reproducible on free-tier compute

---

## 1. Project Overview

### 1.1 Problem Statement
Small Language Models (SLMs) are increasingly deployed for narrow, cost-sensitive
tasks (academic assistants, internal tools, edge deployment). However, most
open-source fine-tuning tutorials stop at "here's your fine-tuned model" without
addressing whether the model is *safe and trustworthy to deploy* — does it
hallucinate, does it exhibit bias, and is it documented well enough for someone
else to audit and use responsibly?

### 1.2 Objective
Build a single, reproducible, end-to-end pipeline that:
1. Fine-tunes a small open-source LLM (SLM) on a narrow academic/industry task.
2. Wraps that model as an **agent** capable of answering queries using tools
   (retrieval, calculator, etc.).
3. Runs the agent's outputs through a **hallucination detector**
   (NLI-based factual consistency check against source documents).
4. Runs the base/fine-tuned model through a **bias & fairness audit**
   (templated bias probes, e.g. WinoBias-style).
5. Auto-generates a **Model Card** (Mitchell et al. 2019 format) summarizing
   training data, intended use, evaluation results, hallucination rate, and
   bias audit findings.
6. Exposes the whole thing via a simple **Gradio/Streamlit UI**, deployed
   free on Hugging Face Spaces.

### 1.3 Why This Project (Research Alignment)
| Research Interest | How This Project Covers It |
|---|---|
| NLP / Transformers | Fine-tuning + NLI-based entailment checking (transformer classifiers) |
| AI safety / trustworthy AI | Hallucination detection, bias auditing, guardrails |
| SLMs for custom solutions | Core deliverable: task-specific fine-tuned small model |
| LLMs / autonomous agents | Agent wrapper with tool use and guardrail-gated responses |
| Responsible AI deployment | Model card generation, audit reports, documented limitations |

### 1.4 Non-Goals (explicitly out of scope, keep this pragmatic)
- No training from scratch (fine-tuning only).
- No paid APIs (OpenAI GPT-4, Claude API, etc.) — open-source models only.
- No multi-GPU / cluster training — must run on free Colab/Kaggle T4 (16GB VRAM).
- Not aiming for SOTA accuracy — aiming for a **rigorous, well-documented,
  reproducible pipeline**.

---

## 2. System Architecture

```
                        ┌─────────────────────────┐
                        │   User Query (UI/CLI)   │
                        └────────────┬─────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │       Agent Layer        │
                        │  (LangGraph / custom     │
                        │   ReAct loop)             │
                        │  - decides tool use       │
                        │  - calls fine-tuned SLM   │
                        └────┬───────────────┬──────┘
                             ▼               ▼
                  ┌───────────────┐  ┌───────────────┐
                  │ Retrieval Tool │  │ Fine-tuned SLM │
                  │ (FAISS/Chroma) │  │ (LoRA adapter) │
                  └───────────────┘  └───────┬───────┘
                                              ▼
                        ┌─────────────────────────────┐
                        │   Guardrail / Safety Layer    │
                        │  1. Hallucination check        │
                        │     (NLI entailment vs source) │
                        │  2. Refuse/flag if unsupported  │
                        └────────────┬─────────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │   Final Response to User  │
                        │  (+ confidence/flag label) │
                        └─────────────────────────┘

        (offline, run once / on-demand — not in the request path)
                        ┌─────────────────────────┐
                        │      Bias Audit Module     │
                        │  templated probes → metrics │
                        └────────────┬─────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │  Model Card Generator      │
                        │  (reads eval + audit logs)  │
                        │  → MODEL_CARD.md            │
                        └─────────────────────────┘
```

### 2.1 Core Components

| # | Component | Description | Key Libraries |
|---|---|---|---|
| 1 | **Base model** | Small open model (0.5B–3B params) | `Qwen2.5-1.5B-Instruct`, `Phi-3-mini`, or `Gemma-2-2b-it` |
| 2 | **Fine-tuning module** | LoRA/QLoRA fine-tune on a narrow task/dataset | `peft`, `transformers`, `bitsandbytes`, `trl` |
| 3 | **Agent orchestration** | ReAct-style loop deciding when to retrieve vs. answer directly | `LangGraph` or hand-rolled Python state machine |
| 4 | **Retrieval tool (RAG)** | Vector search over a small curated corpus (grounds answers, feeds hallucination checker) | `sentence-transformers`, `FAISS` or `Chroma` |
| 5 | **Hallucination detector** | NLI entailment model checks if answer is supported by retrieved context | `roberta-large-mnli` or `vectara/hallucination_evaluation_model` |
| 6 | **Bias audit module** | Runs templated prompt pairs (e.g. gendered occupation prompts), measures skew | Custom scripts + `pandas`, `matplotlib` |
| 7 | **Model card generator** | Python script that reads eval/audit JSON logs and renders a Markdown model card | Jinja2 template |
| 8 | **UI / demo** | Chat interface showing answer + hallucination flag + "trust score" | `Gradio`, hosted on HF Spaces (free tier) |
| 9 | **Eval harness** | Task accuracy, hallucination rate, bias metrics, latency | `evaluate`, custom scripts |

### 2.2 Tech Stack (100% free tier)

- **Compute:** Google Colab free T4 GPU (or Kaggle, 30 free GPU hrs/week)
- **Model hosting:** Hugging Face Hub (free public repo for LoRA adapter)
- **Demo hosting:** Hugging Face Spaces (free CPU or ZeroGPU tier)
- **Experiment tracking:** Weights & Biases free tier (optional) or plain JSON logs
- **Version control:** GitHub

---

## 3. Functional Requirements

### FR1 — Fine-Tuning Pipeline
- FR1.1: Load a base SLM (≤3B params) in 4-bit via `bitsandbytes`.
- FR1.2: Apply LoRA fine-tuning on a chosen narrow dataset (see §5 for dataset choice).
- FR1.3: Save adapter weights + training config + training loss curve.
- FR1.4: Script must be a single runnable notebook/script, parameterized (model name, dataset path, LoRA rank, epochs).

### FR2 — Agent Layer
- FR2.1: Accept a natural language query.
- FR2.2: Decide whether to retrieve context (RAG) or answer directly.
- FR2.3: Call the fine-tuned SLM with retrieved context injected into the prompt.
- FR2.4: Return both the answer and the retrieved source snippets (for auditability).

### FR3 — Hallucination Detection
- FR3.1: Given (answer, source context), compute an entailment score using an NLI model.
- FR3.2: Classify into `supported / partially supported / unsupported`.
- FR3.3: If `unsupported`, surface a warning to the user in the UI rather than silently returning the answer.
- FR3.4: Log every (query, answer, context, verdict) tuple for later analysis.

### FR4 — Bias Audit
- FR4.1: Maintain a bank of ≥30 templated bias-probe prompts across ≥2 axes (e.g., gender-occupation, nationality-sentiment).
- FR4.2: Run all probes through the model, collect completions.
- FR4.3: Score outputs (e.g., sentiment skew, stereotype-consistent completion rate).
- FR4.4: Output a report (`bias_audit_report.json` + a plot) summarizing skew per axis.

### FR5 — Model Card Generation
- FR5.1: Auto-populate a model card from: base model metadata, training config, eval metrics, hallucination rate, bias audit summary, known limitations.
- FR5.2: Output as `MODEL_CARD.md` following the Hugging Face / Mitchell et al. schema.
- FR5.3: Re-runnable — regenerating the card after new eval runs should be a single command.

### FR6 — Evaluation Harness
- FR6.1: Task-specific accuracy/F1 (before vs. after fine-tuning).
- FR6.2: Hallucination rate (% unsupported answers on a held-out eval set).
- FR6.3: Bias metrics (from FR4).
- FR6.4: Latency and model size (for the SLM cost-performance narrative).

### FR7 — UI / Demo
- FR7.1: Simple chat UI (Gradio) where user types a question.
- FR7.2: Display answer + hallucination verdict + retrieved sources.
- FR7.3: Link to the model card from within the UI.

---

## 4. Non-Functional Requirements

- **Reproducibility:** Anyone should be able to clone the repo, run `pip install -r requirements.txt`, and reproduce fine-tuning + eval on a free Colab T4 in under a few hours.
- **Cost:** $0. No paid API keys required anywhere in the pipeline.
- **Documentation:** README with architecture diagram, setup steps, and results table. Every module has docstrings.
- **Modularity:** Each component (fine-tune, agent, hallucination check, bias audit, model card) must run standalone via CLI, not just inside one monolithic notebook.
- **Transparency:** All prompts (bias probes, agent system prompt) checked into the repo — nothing hidden.

---

## 5. Concrete Design Decisions (to stop analysis-paralysis)

To keep this pragmatic, commit to these defaults unless you have a strong reason to change them:

- **Base model:** `Qwen2.5-1.5B-Instruct` (small, strong instruction-following, fits comfortably in free Colab T4 with 4-bit quantization).
- **Task/domain:** A **university course Q&A assistant** — fine-tune on a small curated Q&A dataset built from a course syllabus / lecture notes / an open textbook (you generate ~300–500 Q&A pairs, or use an existing academic QA dataset like SQuAD subset or a CS-domain QA set). This is relatable, demoable, and easy to build a retrieval corpus for.
- **Retrieval corpus:** The same source documents (syllabus, notes, textbook chapters) chunked and embedded — this doubles as your hallucination-check ground truth.
- **Hallucination detector:** `vectara/hallucination_evaluation_model` (purpose-built, free, small) or fallback to `roberta-large-mnli` entailment.
- **Bias probes:** Start with a trimmed WinoBias-style template set (~30–40 prompts) — small enough to build by hand in a day, large enough to be statistically meaningful.
- **Agent framework:** Hand-roll a simple ReAct loop in ~150 lines of Python rather than pulling in LangGraph — fewer moving parts, easier to explain in the README, and avoids version-churn issues from heavier frameworks. (Mention LangGraph as a "future work" swap-in.)

---

## 6. Repository Structure

```
trustworthy-slm-agent/
├── README.md                     # project overview, results, how to run
├── docs/
│   ├── SPEC.md                   # this document
│   └── architecture.png
├── requirements.txt
├── configs/
│   └── lora_config.yaml
├── data/
│   ├── raw/                      # source docs (syllabus, notes, textbook)
│   ├── qa_pairs.jsonl            # fine-tuning dataset
│   └── bias_probes.jsonl
├── src/
│   ├── finetune.py                # LoRA fine-tuning script
│   ├── build_index.py             # builds FAISS retrieval index
│   ├── agent.py                   # ReAct agent loop
│   ├── hallucination_check.py     # NLI-based entailment scoring
│   ├── bias_audit.py              # runs bias probes, produces report
│   ├── generate_model_card.py     # renders MODEL_CARD.md
│   └── evaluate.py                # task accuracy, hallucination rate, latency
├── notebooks/
│   └── 01_finetune_colab.ipynb    # the actual Colab-runnable notebook
├── app/
│   └── gradio_app.py              # HF Spaces demo
├── results/
│   ├── eval_metrics.json
│   ├── bias_audit_report.json
│   └── hallucination_log.jsonl
├── MODEL_CARD.md                  # auto-generated output
└── tests/
    └── test_pipeline.py            # smoke tests for each module
```

---

## 7. Execution Plan (Week-by-Week)

Assume ~6-10 hrs/week. Adjust pace as needed — the phases are sequential dependencies, not fixed calendar weeks.

### Phase 0 — Setup (Week 1)
- [ ] Create GitHub repo with the structure above (empty stubs + README skeleton).
- [ ] Set up Colab notebook, confirm free T4 access, install `transformers`, `peft`, `bitsandbytes`, `trl`, `faiss-cpu`, `sentence-transformers`, `gradio`.
- [ ] Pick and download source documents for your domain (e.g., an open OCW course's notes, or a public textbook chapter set).
- [ ] Decide final scope: which course/domain, confirm ~300-500 Q&A pairs is feasible.

### Phase 1 — Data Preparation (Week 1-2)
- [ ] Chunk source documents (e.g., 200-400 token chunks) → build retrieval corpus.
- [ ] Generate/curate Q&A pairs for fine-tuning:
  - Option A: Manually write 300-500 pairs from the source material (slower, higher quality, most defensible for a "responsible AI" project since you control provenance).
  - Option B: Use a free open model (e.g., via Groq free tier or local Qwen) to auto-generate candidate Q&A pairs from the chunks, then manually review/filter — much faster, still transparent since you document the generation+filtering process.
- [ ] Split into train/val/test (e.g., 70/15/15).
- [ ] Write `data/README.md` documenting exact provenance and licensing of source material.

### Phase 2 — Fine-Tuning (Week 2-3)
- [ ] Implement `src/finetune.py`: load base model 4-bit, apply LoRA (rank 8-16), train on Q&A pairs.
- [ ] Run in Colab, save adapter to `results/` and push to HF Hub (free public repo).
- [ ] Log training loss curve, save as artifact.
- [ ] Sanity-check: qualitatively compare base vs. fine-tuned outputs on 5-10 sample questions.

### Phase 3 — Retrieval + Agent (Week 3-4)
- [ ] `src/build_index.py`: embed chunks with `sentence-transformers` (e.g., `all-MiniLM-L6-v2`), build FAISS index.
- [ ] `src/agent.py`: implement ReAct loop —
  1. Receive query
  2. Retrieve top-k chunks
  3. Construct prompt with context + query
  4. Call fine-tuned SLM
  5. Return answer + retrieved sources
- [ ] Test agent end-to-end on 10-15 manual queries.

### Phase 4 — Hallucination Detection (Week 4)
- [ ] `src/hallucination_check.py`: for each (answer, retrieved context) pair, run NLI entailment model, output verdict.
- [ ] Wire into `agent.py` so every response is checked before being returned.
- [ ] Build a small eval set (20-30 Q&A pairs, some intentionally answerable and some deliberately out-of-scope) to measure hallucination rate.

### Phase 5 — Bias Audit (Week 5)
- [ ] Write `data/bias_probes.jsonl` (~30-40 templated prompts across 2 axes).
- [ ] `src/bias_audit.py`: run probes through fine-tuned model, score completions (e.g., sentiment classifier or keyword-based stereotype scoring), aggregate metrics.
- [ ] Generate a plot (bar chart of skew per axis) → save to `results/`.

### Phase 6 — Model Card Generation (Week 5-6)
- [ ] Design a Jinja2 template following the Mitchell et al. / HF Model Card schema (intended use, training data, eval results, ethical considerations, limitations).
- [ ] `src/generate_model_card.py`: reads all JSON logs from `results/`, renders `MODEL_CARD.md`.
- [ ] Manually review and add qualitative "known failure modes" section based on what you observed in Phases 4-5.

### Phase 7 — Evaluation & Benchmarking (Week 6)
- [ ] `src/evaluate.py`: compute task accuracy (base vs fine-tuned), hallucination rate, latency, model size.
- [ ] Produce a results table for the README (this is your "SLM vs LLM" style evidence — even just base vs. fine-tuned is compelling).
- [ ] Optional stretch: compare against a free-tier general LLM API (e.g., Groq's free Llama-3-70B) on the same eval set for a cost/accuracy contrast table.

### Phase 8 — UI & Deployment (Week 7)
- [ ] `app/gradio_app.py`: chat interface showing answer, hallucination flag, sources, and a link to the model card.
- [ ] Deploy to Hugging Face Spaces (free CPU tier is fine for a 1.5B model with 4-bit quantization, or use ZeroGPU if available).
- [ ] Test the live demo end-to-end.

### Phase 9 — Documentation & Polish (Week 7-8)
- [ ] Write final README: problem statement, architecture diagram, setup instructions, results table, live demo link, limitations, future work.
- [ ] Add `tests/test_pipeline.py` — basic smoke tests (does each script run without error on a tiny sample).
- [ ] Clean notebook outputs, add comments.
- [ ] Tag a `v1.0` release on GitHub.
- [ ] Optional: write a short (4-6 page) writeup of the bias audit + hallucination findings as a mini technical report, add to `docs/`.

---

## 8. Success Criteria / Definition of Done

- [ ] Repo is public, cloneable, and `README.md` alone is enough for a stranger to understand and reproduce the project.
- [ ] Fine-tuning is fully reproducible on free Colab T4 in <2 hrs.
- [ ] Live Gradio demo works on Hugging Face Spaces free tier.
- [ ] `MODEL_CARD.md` is auto-generated (not hand-written) from real eval logs.
- [ ] Hallucination rate and bias audit numbers are real, logged, and reported honestly — including negative/limitation findings (this is more credible than a project claiming zero issues).
- [ ] Every claim in the README is backed by a file in `results/`.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Colab free GPU session timeouts mid-training | Checkpoint every N steps; keep LoRA training short (small dataset, few epochs) |
| Bias audit results are noisy/small-sample | Be explicit in README about sample size limitations; treat as illustrative, not definitive |
| Hallucination detector itself is imperfect | Document its known limitations in the model card; don't oversell it as ground truth |
| Scope creep (adding LangGraph, more agents, more tools) | Stick to the hand-rolled ReAct loop for v1; note extensions as "future work" |

---

## 10. Stretch Goals (only after v1.0 is done)

- Swap hand-rolled agent for LangGraph, compare complexity/maintainability.
- Add a second SLM (e.g., Gemma-2-2b) and compare hallucination/bias profiles side by side.
- Add adversarial/red-team prompt set and report jailbreak resistance.
- Package as a pip-installable `trustworthy-slm` toolkit (ties back to the "Responsible AI Toolkit" idea from earlier).
