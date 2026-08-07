# Trustworthy SLM Agent

This project fine-tunes a small language model on a single research paper,
then wraps it with retrieval, automated hallucination checking, and a bias
audit — basically everything I could realistically build on free-tier
compute to answer one question: fine-tuning aside, can I actually trust
what this model says, and how would I know if I couldn't?

The paper I picked is *"Why Language Models Hallucinate"* by Kalai,
Nachum, Vempala, and Zhang (2025, arXiv:2509.04664). Using that specific
paper felt right for this project — the agent is essentially being tested
against the exact theory it was trained to explain, so when it messes up,
the mistake is directly readable against the paper's own framework.

## What it actually does

You ask it a question about the paper. It searches a small FAISS index
built from the paper's text, pulls the most relevant paragraphs, and feeds
those to a LoRA-fine-tuned Qwen2.5-1.5B model along with your question. The
answer that comes back then gets checked against those same retrieved
paragraphs using an NLI model — basically asking "does this answer
actually follow from what we just gave it, or did it wander off and make
something up?" If the answer doesn't hold up, it gets withheld and the app
tells you so, instead of quietly showing you something that sounds
confident but isn't backed by anything.

```
question
   │
   ▼
retrieve relevant paragraphs (FAISS + MiniLM embeddings)
   │
   ▼
generate answer (fine-tuned Qwen2.5-1.5B + LoRA adapter)
   │
   ▼
check answer against retrieved paragraphs (DeBERTa NLI model)
   │
   ▼
supported → show it
partially supported → show it, but flagged
unsupported → don't show it, explain why not
```

## Repo layout

```
app/gradio_app.py       the web UI
configs/lora_config.yaml LoRA + training settings
data/                     source text, chunks, Q&A pairs, the FAISS index
docs/SPEC.md              the original planning doc for this project
notebooks/                Colab notebooks used for training
results/                  every eval output and my write-ups on what they mean
src/                      all the actual pipeline scripts
tests/test_pipeline.py    smoke tests
MODEL_CARD.md             auto-generated model card
```

## Getting it running

```powershell
git clone https://github.com/MehboobHussainPerhoya/trustworthy-slm-agent.git
cd trustworthy-slm-agent
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Fair warning: the heavy stuff (fine-tuning, running the agent, the eval
scripts) needs a GPU, and my machine doesn't have one, so all of that runs
in Colab on the free T4 tier. The local setup here is mostly for editing
code and the lightweight, CPU-only steps like chunking the source text and
building the retrieval index. `docs/SPEC.md` has the full breakdown of
what runs where.

## How the pieces fit together, in order

1. `src/extract_text.py` — pulls text out of the source PDF
2. `src/chunk_document.py` — splits it into 55 labeled sections
3. `src/split_dataset.py` — 70/15/15 split of the Q&A dataset
4. `src/finetune.py` — LoRA fine-tunes Qwen2.5-1.5B-Instruct
5. `src/build_index.py` — builds the FAISS retrieval index
6. `src/agent.py` — the actual retrieval + generation + checking agent
7. `src/evaluate_hallucination.py` — runs the hallucination eval set
8. `src/bias_audit.py` — the gender/nationality bias probes
9. `src/evaluate.py` — base vs. fine-tuned comparison on held-out data
10. `src/generate_model_card.py` — builds `MODEL_CARD.md` from all the results
11. `app/gradio_app.py` — the demo itself

## Results

I'll just go through what I actually found, including the parts that
weren't clean wins.

**Fine-tuning.** Trained on 116 examples for 3 epochs. Eval loss dropped
from 1.53 to 1.23 across the three epochs, and token accuracy went from
71% to about 76.5%, mostly in the first two epochs — by epoch 3 it had
basically leveled off. Only about 1.2% of the model's parameters were
actually being trained (LoRA on a 1.5B model), which is the whole point of
doing it this way on free compute.

**Did fine-tuning actually help?** I checked this properly on the 26
held-out test questions the model never saw during training, scoring
answers against the reference answers with token F1. Base model averaged
0.135, fine-tuned averaged 0.185 — about a 38% relative improvement.
Fine-tuned answers were also faster to generate (6.9s vs 10.3s on
average), probably because they're more concise. Not every question got
better — both RAG-related questions actually scored worse after
fine-tuning — but the overall trend was a real, measurable improvement,
not just something I'm eyeballing from a couple of examples.

**Does the hallucination checker actually catch anything?** This is the
part I was most curious about. I built a 20-question eval set — 10
answerable from the paper, 10 deliberately not (things like "what's the
capital of France," which the agent has no business answering
confidently). The out-of-scope questions were caught 100% of the time —
every single one either got refused outright or flagged as not backed by
the retrieved sources. That's the result I care about most, since it's
the actual safety property this whole project is trying to demonstrate.

The in-scope accuracy was lower, 60%, which sounds worse than it is once
you actually read the failures. I went through all of them by hand: two
of the "not fully supported" verdicts turned out to be correct answers
that the checker was just being overly strict about (it didn't like that
the wording didn't match the source text closely enough, even though the
meaning was right). One was a genuine mix-up where the model conflated
two different examples from the paper and probably should have been
flagged more strongly than it was. And separately, the checker correctly
caught a real hallucination in the model's answer to its own most
important question — "why do language models hallucinate" — where it
claimed something about calibration that actually contradicts what the
paper says. That one felt like a good sign the system works, even though
the raw accuracy number looks middling.

**Bias audit.** I tested two things: whether the model associates certain
occupations with a gender, and whether it describes different
nationalities with different sentiment. For occupations, the answer is
yes, pretty clearly — 19 out of 20 occupations I tested showed a skew in
the stereotype-consistent direction (mechanic skewed hard toward "he,"
babysitter hard toward "she," and so on), with two interesting exceptions:
hairdresser actually skewed toward "he," and flight attendant came out
exactly neutral despite a pretty strong historical stereotype. For
nationality, I didn't find much difference in sentiment across the groups
I tested, but I don't think that means there's no bias there — the
completions were pretty formulaic across the board ("very polite," "strong
sense of family," that kind of thing), which makes me think the sentiment
score just isn't sensitive enough to pick up on subtler stuff. I wrote
this up honestly rather than calling it a clean pass.

Full details and the actual plot are in `results/bias_audit_analysis.md`
and `results/bias_audit_plot.png`.

## Live demo

The agent is deployed as a Gradio app that gates its own answers by the
hallucination verdict — if the checker says an answer isn't backed by the
retrieved sources, the app doesn't show it, it explains why it's holding
back instead. I tested this live with a couple of questions clearly
outside the paper's scope and it worked as intended.

Right now the demo runs on a temporary Colab-hosted link rather than a
permanent one, because Hugging Face Spaces now requires either a paid plan
or meeting their free ZeroGPU eligibility bar, and getting that sorted out
turned into more of a detour than I expected. I'll move it to a permanent
Spaces deployment once that's resolved. In the meantime the app code is
in `app/gradio_app.py` and runs fine from Colab if you want to spin it up
yourself.

## What I'd still change

- The Q&A dataset is small (166 pairs after I caught and removed a
  duplicate a smoke test flagged). Facts that only showed up once or
  twice in training aren't reliably learned — retrieval grounding helps
  with this but doesn't fully fix it.
- The hallucination checker isn't a source of ground truth. It's
  conservative in a way that sometimes marks down correct answers, and it
  missed at least one real mistake in my manual review. Useful signal,
  not a verdict to trust blindly.
- There's real, measurable gender bias in how the model talks about
  occupations. That's coming from the base model's pretraining, not
  something this project's fine-tuning introduced, but it's still there
  and worth being upfront about.
- The nationality/sentiment bias check probably isn't sensitive enough as
  designed. A better version would look at what's actually being said
  about each group, not just whether it scores positive or negative.
- This only knows about one paper. Asking it about anything else is
  exactly what the out-of-scope testing is meant to catch, and it does,
  but it's worth saying plainly: this isn't a general-purpose assistant.

## Model card

The full auto-generated model card is in [`MODEL_CARD.md`](MODEL_CARD.md).
It gets regenerated from the actual result files any time I rerun
`src/generate_model_card.py`, so it stays in sync with whatever the
current numbers actually are rather than going stale.

## Data and licensing

The source paper is CC BY 4.0 (Kalai et al., 2025, arXiv:2509.04664), so
using it here is fine — I've documented exactly how it was processed and
what's derived from it in `data/README.md`. The PDF itself isn't checked
into this repo; only the extracted text and everything built from it are.