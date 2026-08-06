"""
generate_model_card.py
Reads training config + all Phase 2/4/5 result files and renders a
Hugging Face-style MODEL_CARD.md. Re-runnable: run again any time results
change, and the card regenerates from the current state of results/.

Usage (from project root, no GPU needed):
    python src/generate_model_card.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import Template

CONFIG_PATH = Path("configs/lora_config.yaml")
TRAIN_METRICS_PATH = Path("results/eval_metrics.json")
HALLUCINATION_SUMMARY_PATH = Path("results/hallucination_eval_summary.json")
BIAS_REPORT_PATH = Path("results/bias_audit_report.json")
TEST_SET_PATH = Path("data/qa_test.jsonl")
OUTPUT_PATH = Path("MODEL_CARD.md")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

TEMPLATE = """\
# Model Card: Trustworthy SLM Agent (Qwen2.5-1.5B, LoRA fine-tuned)

*Auto-generated on {{ generated_date }}. Regenerate with `python src/generate_model_card.py`.*

## Model Details

- **Base model:** {{ base_model }}
- **Fine-tuning method:** LoRA (rank {{ lora_r }}, alpha {{ lora_alpha }}, dropout {{ lora_dropout }}), applied to: {{ target_modules }}
- **Adapter repository:** {{ adapter_repo }}
- **Developed by:** {{ developer_name }}
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

- **Source:** {{ training_data_source }}
- **Dataset size:** {{ total_qa_pairs }} human-reviewed Q&A pairs
  ({{ train_count }} train / {{ val_count }} val / {{ test_count }} test, 70/15/15 split, seed 42)
- **Provenance:** Q&A pairs were LLM-drafted from source text chunks, then
  manually reviewed and corrected before use. See `data/README.md` for full
  provenance and licensing details.

## Training Procedure

- **Quantization:** 4-bit QLoRA (nf4, bfloat16 compute dtype) during training
- **Epochs:** {{ epochs }}
- **Effective batch size:** {{ effective_batch_size }}
- **Learning rate:** {{ learning_rate }} ({{ lr_scheduler }} schedule)
- **Hardware:** Free-tier Google Colab T4 GPU

### Training Results

| Epoch | Eval Loss | Eval Token Accuracy |
|---|---|---|
{% for epoch, loss in eval_loss_by_epoch.items() -%}
| {{ epoch }} | {{ loss }} | {{ eval_acc_by_epoch.get(epoch, "n/a") }} |
{% endfor %}

- **Final train loss:** {{ final_train_loss }}
- **Trainable parameters:** {{ trainable_params }} / {{ total_params }} ({{ trainable_pct }}%)

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
| In-scope accuracy (verdict = "supported") | {{ in_scope_accuracy }} |
| Out-of-scope appropriate abstention/flagging rate | {{ out_of_scope_abstention_rate }} |
| Overall hallucination rate | {{ overall_hallucination_rate }} |

**Key finding:** the checker correctly flagged a real hallucination on the
model's answer to its single most central question ("why do models
hallucinate"), and correctly flagged 100% of out-of-scope questions rather
than letting fabricated answers pass silently. Manual review also found the
checker to be conservative (some correct answers scored only "partially
supported" due to paraphrase mismatch) and to have missed at least one real
factual conflation. Full detail: `results/hallucination_eval_analysis.md`.

### Bias Audit

Tested on two axes: gender-occupation pronoun association ({{ gender_probe_count }}
occupations) and nationality/group sentiment association ({{ nationality_probe_count }} groups).

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

1. **Small fine-tuning dataset** ({{ total_qa_pairs }} pairs) — sparse,
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
"""


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_safe(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"Warning: {path} not found — {label} section will show placeholders.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_yaml(CONFIG_PATH)
    train_metrics = load_json_safe(TRAIN_METRICS_PATH, "training results")
    hallucination_summary = load_json_safe(HALLUCINATION_SUMMARY_PATH, "hallucination evaluation")
    bias_report = load_json_safe(BIAS_REPORT_PATH, "bias audit")

    context = {
        "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "base_model": cfg["base_model"],
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["alpha"],
        "lora_dropout": cfg["lora"]["dropout"],
        "target_modules": ", ".join(cfg["lora"]["target_modules"]),
        "adapter_repo": cfg["hub"]["repo_id"],
        "developer_name": "[Your Name]",

        "training_data_source": (
            "Kalai, Nachum, Vempala, and Zhang (2025), \"Why Language Models "
            "Hallucinate,\" arXiv:2509.04664"
        ),
        "total_qa_pairs": train_metrics.get("training_examples", 0) + train_metrics.get("val_examples", 0) + count_jsonl_lines(TEST_SET_PATH),
        "train_count": train_metrics.get("training_examples", "n/a"),
        "val_count": train_metrics.get("val_examples", "n/a"),
        "test_count": count_jsonl_lines(TEST_SET_PATH) or "n/a",

        "epochs": cfg["training"]["num_train_epochs"],
        "effective_batch_size": cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"],
        "learning_rate": cfg["training"]["learning_rate"],
        "lr_scheduler": cfg["training"]["lr_scheduler_type"],

        "eval_loss_by_epoch": train_metrics.get("eval_loss_by_epoch", {}),
        "eval_acc_by_epoch": train_metrics.get("eval_token_accuracy_by_epoch", {}),
        "final_train_loss": train_metrics.get("final_train_loss", "n/a"),
        "trainable_params": f"{train_metrics.get('trainable_params', 0):,}" if train_metrics.get("trainable_params") else "n/a",
        "total_params": f"{train_metrics.get('total_params', 0):,}" if train_metrics.get("total_params") else "n/a",
        "trainable_pct": train_metrics.get("trainable_param_pct", "n/a"),

        "in_scope_accuracy": f"{hallucination_summary.get('in_scope', {}).get('accuracy', 'n/a')}",
        "out_of_scope_abstention_rate": f"{hallucination_summary.get('out_of_scope', {}).get('appropriate_abstention_rate', 'n/a')}",
        "overall_hallucination_rate": f"{hallucination_summary.get('overall_hallucination_rate', 'n/a')}",

        "gender_probe_count": len(bias_report.get("gender_occupation", {}).get("results", [])) or 20,
        "nationality_probe_count": len(bias_report.get("nationality_sentiment", {}).get("results", [])) or 18,
    }

    template = Template(TEMPLATE)
    rendered = template.render(**context)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Model card written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()