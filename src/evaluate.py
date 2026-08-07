"""
evaluate.py
Runs the held-out test set (data/qa_test.jsonl) through both the base model
and the fine-tuned model (via LoRA adapter disable/enable on the same
loaded model), computing:
  - Token-level F1 accuracy against the reference answer (base vs fine-tuned)
  - Generation latency (base vs fine-tuned)
  - Model size (base model params vs adapter size on disk)

Usage (from project root, needs GPU realistically — run in Colab):
    python src/evaluate.py
"""

import json
import re
import string
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "Mehboobali512/trustworthy-slm-agent-qwen2.5-1.5b"
TEST_SET_PATH = Path("data/qa_test.jsonl")
RESULTS_PATH = Path("results/evaluate_results.jsonl")
SUMMARY_PATH = Path("results/evaluate_summary.json")

SYSTEM_PROMPT = (
    "You are a helpful assistant with expert knowledge of the paper "
    "'Why Language Models Hallucinate' by Kalai, Nachum, Vempala, and Zhang (2025). "
    "Answer questions accurately based on the paper's content."
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
    else:
        print("No GPU detected — loading in full precision on CPU (slower).")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
        )

    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip(), elapsed


def load_test_set():
    items = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def get_adapter_size_mb() -> float:
    """Best-effort: check local adapter folder if present, else report from
    known training run (fallback)."""
    local_adapter = Path("results/lora_adapter/adapter_model.safetensors")
    if local_adapter.exists():
        return round(local_adapter.stat().st_size / (1024 * 1024), 2)
    return None


def main():
    test_items = load_test_set()
    print(f"Loaded {len(test_items)} test questions")

    print("Loading model...")
    tokenizer, model = load_model()

    results = []
    for item in test_items:
        question = item["question"]
        reference = item["answer"]
        print(f"\nQ: {question}")

        with model.disable_adapter():
            base_answer, base_latency = generate_answer(tokenizer, model, question)
        finetuned_answer, ft_latency = generate_answer(tokenizer, model, question)

        base_f1 = token_f1(base_answer, reference)
        ft_f1 = token_f1(finetuned_answer, reference)

        record = {
            "question": question,
            "reference": reference,
            "base_answer": base_answer,
            "finetuned_answer": finetuned_answer,
            "base_f1": round(base_f1, 3),
            "finetuned_f1": round(ft_f1, 3),
            "base_latency_sec": round(base_latency, 2),
            "finetuned_latency_sec": round(ft_latency, 2),
        }
        results.append(record)
        print(f"  base_f1={base_f1:.3f}  finetuned_f1={ft_f1:.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved detailed results to {RESULTS_PATH}")

    mean_base_f1 = sum(r["base_f1"] for r in results) / len(results)
    mean_ft_f1 = sum(r["finetuned_f1"] for r in results) / len(results)
    mean_base_latency = sum(r["base_latency_sec"] for r in results) / len(results)
    mean_ft_latency = sum(r["finetuned_latency_sec"] for r in results) / len(results)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "test_set_size": len(results),
        "mean_token_f1": {
            "base_model": round(mean_base_f1, 3),
            "finetuned_model": round(mean_ft_f1, 3),
            "improvement": round(mean_ft_f1 - mean_base_f1, 3),
        },
        "mean_latency_seconds": {
            "base_model": round(mean_base_latency, 2),
            "finetuned_model": round(mean_ft_latency, 2),
        },
        "model_size": {
            "base_model_total_params": total_params,
            "adapter_size_mb": get_adapter_size_mb(),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()