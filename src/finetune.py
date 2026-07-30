"""
finetune.py
LoRA/QLoRA fine-tuning of a small instruct model on the qa_train.jsonl /
qa_val.jsonl datasets. Designed to run on a free Colab T4 GPU.

Usage (from project root):
    python src/finetune.py --config configs/lora_config.yaml

Requires (installed separately, typically in Colab):
    transformers, peft, bitsandbytes, trl, accelerate, datasets, pyyaml
"""

import argparse
import inspect
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_formatting_func(tokenizer, system_prompt: str):
    """Returns a function that turns a {question, answer} record into a
    single training string using the model's chat template."""

    def format_example(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return format_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading base model: {cfg['base_model']}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_dataset("json", data_files=cfg["paths"]["train_data"], split="train")
    val_ds = load_dataset("json", data_files=cfg["paths"]["val_data"], split="train")
    print(f"  train: {len(train_ds)} examples")
    print(f"  val:   {len(val_ds)} examples")

    formatting_func = build_formatting_func(tokenizer, cfg["system_prompt"])

    # trl renamed SFTConfig's sequence-length argument from `max_seq_length`
    # to `max_length` starting in v0.16.0. Detect which one this installed
    # version actually accepts so the script keeps working across versions.
    sftconfig_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sftconfig_params:
        seq_len_kwarg = {"max_length": cfg["training"]["max_seq_length"]}
    elif "max_seq_length" in sftconfig_params:
        seq_len_kwarg = {"max_seq_length": cfg["training"]["max_seq_length"]}
    else:
        print("Warning: neither 'max_length' nor 'max_seq_length' found in "
              "SFTConfig signature for this trl version; skipping this arg.")
        seq_len_kwarg = {}

    # transformers renamed `evaluation_strategy` -> `eval_strategy` around v4.46.
    if "eval_strategy" in sftconfig_params:
        eval_strategy_kwarg = {"eval_strategy": cfg["training"]["eval_strategy"]}
    elif "evaluation_strategy" in sftconfig_params:
        eval_strategy_kwarg = {"evaluation_strategy": cfg["training"]["eval_strategy"]}
    else:
        eval_strategy_kwarg = {}

    training_args = SFTConfig(
        output_dir=cfg["paths"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        seed=cfg["training"]["seed"],
        bf16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=cfg["hub"]["push_to_hub"],
        hub_model_id=cfg["hub"]["repo_id"] if cfg["hub"]["push_to_hub"] else None,
        **seq_len_kwarg,
        **eval_strategy_kwarg,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {cfg['paths']['output_dir']}")
    trainer.save_model(cfg["paths"]["output_dir"])
    tokenizer.save_pretrained(cfg["paths"]["output_dir"])

    if cfg["hub"]["push_to_hub"]:
        print(f"Pushing adapter to Hugging Face Hub: {cfg['hub']['repo_id']}")
        trainer.push_to_hub()

    print("Done.")


if __name__ == "__main__":
    main()