"""
agent.py
A retrieval-grounded agent: given a user question, retrieves the most
relevant chunks from the paper, injects them into the model's prompt, and
generates an answer. Works with or without a GPU (uses 4-bit quantization
if CUDA is available, falls back to plain CPU inference otherwise).

Usage (from project root):
    python src/agent.py                       # interactive CLI
    python src/agent.py --query "..."          # single query, then exit
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

INDEX_DIR = Path("data/index")
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "Mehboobali512/trustworthy-slm-agent-qwen2.5-1.5b"  # your HF adapter

SYSTEM_PROMPT = (
    "You are a helpful assistant with expert knowledge of the paper "
    "'Why Language Models Hallucinate' by Kalai, Nachum, Vempala, and Zhang (2025). "
    "Answer the user's question using ONLY the provided context excerpts from the "
    "paper. If the context does not contain enough information to answer "
    "confidently, say so honestly rather than guessing."
)

TOP_K = 3


class Retriever:
    """Loads the FAISS index and chunk metadata, retrieves top-k relevant chunks."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index = faiss.read_index(str(index_dir / "chunks.faiss"))
        with open(index_dir / "chunk_metadata.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(index_dir / "embedding_model.txt", "r") as f:
            embedding_model_name = f.read().strip()
        self.embedder = SentenceTransformer(embedding_model_name)

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        query_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results


class Generator:
    """Loads the fine-tuned model (base + LoRA adapter) and generates answers."""

    def __init__(self, adapter_repo: str = ADAPTER_REPO):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
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

        self.model = PeftModel.from_pretrained(base_model, adapter_repo)
        self.model.eval()

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        context_text = "\n\n".join(
            f"[Excerpt from {c['section']}]\n{c['text']}" for c in context_chunks
        )
        user_message = f"Context:\n{context_text}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        answer = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return answer.strip()


class Agent:
    """Ties retrieval and generation together into a single query interface."""

    def __init__(self):
        print("Loading retriever...")
        self.retriever = Retriever()
        print("Loading generator (this may take a minute)...")
        self.generator = Generator()
        print("Agent ready.\n")

    def ask(self, question: str, k: int = TOP_K) -> dict:
        retrieved = self.retriever.retrieve(question, k=k)
        answer = self.generator.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"section": c["section"], "score": round(c["score"], 3), "text": c["text"][:200] + "..."}
                for c in retrieved
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Single query, then exit")
    args = parser.parse_args()

    agent = Agent()

    if args.query:
        result = agent.ask(args.query)
        print_result(result)
        return

    print("Interactive mode. Type a question, or 'quit' to exit.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        result = agent.ask(question)
        print_result(result)


def print_result(result: dict):
    print(f"\nA: {result['answer']}\n")
    print("Sources retrieved:")
    for s in result["sources"]:
        print(f"  - [{s['section']}] (score: {s['score']}) {s['text']}")
    print()


if __name__ == "__main__":
    main()
