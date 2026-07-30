"""
split_dataset.py
Splits data/qa_pairs.jsonl into train/val/test (70/15/15) with a fixed
random seed for reproducibility.

Usage:
    python src/split_dataset.py
"""

import json
import random
from pathlib import Path

SRC = Path("data/qa_pairs.jsonl")
OUT_DIR = Path("data")
SEED = 42
SPLIT = (0.70, 0.15, 0.15)  # train, val, test


def main():
    random.seed(SEED)
    lines = SRC.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(l) for l in lines]
    random.shuffle(records)

    n = len(records)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        out_path = OUT_DIR / f"qa_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} examples -> {out_path}")


if __name__ == "__main__":
    main()