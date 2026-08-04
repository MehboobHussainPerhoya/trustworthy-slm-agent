"""
test_retrieval.py
Quick standalone test of the Retriever only (no model loading, fast).
Run from project root:
    python test_retrieval.py
"""

import sys
sys.path.insert(0, "src")

from agent import Retriever

r = Retriever()
results = r.retrieve("What is the singleton rate?")

for x in results:
    print(f"\n[{x['section']}] (score: {round(x['score'], 3)})")
    print(x["text"][:200])