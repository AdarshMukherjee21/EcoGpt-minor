"""Smoke-test the local MiniLM matrix-factorization router.

Constructs the MF router from a locally-trained checkpoint (e.g. the one in
colab_outputs/checkpoint) and prints strong-model win rates / routing
decisions for a few sample prompts. Embeddings are computed locally via
sentence-transformers — no OpenAI API key needed.

Usage (from the project root):
    python routellm/routers/matrix_factorization/check_local_router.py
    python routellm/routers/matrix_factorization/check_local_router.py \
        --checkpoint colab_outputs/checkpoint \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from routellm.routers.routers import MatrixFactorizationRouter

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that merges two sorted lists in O(n) time.",
    "Explain the difference between UDP and TCP, with examples.",
    "Derive the quadratic formula step by step.",
    "List 10 baby names starting with letter A.",
    "What are the main causes of inflation in an economy?",
    "Write a bash one-liner to find all .log files larger than 100MB.",
    "Summarize the plot of Hamlet in three sentences.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="colab_outputs/checkpoint",
        help="Local dir/file containing the trained MF weights.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local SentenceTransformer used for prompt embeddings.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Routing threshold (winrate >= threshold -> strong).",
    )
    args = parser.parse_args()

    router = MatrixFactorizationRouter(
        checkpoint_path=args.checkpoint,
        embedding_provider="sentence-transformers",
        embedding_model=args.embedding_model,
    )

    routed_pair = SimpleNamespace(
        strong="gpt-4-1106-preview", weak="mixtral-8x7b-instruct-v0.1"
    )

    print(f"\n{'winrate':>7}  {'route':>6}  prompt")
    print("-" * 72)
    for prompt in SAMPLE_PROMPTS:
        winrate = router.calculate_strong_win_rate(prompt)
        routed = router.route(prompt, args.threshold, routed_pair)
        label = "strong" if routed == routed_pair.strong else "weak"
        print(f"{winrate:7.3f}  {label:>6}  {prompt}")

    print(f"\nSample prompts routed at threshold {args.threshold}")


if __name__ == "__main__":
    main()
