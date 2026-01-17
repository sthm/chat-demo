#!/usr/bin/env python3
"""Upload evaluation dataset to LangSmith for offline evaluation.

Supports two modes:
1. Ground Truth Mode: Single-source tests from dataset.csv
2. Multi-Source Mode: Multiple-source tests from synthetic_dataset.csv

Usage:
    python evaluate/dataset_generator.py --mode [ground-truth|multi-source] [--replace]

    --mode: Which dataset to upload (ground-truth or multi-source)
    --replace: Delete and recreate the dataset if it already exists
"""

import os
import csv
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)


def upload_dataset_from_csv(mode: str = "multi-source", replace: bool = False):
    """Upload evaluation dataset to LangSmith for offline evaluation.

    Args:
        mode: "ground-truth" or "multi-source"
        replace: If True, delete and recreate existing dataset. If False, skip if exists.
    """

    client = Client()

    # Configuration based on mode
    if mode == "ground-truth":
        dataset_name = "chatbot-ground-full"
        dataset_path = Path(__file__).parent.parent / "data" / "eval_dataset_ground_truth.csv"
        description = "Ground truth evaluation dataset (single-source) from dataset.csv"
    else:  # multi-source
        dataset_name = "chatbot-multi-full"
        dataset_path = Path(__file__).parent.parent / "data" / "eval_dataset_multi_source.csv"
        description = "Multi-source evaluation dataset from synthetic_dataset.csv"

    # Verify dataset file exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print(f"   Run: python scripts/create_langsmith_dataset.py --mode {mode}")
        exit(1)

    # Check if dataset already exists
    existing_dataset = None
    try:
        existing_dataset = client.read_dataset(dataset_name=dataset_name)
    except:
        pass

    # If dataset exists and --replace not specified, skip upload
    if existing_dataset and not replace:
        print(f"✅ Dataset '{dataset_name}' already exists (ID: {existing_dataset.id})")
        print(f"   Examples: {existing_dataset.example_count}")
        print(f"   View at: https://smith.langchain.com/datasets/{existing_dataset.id}")
        print(f"\n   Use --replace flag to delete and recreate")
        return existing_dataset

    print(f"📚 Uploading dataset from: {dataset_path}")

    # Read CSV
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                "inputs": {
                    "question": row["question"]
                },
                "outputs": {
                    "retrieved_chunks": row["retrieved_chunks"],
                    "answer": row["answer"],
                    "cited_chunks": row["cited_chunks"],
                }
            })

    print(f"   Loaded {len(examples)} examples from CSV")

    # Delete existing dataset if --replace specified
    if existing_dataset:
        print(f"   Found existing dataset '{dataset_name}', deleting...")
        client.delete_dataset(dataset_id=existing_dataset.id)

    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description
    )
    print(f"   Created new dataset: {dataset_name}")

    # Upload examples
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )

    print(f"   ✅ Uploaded {len(examples)} examples to LangSmith")
    print(f"\n   Dataset ID: {dataset.id}")
    print(f"   View at: https://smith.langchain.com/datasets/{dataset.id}")
    print(f"\nNext step:")
    print(f"   Run evaluation with: python evaluate/eval_full_dataset.py --replace")

    return dataset


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Upload evaluation dataset to LangSmith")
    parser.add_argument(
        "--mode",
        choices=["ground-truth", "multi-source"],
        default="multi-source",
        help="Evaluation mode: ground-truth (single-source) or multi-source"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and recreate the dataset if it already exists"
    )
    args = parser.parse_args()

    # Verify LangSmith env vars are set
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ Error: LANGSMITH_API_KEY not set")
        print("   Set it with: export LANGSMITH_API_KEY=your_key")
        exit(1)

    print(f"📤 Uploading {args.mode} dataset to LangSmith...")
    upload_dataset_from_csv(mode=args.mode, replace=args.replace)
