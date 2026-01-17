#!/usr/bin/env python3
"""Create evaluation datasets with generated questions for evaluation.

This script supports two modes:
1. Ground Truth Mode: Single-source tests from dataset.csv
2. Multi-Source Mode: Multiple-source tests from synthetic_dataset.csv

Output: data/eval_dataset_ground_truth.csv or data/eval_dataset_multi_source.csv
"""

import csv
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import tracing_context

# Load environment
load_dotenv()

# Initialize LLM for question generation
question_generator = init_chat_model("gpt-4o-mini", temperature=0.7)


def generate_question_for_topic(topic: str, retrieved_chunks: str) -> str:
    """Generate a natural customer question for a given topic.

    Args:
        topic: The topic/category (e.g., "account closure")
        retrieved_chunks: The procedural text to understand context

    Returns:
        A natural customer question
    """
    # Take first 500 chars of chunks for context
    context = retrieved_chunks[:500]

    prompt = f"""You are generating realistic customer service questions for a banking/credit card support KB.

Topic: {topic}

Context from KB (first 500 chars):
{context}

Generate ONE natural question that a customer would ask about this topic.
The question should be:
- Conversational and realistic (how a real customer would ask)
- 10-20 words long
- Specific enough to be useful
- NOT using technical jargon

Examples of good questions:
- "How do I close my credit card account?"
- "What happens if I miss my payment due date?"
- "Can I get a refund for the annual fee?"

Generate ONLY the question, no explanation or other text."""

    with tracing_context(project_name="synthetic-generation"):
        response = question_generator.invoke([{"role": "user", "content": prompt}])

    return response.content.strip()


def create_ground_truth_dataset(max_examples: int = 10):
    """Create single-source eval dataset from dataset.csv (ground truth).

    Tests if agent correctly cites the specific ground truth source.
    Samples diverse rows (different chunks) for each topic to get variety.
    """
    base_path = Path(__file__).parent.parent
    input_path = base_path / "data" / "dataset.csv"
    output_path = base_path / "data" / "eval_dataset_ground_truth.csv"

    print(f"\n📚 GROUND TRUTH MODE")
    print(f"   Reading from: {input_path}")

    # Read dataset.csv and group by topic
    topic_rows = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row["question"]
            if topic not in topic_rows:
                topic_rows[topic] = []
            topic_rows[topic].append(row)

    print(f"   Found {len(topic_rows)} unique topics")
    for topic, rows in topic_rows.items():
        print(f"      • {topic}: {len(rows)} rows")

    # Sample diverse rows from each topic to reach max_examples
    import random
    random.seed(42)  # For reproducibility

    sampled_rows = []
    topics = list(topic_rows.keys())
    rows_per_topic = max_examples // len(topics)
    remaining = max_examples % len(topics)

    for i, topic in enumerate(topics):
        # Take more from first topic if we have remainder
        n_samples = rows_per_topic + (1 if i < remaining else 0)
        n_samples = min(n_samples, len(topic_rows[topic]))

        # Sample diverse rows (evenly spaced through the list)
        topic_samples = topic_rows[topic]
        step = len(topic_samples) // n_samples if n_samples > 0 else 1
        selected = [topic_samples[j * step] for j in range(n_samples)]

        sampled_rows.extend(selected)
        print(f"   Sampled {len(selected)} diverse rows from '{topic}'")

    print(f"\n   Total sampled: {len(sampled_rows)} rows")

    # Generate questions for each sampled row
    print(f"\n🤖 Generating natural questions for {len(sampled_rows)} chunks...")

    eval_examples = []
    for i, row in enumerate(sampled_rows, 1):
        topic = row["question"]
        retrieved_chunks = row["retrieved_chunks"]

        print(f"   [{i}/{len(sampled_rows)}] {topic} (chunk {i})...")

        question = generate_question_for_topic(topic, retrieved_chunks)

        # Single source attribution
        attributed_retrieved = f"[Source: {topic}]\n{retrieved_chunks}"
        attributed_cited = f"[Source: {topic}]\n{row['cited_chunks']}"

        eval_examples.append({
            "question": question,
            "retrieved_chunks": attributed_retrieved,
            "answer": row["answer"],
            "cited_chunks": attributed_cited,
            "mode": "ground_truth"
        })

        print(f"      → {question}")

    # Write output
    print(f"\n💾 Writing to: {output_path}")

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "retrieved_chunks", "answer", "cited_chunks", "mode"])
        writer.writeheader()
        writer.writerows(eval_examples)

    print(f"   ✅ Wrote {len(eval_examples)} ground truth examples")
    return output_path


def create_multi_source_dataset(max_examples: int = 10):
    """Create multi-source eval dataset from synthetic_dataset.csv.

    Simulates realistic KB retrieval with multiple sources.
    Samples diverse rows (different chunks) for each topic to get variety.
    """
    base_path = Path(__file__).parent.parent
    input_path = base_path / "data" / "synthetic_dataset.csv"
    output_path = base_path / "data" / "eval_dataset_multi_source.csv"

    print(f"\n📚 MULTI-SOURCE MODE")
    print(f"   Reading from: {input_path}")

    # Read synthetic_dataset.csv and group by topic
    topic_rows = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row["question"]
            if topic not in topic_rows:
                topic_rows[topic] = []
            topic_rows[topic].append(row)

    print(f"   Found {len(topic_rows)} unique topics")
    total_rows = sum(len(rows) for rows in topic_rows.values())
    print(f"   Total rows: {total_rows}")

    # Sample diverse topics (not just first 15)
    import random
    random.seed(42)  # For reproducibility

    topics = list(topic_rows.keys())

    # If we need fewer topics than available, sample randomly
    if len(topics) > max_examples:
        print(f"   Sampling {max_examples} random topics from {len(topics)} available")
        selected_topics = random.sample(topics, max_examples)
    else:
        selected_topics = topics

    # For each selected topic, pick one diverse row
    sampled_rows = []
    for topic in selected_topics:
        topic_samples = topic_rows[topic]
        # Pick a random row from this topic (or first if only one)
        selected_row = random.choice(topic_samples) if len(topic_samples) > 1 else topic_samples[0]
        sampled_rows.append(selected_row)

    print(f"   Sampled {len(sampled_rows)} diverse rows")

    # Generate questions for each sampled row
    print(f"\n🤖 Generating natural questions for {len(sampled_rows)} chunks...")

    eval_examples = []
    for i, row in enumerate(sampled_rows, 1):
        topic = row["question"]
        retrieved_chunks = row["retrieved_chunks"]

        print(f"   [{i}/{len(sampled_rows)}] {topic[:50]}...")

        question = generate_question_for_topic(topic, retrieved_chunks)

        # Single source attribution
        attributed_retrieved = f"[Source: {topic}]\n{retrieved_chunks}"
        attributed_cited = f"[Source: {topic}]\n{row['cited_chunks']}"

        eval_examples.append({
            "question": question,
            "retrieved_chunks": attributed_retrieved,
            "answer": row["answer"],
            "cited_chunks": attributed_cited,
            "mode": "multi_source"
        })

        print(f"      → {question}")

    # Write output
    print(f"\n💾 Writing to: {output_path}")

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "retrieved_chunks", "answer", "cited_chunks", "mode"])
        writer.writeheader()
        writer.writerows(eval_examples)

    print(f"   ✅ Wrote {len(eval_examples)} multi-source examples")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation datasets")
    parser.add_argument(
        "--mode",
        choices=["ground-truth", "multi-source", "both"],
        default="both",
        help="Evaluation mode: ground-truth (single-source from dataset.csv), multi-source (from synthetic_dataset.csv), or both"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of examples to generate (default: 10)"
    )
    args = parser.parse_args()

    print(f"=" * 80)
    print(f"EVALUATION DATASET GENERATOR")
    print(f"=" * 80)

    output_paths = []

    if args.mode in ["ground-truth", "both"]:
        path = create_ground_truth_dataset(max_examples=args.max_examples)
        output_paths.append(("Ground Truth", path))

    if args.mode in ["multi-source", "both"]:
        path = create_multi_source_dataset(max_examples=args.max_examples)
        output_paths.append(("Multi-Source", path))

    print(f"\n" + "=" * 80)
    print(f"GENERATION COMPLETE")
    print(f"=" * 80)
    for name, path in output_paths:
        print(f"   {name}: {path}")

    print(f"\nNext step:")
    print(f"   Upload to LangSmith: python evaluate/dataset_generator.py --mode [ground-truth|multi-source] --replace")
