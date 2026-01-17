# Quick test with a single example from dataset
import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client, traceable, aevaluate
from src.agent.docs_graph import docs_agent
from evaluate.evaluators import (
    evaluate_citation_grounding,
    evaluate_citation_presence,
    evaluate_citation_accuracy,
    evaluate_answer_correctness,
)


@traceable
async def run_agent(inputs: dict) -> dict:
    """Run the KB agent locally with the question input."""
    question = inputs["question"]
    print(f"\nQuestion: {question}")

    input_data = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    # Run agent locally (no thread_id needed for single example)
    result = await asyncio.to_thread(docs_agent.invoke, input_data)

    return {"messages": result["messages"]}


async def main(mode: str = "multi-source", replace: bool = False):
    """Run evaluation on a single example from evaluation dataset.

    Args:
        mode: "ground-truth" or "multi-source"
        replace: If True, delete existing test dataset. If False, skip if exists.
    """

    # Initialize client
    print("Initializing LangSmith client...")
    client = Client()

    # Get source dataset based on mode
    if mode == "ground-truth":
        source_dataset_name = "chatbot-ground-full"
        test_dataset_name = "chatbot-ground-single"
    else:  # multi-source
        source_dataset_name = "chatbot-multi-full"
        test_dataset_name = "chatbot-multi-single"

    try:
        dataset = client.read_dataset(dataset_name=source_dataset_name)
        print(f"Found dataset: {source_dataset_name} ({mode} mode)")
    except Exception as e:
        print(f"❌ Error: Dataset '{source_dataset_name}' not found")
        print(f"   Run: python evaluate/dataset_generator.py --mode {mode} --replace")
        print(f"   Error: {e}")
        return

    # Get just the first example
    examples = list(client.list_examples(dataset_id=dataset.id, limit=1))

    if not examples:
        print("No examples found in dataset")
        return

    example = examples[0]
    print(f"Testing with: {example.inputs.get('question', 'N/A')[:60]}...")
    print(f"\nGround truth:")
    print(f"   Answer: {example.outputs.get('answer', '')[:100]}...")
    print(f"   Has retrieved chunks: {bool(example.outputs.get('retrieved_chunks'))}")
    print(f"   Has cited chunks: {bool(example.outputs.get('cited_chunks'))}")

    # Check if test dataset already exists
    existing_dataset = None
    try:
        existing_dataset = client.read_dataset(dataset_name=test_dataset_name)
    except:
        pass

    # If dataset exists and --replace not specified, skip
    if existing_dataset and not replace:
        print(f"\n✅ Test dataset '{test_dataset_name}' already exists (ID: {existing_dataset.id})")
        print(f"   Previous results are still viewable in LangSmith")
        print(f"   Use --replace flag to delete and rerun")
        return

    # Delete if exists and --replace specified
    if existing_dataset and replace:
        client.delete_dataset(dataset_id=existing_dataset.id)
        print(f"\n🗑️  Deleted previous test dataset")

    # Create new dataset with single example
    test_dataset = client.create_dataset(dataset_name=test_dataset_name)
    client.create_example(
        dataset_id=test_dataset.id,
        inputs=example.inputs,
        outputs=example.outputs
    )

    print(f"\nRunning evaluation...")
    print(f"   Agent: Local KB agent")
    print(f"   Evaluators: citation_grounding, citation_presence, citation_accuracy, answer_correctness")
    print()

    # Run evaluation
    results = await aevaluate(
        run_agent,
        data=test_dataset_name,
        evaluators=[
            evaluate_citation_grounding,
            evaluate_citation_presence,
            evaluate_citation_accuracy,
            evaluate_answer_correctness,
        ],
        experiment_prefix="test-single",
        max_concurrency=1,
    )

    print(f"\n" + "=" * 80)
    print(f"EVALUATION COMPLETE")
    print(f"=" * 80)
    print(f"   Experiment: {results.experiment_name}")
    print(f"   View at: https://smith.langchain.com/")
    print(f"\n💡 Dataset '{test_dataset_name}' persists for viewing in LangSmith")
    print(f"   Use --replace to delete and rerun")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test evaluation with a single example")
    parser.add_argument(
        "--mode",
        choices=["ground-truth", "multi-source"],
        default="multi-source",
        help="Evaluation mode: ground-truth (single-source) or multi-source"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and recreate the test dataset if it already exists"
    )
    args = parser.parse_args()

    asyncio.run(main(mode=args.mode, replace=args.replace))
