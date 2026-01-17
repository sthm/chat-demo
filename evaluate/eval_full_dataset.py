# Main evaluation runner for the KB retrieval agent
# Runs offline evaluation against ground truth dataset (dataset.csv)
import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from langsmith import Client, traceable, aevaluate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.docs_graph import docs_agent
from evaluate.evaluators import (
    evaluate_citation_grounding,
    evaluate_citation_presence,
    evaluate_citation_accuracy,
    evaluate_answer_correctness,
)


# Load environment variables
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

# Configuration constants
EXPERIMENT_PREFIX = os.getenv("EVAL_EXPERIMENT_PREFIX", "chatbot-eval")
MAX_CONCURRENCY = 3

def get_dataset_name(mode: str) -> str:
    """Get dataset name based on evaluation mode."""
    if mode == "ground-truth":
        return "chatbot-ground-full"
    else:  # multi-source
        return "chatbot-multi-full"


def _validate_environment() -> None:
    """Validate that all required environment variables are set.

    Raises:
        SystemExit: If any required environment variables are missing
    """
    required_vars = {
        "LANGSMITH_API_KEY": "LangSmith API key for evaluation tracking",
        "ANTHROPIC_API_KEY": "Anthropic API key (or OPENAI_API_KEY)",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if var == "ANTHROPIC_API_KEY":
            # Either Anthropic or OpenAI is fine
            if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
                missing_vars.append(f"  - ANTHROPIC_API_KEY or OPENAI_API_KEY: LLM provider API key")
        elif not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")

    if missing_vars:
        print("Error: Missing required environment variables:\n")
        print("\n".join(missing_vars))
        print("\nPlease set these variables in your .env file.")
        sys.exit(1)


@traceable
async def run_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Target function that runs the KB retrieval agent locally.

    This function is called by LangSmith's evaluate function for each
    example in the dataset. It invokes the local agent and returns
    the response messages.

    Args:
        inputs: Dict with "question" key containing the user's question

    Returns:
        Dict with "messages" key containing the agent's response messages
    """
    question = inputs["question"]
    print(f"\nProcessing: {question[:80]}...")

    # Prepare agent input
    input_data = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    # Invoke agent locally
    result = await asyncio.to_thread(docs_agent.invoke, input_data)

    # Print agent response for observability
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            content = last_message.content
        elif isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = str(last_message)

        print(f"\nAgent Response:")
        print(f"   {content[:500]}..." if len(content) > 500 else f"   {content}")
        print()

    return {"messages": result["messages"]}


def _load_dataset(client: Client, mode: str) -> Any:
    """Load the evaluation dataset from LangSmith.

    Args:
        client: LangSmith client instance
        mode: Evaluation mode (ground-truth or multi-source)

    Returns:
        The loaded dataset

    Raises:
        SystemExit: If dataset cannot be found
    """
    dataset_name = get_dataset_name(mode)

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Found dataset: {dataset_name} ({mode} mode)")
        print(f"   Examples: {dataset.example_count}")
        return dataset

    except Exception as e:
        print(f"Error: Could not find dataset '{dataset_name}'")
        print(f"   Run: python evaluate/dataset_generator.py --mode {mode} --replace")
        print(f"   Error: {e}")
        sys.exit(1)


def _print_evaluation_config(dataset_name: str, mode: str) -> None:
    """Print the evaluation configuration."""
    print("\nStarting offline evaluation...")
    print(f"   Dataset: {dataset_name} ({mode} mode)")
    print(f"   Agent: Local KB retrieval agent")
    print(f"   Evaluators: 4 (citation_grounding, citation_presence, citation_accuracy, answer_correctness)")
    print(f"   Max concurrency: {MAX_CONCURRENCY}")
    print()


def _print_results(results: Any) -> None:
    """Print evaluation results summary.

    Args:
        results: The evaluation results from aevaluate
    """
    print(f"\nEvaluation complete!")
    print(f"   View results: https://smith.langchain.com/")

    # Extract experiment ID from name if available
    if hasattr(results, "experiment_name"):
        experiment_id = results.experiment_name.split("-")[-1]
        print(f"   Experiment ID: {experiment_id}")

    # Print score summary
    print(f"\n📊 Summary:")

    # Results might be an object with properties, not a dict
    if hasattr(results, "__dict__"):
        results_dict = results.__dict__
    else:
        results_dict = {}

    for key, value in results_dict.items():
        if isinstance(value, dict) and "mean" in value:
            score = value["mean"] * 100 if value["mean"] is not None else 0
            count = value.get("count", 0)
            print(f"   {key}: {score:.1f}% (n={count})")


async def main(mode: str = "multi-source", num_examples: int | None = None, replace: bool = False) -> None:
    """Main evaluation function.

    Validates environment, loads dataset, runs evaluation, and prints results.

    Args:
        mode: Evaluation mode (ground-truth or multi-source)
        num_examples: Optional number of examples to run (defaults to all)
        replace: If False and recent experiments exist, prompt before running
    """
    # Validate environment variables
    _validate_environment()

    print(f"✅ Running local KB retrieval agent in {mode} mode...")

    # Initialize LangSmith client
    client = Client()

    # Load dataset
    dataset = _load_dataset(client, mode)
    dataset_name = get_dataset_name(mode)

    # Check for recent experiments if --replace not specified
    if not replace:
        print(f"\n⚠️  Running new evaluation (will create new experiment)")
        print(f"   Use --replace flag to acknowledge you want to create a new experiment run")

    # Get examples from dataset
    if num_examples is not None:
        # Fetch limited number of examples
        examples = list(client.list_examples(dataset_name=dataset_name, limit=num_examples))
        data_input = examples
        print(f"\n   Running on {num_examples} example(s) only")
    else:
        data_input = dataset_name

    # Print configuration
    _print_evaluation_config(dataset_name, mode)

    # Run evaluation with citation evaluators
    results = await aevaluate(
        run_agent,
        data=data_input,
        evaluators=[
            evaluate_citation_grounding,
            evaluate_citation_presence,
            evaluate_citation_accuracy,
            evaluate_answer_correctness,
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=MAX_CONCURRENCY,
    )

    # Print results
    _print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KB agent evaluation")
    parser.add_argument(
        "--mode",
        choices=["ground-truth", "multi-source"],
        default="multi-source",
        help="Evaluation mode: ground-truth (single-source) or multi-source"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Run evaluation on only the first N examples (default: all)"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Acknowledge that you want to create a new evaluation experiment"
    )
    args = parser.parse_args()

    asyncio.run(main(mode=args.mode, num_examples=args.num_examples, replace=args.replace))
