# KB Agent Demo - Threads & Insights

Demo showing LangSmith threads and insights using a customer support KB agent.

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY and LANGSMITH_API_KEY to .env
```

## Generate Test Data

```bash
# Generate 200 single-turn traces
uv run scripts/generate_traces.py

# Generate 20 multi-turn conversation threads
uv run scripts/generate_threads.py

# Generate question bank (if needed)
uv run scripts/generate_question_bank.py
```

## Run Evaluation

### Generate Evaluation Datasets

```bash
# Generate both ground-truth and multi-source datasets
uv run scripts/create_langsmith_dataset.py --mode both --max-examples 10

# Or generate specific mode only
uv run scripts/create_langsmith_dataset.py --mode ground-truth --max-examples 5
uv run scripts/create_langsmith_dataset.py --mode multi-source --max-examples 10
```

### Upload to LangSmith

```bash
# Upload ground-truth dataset (single-source tests from dataset.csv)
uv run evaluate/dataset_generator.py --mode ground-truth --replace

# Upload multi-source dataset (realistic multi-source tests from synthetic_dataset.csv)
uv run evaluate/dataset_generator.py --mode multi-source --replace
```

### Run Evaluations

```bash
# Test with single example
uv run evaluate/eval_single_example.py --mode ground-truth --replace
uv run evaluate/eval_single_example.py --mode multi-source --replace

# Run full dataset evaluation
uv run evaluate/eval_full_dataset.py --mode ground-truth --replace
uv run evaluate/eval_full_dataset.py --mode multi-source --replace

# Note: --replace flag creates new evaluation experiments
# Without --replace, scripts will skip if results already exist
```

### Evaluation Modes

**Ground Truth Mode**: Tests if agent correctly cites the specific ground truth source from dataset.csv
**Multi-Source Mode**: Tests realistic multi-source citation behavior using synthetic_dataset.csv

**Dataset Names in LangSmith**:
- `chatbot-ground-full`: Full ground truth dataset
- `chatbot-ground-single`: Single example test (ground truth)
- `chatbot-multi-full`: Full multi-source dataset
- `chatbot-multi-single`: Single example test (multi-source)

## View Results

All traces and threads appear in your LangSmith project. View:
- **Threads**: Multi-turn conversations with state
- **Insights**: Evaluation metrics and patterns

## Test Agent Locally

```bash
# Start LangGraph Studio
langgraph dev
```

Open http://localhost:8000 to test the agent interactively.
