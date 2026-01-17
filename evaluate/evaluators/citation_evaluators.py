#!/usr/bin/env python3
"""Citation adherence evaluators for KB agent offline evaluation.

These evaluators assess how well the agent's citations adhere to ground truth:
1. Citation Grounding: Are agent's sources found in retrieved_chunks?
2. Citation Coverage: Does agent cover the ground truth cited_chunks?
3. Citation Accuracy (LLM-as-Judge): Do citations accurately support the answer?
"""

import json
from typing import Any, Optional, List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


# Pydantic models for structured outputs
class EvaluationResult(BaseModel):
    """Structured evaluation result with score and reasoning."""
    score: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the score")


# Global judge LLM - will be initialized lazily
_judge_llm = None


def _get_judge_llm():
    """Get or initialize the judge LLM (GPT-4o-mini for fast, cheap evaluation)."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = init_chat_model("gpt-4o-mini", temperature=0)
    return _judge_llm


def _extract_final_response(run: Any) -> Optional[str]:
    """Extract the final AI response from the run outputs.

    Args:
        run: The agent execution run containing outputs

    Returns:
        The final response text, or None if not found
    """
    if not hasattr(run, "outputs") or not run.outputs:
        return None

    messages = run.outputs.get("messages", [])

    if not messages:
        return None

    # Get last AI message (skip tool calls and user messages)
    for msg in reversed(messages):
        # Handle both dict and object message formats
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if msg_type == "ai" and content and not tool_calls:
                return content
        else:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])

            if msg_type == "ai" and content and not tool_calls:
                return content

    return None


def _parse_citation_line(line: str) -> Optional[str]:
    """Parse a single citation line from the Relevant docs section.

    Handles multiple formats:
    - Markdown links: - [Title](URL)
    - Plain text with notes: - title (notes)
    - Plain text with em-dash: - title — notes
    - Simple plain text: - title

    Args:
        line: A single line from the Relevant docs section

    Returns:
        The extracted KB source name, or None if line is invalid
    """
    import re

    line = line.strip()
    if not line or not line.startswith('-'):
        return None

    # Try markdown link format first: - [Title](URL)
    markdown_match = re.search(r'\[([^\]]+)\]\([^\)]*\)', line)
    if markdown_match:
        return markdown_match.group(1)

    # Try plain text format: - title (with or without notes)
    plain_match = re.search(r'^\s*-\s*(.+?)(?:\s*\(.*\))?\s*$', line)
    if plain_match:
        title = plain_match.group(1).strip()
        # Remove trailing notes with em-dash
        title = re.sub(r'\s*—.*$', '', title).strip()
        if title and not title.startswith('['):
            return title

    return None


def _clean_source_name(source: str) -> str:
    """Clean up a KB source name by removing notes and formatting.

    Args:
        source: Raw source name that may contain notes

    Returns:
        Cleaned source name
    """
    import re

    # Remove parenthetical notes like "(KB)" or "(internal article)"
    cleaned = re.sub(r'\s*\([^\)]*\)\s*', '', source).strip()
    # Remove any remaining brackets
    cleaned = re.sub(r'[\[\]]', '', cleaned).strip()
    # Remove trailing em-dashes and notes
    cleaned = re.sub(r'\s*—.*$', '', cleaned).strip()

    return cleaned


def _extract_kb_sources_from_response(response: str) -> List[str]:
    """Extract KB sources from the final agent response.

    The agent includes a "Relevant docs:" section at the end of responses
    listing KB article names. This parses that section.

    Args:
        response: The agent's final response text

    Returns:
        List of KB source names/topics cited
    """
    import re

    sources = []

    # Look for "Relevant docs:" section
    relevant_docs_match = re.search(
        r'Relevant docs?:\s*(.*?)(?:\n\n|\Z)',
        response,
        re.IGNORECASE | re.DOTALL
    )

    if relevant_docs_match:
        docs_section = relevant_docs_match.group(1)

        # Parse each line separately to handle mixed formats
        lines = docs_section.strip().split('\n')
        for line in lines:
            parsed_source = _parse_citation_line(line)
            if parsed_source:
                sources.append(parsed_source)

    # Clean up sources and deduplicate
    cleaned_sources = []
    for source in sources:
        cleaned = _clean_source_name(source)
        if cleaned and cleaned not in cleaned_sources:
            cleaned_sources.append(cleaned)

    return cleaned_sources


def evaluate_citation_grounding(run: Any, example: Any) -> dict:
    """Evaluate whether agent's KB sources are grounded in the retrieved chunks.

    Checks if the KB source names cited in "Relevant docs:" appear in the ground truth
    retrieved_chunks. Gives partial credit if some sources are hallucinated.

    Per user requirement:
    - If no KB titles are listed, score is 0.0
    - URLs may be hallucinated, so credit is based on KB article name only
    - Hallucinated names get partial credit

    Args:
        run: The agent execution run with final output
        example: The dataset example with ground truth

    Returns:
        A dict with keys:
            - key: "citation_grounding"
            - score: float between 0.0 and 1.0
            - comment: human-readable description
    """
    # Extract agent's response
    agent_response = _extract_final_response(run)

    if not agent_response:
        return {
            "key": "citation_grounding",
            "score": 0.0,
            "comment": "No response found"
        }

    # Extract KB sources from "Relevant docs:" section
    agent_sources = _extract_kb_sources_from_response(agent_response)

    # If agent didn't cite any KB sources, score is 0 (per user requirement)
    if not agent_sources:
        print(f"\n📌 Citation Grounding: 0.0%")
        print(f"   No KB sources listed in 'Relevant docs:' section")
        return {
            "key": "citation_grounding",
            "score": 0.0,
            "comment": "No KB sources listed in response"
        }

    # Get ground truth retrieved chunks (lowercased for comparison)
    gt_retrieved = example.outputs.get("retrieved_chunks", "").lower()

    # Check how many cited sources appear in ground truth
    grounded_count = 0
    for source in agent_sources:
        source_lower = source.lower()
        # Check if KB source name appears in retrieved chunks
        if source_lower in gt_retrieved:
            grounded_count += 1

    # Calculate grounding score
    score = grounded_count / len(agent_sources) if agent_sources else 0.0

    # Build detailed reasoning
    grounded_sources = [s for s in agent_sources if s.lower() in gt_retrieved]
    hallucinated_sources = [s for s in agent_sources if s.lower() not in gt_retrieved]

    reasoning = f"{grounded_count}/{len(agent_sources)} KB sources grounded in retrieved chunks. "
    if grounded_sources:
        reasoning += f"Grounded: {', '.join(grounded_sources[:3])}. "
    if hallucinated_sources:
        reasoning += f"Not found in ground truth: {', '.join(hallucinated_sources[:3])}{'...' if len(hallucinated_sources) > 3 else ''}."

    print(f"\n📌 Citation Grounding: {score:.1%}")
    print(f"   {grounded_count}/{len(agent_sources)} KB source(s) grounded in retrieved chunks")
    if grounded_count < len(agent_sources):
        print(f"   Possible hallucinations: {', '.join(hallucinated_sources[:2])}{'...' if len(hallucinated_sources) > 2 else ''}")

    return {
        "key": "citation_grounding",
        "score": score,
        "comment": reasoning.strip()
    }


def evaluate_citation_presence(run: Any, example: Any) -> dict:
    """Evaluate if agent provided properly formatted citations with KB sources.

    Binary check: did the agent include a "Relevant docs:" section with parseable KB sources?

    Args:
        run: The agent execution run with final output
        example: The dataset example with ground truth

    Returns:
        A dict with keys:
            - key: "citation_presence"
            - score: 1.0 if sources present and parseable, 0.0 otherwise
            - comment: human-readable description
    """
    # Extract agent's response
    agent_response = _extract_final_response(run)

    if not agent_response:
        return {
            "key": "citation_presence",
            "score": 0.0,
            "comment": "No response found"
        }

    # Extract KB sources
    agent_sources = _extract_kb_sources_from_response(agent_response)

    if agent_sources:
        reasoning = f"Agent provided properly formatted citations. Found {len(agent_sources)} KB source(s): {', '.join(agent_sources[:3])}{'...' if len(agent_sources) > 3 else ''}. Citations are in 'Relevant docs:' section."
        print(f"\n📊 Citation Presence: Yes")
        print(f"   {len(agent_sources)} KB source(s) cited")
        return {
            "key": "citation_presence",
            "score": 1.0,
            "comment": reasoning
        }
    else:
        # Check if "Relevant docs:" section exists but no sources extracted
        has_section = "Relevant doc" in agent_response
        if has_section:
            reasoning = "Agent included 'Relevant docs:' section but no parseable KB sources were found. May indicate formatting issues or empty citations."
        else:
            reasoning = "Agent did not include a 'Relevant docs:' section in the response. No KB sources were cited."

        print(f"\n📊 Citation Presence: No")
        print(f"   No parseable KB sources found")
        return {
            "key": "citation_presence",
            "score": 0.0,
            "comment": reasoning
        }


def evaluate_citation_accuracy(run: Any, example: Any) -> dict:
    """Evaluate citation accuracy using LLM-as-judge.

    Uses GPT-4o-mini to assess whether the agent's citations accurately
    support the claims made in the response.

    Args:
        run: The agent execution run with final output
        example: The dataset example with ground truth

    Returns:
        A dict with keys:
            - key: "citation_accuracy"
            - score: float between 0.0 and 1.0
            - comment: human-readable description
    """
    # Extract agent's response
    agent_response = _extract_final_response(run)

    if not agent_response:
        return {
            "key": "citation_accuracy",
            "score": 0.0,
            "comment": "No response found"
        }

    # Extract question and ground truth
    question = example.inputs.get("question", "")
    gt_retrieved = example.outputs.get("retrieved_chunks", "")
    gt_answer = example.outputs.get("answer", "")

    # Create evaluation prompt
    eval_prompt = f"""You are evaluating the accuracy of citations in a customer service response.

QUESTION: {question}

AGENT'S RESPONSE:
{agent_response}

GROUND TRUTH RETRIEVED CHUNKS:
{gt_retrieved}

GROUND TRUTH ANSWER:
{gt_answer}

Evaluate the agent's response on citation accuracy:
1. Do the citations accurately support the claims made?
2. Are there unsupported claims that should be cited?
3. Do citations match the content from retrieved chunks?
4. Is the overall answer consistent with ground truth?

Provide a score from 0.0 to 1.0:
- 1.0: Perfect - all claims cited accurately, fully consistent with ground truth
- 0.8: Good - minor citation issues, mostly accurate
- 0.6: Acceptable - some citation issues but core answer correct
- 0.4: Poor - significant citation problems or inconsistencies
- 0.2: Very poor - major citation errors or wrong answer
- 0.0: Completely inaccurate or no citations

"""

    try:
        judge_llm = _get_judge_llm()
        structured_llm = judge_llm.with_structured_output(EvaluationResult)

        result = structured_llm.invoke([
            {
                "role": "system",
                "content": "You are an expert evaluator. Provide a score and detailed reasoning."
            },
            {
                "role": "user",
                "content": eval_prompt
            }
        ])

        score = max(0.0, min(1.0, result.score))  # Clamp to [0, 1]
        reasoning = result.reasoning

        print(f"\n⚖️  Citation Accuracy (LLM Judge): {score:.1%}")
        print(f"   Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")

        return {
            "key": "citation_accuracy",
            "score": score,
            "comment": reasoning
        }

    except Exception as e:
        return {
            "key": "citation_accuracy",
            "score": 0.5,
            "comment": f"Evaluation failed: {str(e)}"
        }


def evaluate_response_presence(run: Any, example: Any) -> dict:
    """Simple check that agent provided a non-empty response.

    Args:
        run: The agent execution run with final output
        example: Not used for this evaluator

    Returns:
        A dict with keys:
            - key: "response_presence"
            - score: 1.0 or 0.0
            - comment: human-readable description
    """
    response = _extract_final_response(run)

    if not response:
        return {
            "key": "response_presence",
            "score": 0.0,
            "comment": "No response found"
        }

    response_length = len(response.strip())
    has_substantive_response = response_length >= 20

    print(f"\n✅ Response Presence: {'Yes' if has_substantive_response else 'No'}")
    print(f"   Response length: {response_length} chars")

    return {
        "key": "response_presence",
        "score": 1.0 if has_substantive_response else 0.0,
        "comment": f"Response length: {response_length} chars"
    }


def evaluate_answer_correctness(run: Any, example: Any) -> dict:
    """Evaluate if the agent's answer is correct compared to the reference answer.

    Uses LLM-as-judge (GPT-4o-mini) to compare the agent's response to the
    ground truth answer and assess correctness.

    Args:
        run: The agent execution run with final output
        example: The dataset example with ground truth answer

    Returns:
        A dict with keys:
            - key: "answer_correctness"
            - score: float between 0.0 and 1.0
            - comment: human-readable description
    """
    # Extract agent's response
    agent_response = _extract_final_response(run)

    if not agent_response:
        return {
            "key": "answer_correctness",
            "score": 0.0,
            "comment": "No response found"
        }

    # Extract question and ground truth answer
    question = example.inputs.get("question", "")
    gt_answer = example.outputs.get("answer", "")

    # Create evaluation prompt
    eval_prompt = f"""You are evaluating the correctness of a customer service agent's response.

QUESTION: {question}

AGENT'S RESPONSE:
{agent_response}

REFERENCE ANSWER (Ground Truth):
{gt_answer}

Evaluate whether the agent's response is correct by comparing it to the reference answer.
Consider:
1. Does the agent answer the question accurately?
2. Is the information consistent with the reference answer?
3. Are there any factual errors or contradictions?
4. Does the response provide the key information from the reference?

Provide a score from 0.0 to 1.0:
- 1.0: Perfect - accurate, complete, and consistent with reference
- 0.8: Good - mostly accurate with minor issues
- 0.6: Acceptable - correct direction but missing key details
- 0.4: Poor - significant gaps or inaccuracies
- 0.2: Very poor - mostly incorrect or misleading
- 0.0: Completely wrong or no useful information
"""

    try:
        judge_llm = _get_judge_llm()
        structured_llm = judge_llm.with_structured_output(EvaluationResult)

        result = structured_llm.invoke([
            {
                "role": "system",
                "content": "You are an expert evaluator. Provide a score and detailed reasoning."
            },
            {
                "role": "user",
                "content": eval_prompt
            }
        ])

        score = max(0.0, min(1.0, result.score))  # Clamp to [0, 1]
        reasoning = result.reasoning

        print(f"\n📝 Answer Correctness (LLM Judge): {score:.1%}")
        print(f"   Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")

        return {
            "key": "answer_correctness",
            "score": score,
            "comment": reasoning
        }

    except Exception as e:
        return {
            "key": "answer_correctness",
            "score": 0.5,
            "comment": f"Evaluation failed: {str(e)}"
        }
