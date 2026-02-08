"""Knowledge Base Retrieval Tools

Simple vector-based retrieval from synthetic dataset using in-memory search.
This implementation uses TF-IDF for simplicity but can be upgraded to dense embeddings.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cache for loaded data and vectorizer
_KB_DATA = None
_VECTORIZER = None
_TFIDF_MATRIX = None


def _load_kb_data() -> List[Dict]:
    """Load knowledge base data from CSVs - both ground truth and synthetic (cached)"""
    global _KB_DATA
    if _KB_DATA is None:
        base_path = Path(__file__).parent.parent.parent / "data"

        # Load both datasets
        _KB_DATA = []

        # Load ground truth dataset
        ground_truth_path = base_path / "dataset.csv"
        if ground_truth_path.exists():
            with open(ground_truth_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["source"] = "ground_truth"
                    _KB_DATA.append(row)

        # Load synthetic dataset
        synthetic_path = base_path / "synthetic_dataset.csv"
        if synthetic_path.exists():
            with open(synthetic_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["source"] = "synthetic"
                    _KB_DATA.append(row)

        print(f"Loaded {len(_KB_DATA)} total KB entries (ground truth + synthetic)")

    return _KB_DATA


def _initialize_vectorizer():
    """Initialize TF-IDF vectorizer and compute document vectors (cached)"""
    global _VECTORIZER, _TFIDF_MATRIX

    if _VECTORIZER is None:
        data = _load_kb_data()

        if not data:
            raise ValueError(
                "No knowledge base data found. Ensure dataset.csv or "
                "synthetic_dataset.csv exists in the data/ directory."
            )

        # Combine question and chunks for better retrieval
        documents = []
        for row in data:
            # Create searchable text from question + retrieved_chunks
            text = f"{row['question']} {row['retrieved_chunks']}"
            documents.append(text)

        # Create and fit vectorizer using local variables so that the
        # global _VECTORIZER is only set after a successful fit.
        # This prevents a failed fit_transform from leaving an unfitted
        # vectorizer in the global cache.
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 3), stop_words="english", min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(documents)

        _VECTORIZER = vectorizer
        _TFIDF_MATRIX = tfidf_matrix

    return _VECTORIZER, _TFIDF_MATRIX


def search_knowledge_base(
    query: str, top_k: int = 5, min_similarity: float = 0.1
) -> List[Dict]:
    """Search knowledge base using TF-IDF similarity.

    Args:
        query: Search query string
        top_k: Number of results to return (default: 5)
        min_similarity: Minimum similarity score threshold (default: 0.1)

    Returns:
        List of dictionaries containing:
        - question: The related question
        - retrieved_chunks: The relevant procedural text
        - answer: Synthesized answer
        - similarity_score: Relevance score (0-1)
    """
    # Load data and initialize vectorizer
    data = _load_kb_data()
    vectorizer, tfidf_matrix = _initialize_vectorizer()

    # Transform query
    query_vector = vectorizer.transform([query])

    # Compute similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Filter by minimum similarity and format results
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score >= min_similarity:
            result = {
                "question": data[idx]["question"],
                "retrieved_chunks": data[idx]["retrieved_chunks"],
                "answer": data[idx]["answer"],
                "similarity_score": float(score),
            }
            results.append(result)

    return results


def get_article_by_topic(topic: str) -> Dict | None:
    """Get a specific article/entry by exact topic match.

    Args:
        topic: Exact topic to search for

    Returns:
        Dictionary with question, chunks, and answer, or None if not found
    """
    data = _load_kb_data()

    # Try exact match first
    for row in data:
        if row["question"].lower() == topic.lower():
            return {
                "question": row["question"],
                "retrieved_chunks": row["retrieved_chunks"],
                "answer": row["answer"],
            }

    # Try partial match
    for row in data:
        if topic.lower() in row["question"].lower():
            return {
                "question": row["question"],
                "retrieved_chunks": row["retrieved_chunks"],
                "answer": row["answer"],
            }

    return None


def list_available_topics(category: str | None = None) -> List[str]:
    """List all available topics in the knowledge base.

    Args:
        category: Optional category filter (e.g., "payment", "dispute", "fraud")

    Returns:
        List of topic strings
    """
    data = _load_kb_data()

    topics = [row["question"] for row in data]

    if category:
        # Filter topics by category keyword
        topics = [t for t in topics if category.lower() in t.lower()]

    return sorted(set(topics))


@lru_cache(maxsize=100)
def search_knowledge_base_cached(query: str, top_k: int = 5) -> str:
    """Cached version of search_knowledge_base that returns formatted string.

    This is useful for LLM tool calling where you want consistent output format.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        Formatted string with search results
    """
    results = search_knowledge_base(query, top_k=top_k)

    if not results:
        return f"No results found for query: {query}"

    output = []
    output.append(f"Found {len(results)} results for: {query}\n")

    for i, result in enumerate(results, 1):
        output.append(f"\n{'=' * 60}")
        output.append(f"Result {i} (similarity: {result['similarity_score']:.3f})")
        output.append(f"{'=' * 60}")
        output.append(f"\nTopic: {result['question']}")
        output.append(f"\nAnswer: {result['answer']}")
        output.append(f"\nDetailed Information:\n{result['retrieved_chunks'][:500]}...")

    return "\n".join(output)


# Tool functions for LangChain integration


def search_kb_tool(query: str, num_results: int = 3) -> str:
    """Search the knowledge base for relevant information.

    Use this tool to find answers to customer questions about banking services,
    credit cards, payments, disputes, fraud protection, and other banking topics.

    Args:
        query: The customer's question or search query
        num_results: Number of results to return (default: 3, max: 10)

    Returns:
        Formatted search results with relevant information
    """
    num_results = min(num_results, 10)  # Cap at 10 results
    results = search_knowledge_base(query, top_k=num_results, min_similarity=0.05)

    if not results:
        return f"No relevant information found for: {query}\nTry rephrasing your query or searching for related topics."

    output = []
    for i, result in enumerate(results, 1):
        output.append(
            f"\n--- Result {i} (relevance: {result['similarity_score']:.2f}) ---"
        )
        output.append(f"Topic: {result['question']}")
        output.append(f"\nAnswer: {result['answer']}")

        # Include chunks for detailed information
        chunks = result["retrieved_chunks"].split("\n\n")
        if chunks:
            output.append("\nDetailed Procedures:")
            # Include first 2-3 chunks for context
            for chunk in chunks[:3]:
                if chunk.strip():
                    output.append(f"  • {chunk.strip()[:300]}...")

    return "\n".join(output)


def get_topic_details(topic: str) -> str:
    """Get detailed information about a specific topic.

    Use this tool when you need complete procedural details about a specific
    banking topic or process.

    Args:
        topic: The specific topic to retrieve

    Returns:
        Complete information about the topic
    """
    article = get_article_by_topic(topic)

    if not article:
        # Try searching for similar topics
        results = search_knowledge_base(topic, top_k=3)
        if results:
            similar_topics = [r["question"] for r in results]
            return (
                f"Topic '{topic}' not found.\n\nDid you mean one of these?\n"
                + "\n".join(f"  - {t}" for t in similar_topics)
            )
        return f"Topic '{topic}' not found in knowledge base."

    output = []
    output.append(f"Topic: {article['question']}")
    output.append(f"\n{'=' * 60}")
    output.append(f"Summary: {article['answer']}")
    output.append(f"\n{'=' * 60}")
    output.append(f"Detailed Information:\n\n{article['retrieved_chunks']}")

    return "\n".join(output)


def list_topics(category: str | None = None) -> str:
    """List available topics in the knowledge base.

    Use this tool to discover what topics are available to search.

    Args:
        category: Optional category filter (e.g., "payment", "dispute", "fraud")

    Returns:
        List of available topics
    """
    topics = list_available_topics(category)

    if not topics:
        return f"No topics found{' for category: ' + category if category else ''}."

    output = [f"Available topics{' in category: ' + category if category else ''}:\n"]
    for topic in topics:
        output.append(f"  • {topic}")

    output.append(f"\nTotal: {len(topics)} topics")

    return "\n".join(output)


# Export tools
__all__ = [
    "search_kb_tool",
    "get_topic_details",
    "list_topics",
    "search_knowledge_base",
    "get_article_by_topic",
    "list_available_topics",
]
