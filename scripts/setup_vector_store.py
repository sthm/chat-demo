"""Setup vector store for knowledge base retrieval.

This script:
1. Loads synthetic dataset CSV
2. Creates embeddings for each chunk
3. Builds FAISS vector index
4. Saves index and metadata for runtime use
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def load_synthetic_data(csv_path: str) -> List[Dict]:
    """Load synthetic dataset from CSV"""
    data = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def create_documents_from_data(data: List[Dict]) -> List[Document]:
    """Convert CSV rows into LangChain documents"""
    documents = []

    for idx, row in enumerate(data):
        question = row["question"]
        retrieved_chunks = row["retrieved_chunks"]
        answer = row["answer"]

        # Split retrieved_chunks into individual chunks (they're separated by newlines/sections)
        chunks = [c.strip() for c in retrieved_chunks.split("\n\n") if c.strip()]

        for chunk_idx, chunk in enumerate(chunks):
            # Create a document for each chunk
            doc = Document(
                page_content=chunk,
                metadata={
                    "question": question,
                    "answer": answer,
                    "chunk_id": f"{idx}_{chunk_idx}",
                    "source": "synthetic_dataset",
                },
            )
            documents.append(doc)

    return documents


def build_vector_store(documents: List[Document], embeddings) -> FAISS:
    """Build FAISS vector store from documents"""
    print(f"Building vector store with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def main():
    """Main setup function"""
    print("=" * 60)
    print("Setting up Vector Store for Knowledge Base")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    csv_path = data_dir / "synthetic_dataset.csv"
    vector_store_dir = data_dir / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading synthetic dataset...")
    data = load_synthetic_data(str(csv_path))
    print(f"✓ Loaded {len(data)} questions from CSV")

    # Create documents
    print("\n[2/4] Creating documents from data...")
    documents = create_documents_from_data(data)
    print(f"✓ Created {len(documents)} document chunks")

    # Initialize embeddings
    print("\n[3/4] Initializing embeddings model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✓ Using OpenAI text-embedding-3-small model")

    # Build vector store
    print("\n[4/4] Building FAISS vector store...")
    vectorstore = build_vector_store(documents, embeddings)

    # Save vector store
    vectorstore_path = vector_store_dir / "faiss_index"
    vectorstore.save_local(str(vectorstore_path))
    print(f"✓ Vector store saved to: {vectorstore_path}")

    # Save metadata
    metadata = {
        "num_documents": len(documents),
        "num_questions": len(data),
        "embedding_model": "text-embedding-3-small",
        "vectorstore_type": "FAISS",
    }
    metadata_path = vector_store_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("Vector Store Setup Complete!")
    print("=" * 60)
    print("\nStats:")
    print(f"  - Questions: {len(data)}")
    print(f"  - Document chunks: {len(documents)}")
    print(f"  - Average chunks per question: {len(documents) / len(data):.1f}")
    print("\nVector store ready for use in retrieval tools!")


if __name__ == "__main__":
    main()
