import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from chunking import split_documents
from embeddings import create_embeddings
from generation import create_llm
from ingestion import load_documents
from rag_chain import create_rag_chain
from retrieval import create_retriever
from vector_store import create_vector_store


def main():
    load_dotenv()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "sample"

    print("Loading documents...")
    documents = load_documents(str(data_dir))

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)

    print("Creating embeddings...")
    embeddings = create_embeddings()

    print("Building vector store...")
    vector_store = create_vector_store(chunks, embeddings)

    print("Creating retriever...")
    retriever = create_retriever(vector_store)

    print("Initializing LLM...")
    llm = create_llm()

    rag_chain = create_rag_chain(retriever, llm)

    print("\nRAG pipeline ready! Type your question or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = rag_chain(question)

        print(f"\nAnswer: {result['answer']}\n")
        print("Sources:")
        for doc in result["sources"]:
            source = doc.metadata.get("source_file", "unknown")
            print(f"  - {source}")
        print()


if __name__ == "__main__":
    main()
