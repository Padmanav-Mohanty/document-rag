from langchain_chroma import Chroma


def create_vector_store(chunks, embeddings):
    """Create and persist a Chroma vector store from document chunks."""

    db_name = "vector_db"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name,
    )

    return vector_store