from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Create and return the Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name
    )