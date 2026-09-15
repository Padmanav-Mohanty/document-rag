from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_documents(directory: str):
    """Load PDF documents from the specified directory."""

    knowledge_base_path = Path(directory)
    pdf_files = list(knowledge_base_path.glob("*.pdf"))

    documents = []

    for file_path in pdf_files:
        loader = PyPDFLoader(str(file_path))
        file_docs = loader.load()

        for doc in file_docs:
            doc.metadata["source_file"] = file_path.name
            documents.append(doc)

    print(f"Loaded {len(documents)} pages across {len(pdf_files)} PDF files.")

    return documents