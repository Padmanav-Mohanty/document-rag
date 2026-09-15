from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """Split documents into overlapping chunks for retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return text_splitter.split_documents(documents)