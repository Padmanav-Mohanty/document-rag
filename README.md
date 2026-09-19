# Document RAG

A Retrieval-Augmented Generation (RAG) pipeline that answers questions over a collection of PDF documents using LangChain, ChromaDB, and OpenRouter LLMs.

## Overview

This project ingests PDF files, splits them into searchable chunks, stores embeddings in a vector database, and uses an LLM to generate grounded answers from the retrieved context.

## Demo

![RAG Demo](assets/rag-demo.png)

**Pipeline:**

```
PDFs → Load → Chunk → Embed → ChromaDB → Retrieve → LLM → Answer
```

## Architecture

The application follows an end-to-end Retrieval-Augmented Generation (RAG) pipeline:

```mermaid
flowchart TD
    A[PDF Documents] --> B[Document Ingestion<br/>LangChain]
    B --> C[Text Chunking<br/>Recursive Text Splitter]
    C --> D[Embedding Generation<br/>HuggingFace Model]
    D --> E[(ChromaDB<br/>Vector Store)]

    subgraph Query Time
        Q[User Question] --> QE[Question Embedding]
        QE --> R[Similarity Search]
        E --> R
        R --> CH[Relevant Chunks]
        CH --> CTX[Context + Question]
        CTX --> LLM[LLM Generation<br/>OpenRouter]
        LLM --> OUT[Answer + Sources]
    end
```

### Pipeline Flow

1. **Document Ingestion**
   PDF documents are loaded and converted into text using LangChain document loaders.

2. **Text Chunking**
   The extracted text is divided into smaller, overlapping chunks using a recursive text splitter. This makes the documents suitable for embedding and retrieval.

3. **Embedding Generation**
   Each text chunk is converted into a numerical vector representation using a Hugging Face embedding model.

4. **Vector Storage**
   The generated embeddings and their corresponding text chunks are stored in ChromaDB.

5. **Retrieval**
   When a user submits a question, the question is embedded and used to retrieve the most semantically relevant document chunks.

6. **LLM Generation**
   The retrieved context is provided to an LLM through OpenRouter, which generates an answer grounded in the retrieved documents.

7. **Source Tracking**
   The application returns the generated answer together with the relevant source documents, allowing the user to trace the information back to the original material.

### End-to-End Flow

```mermaid
flowchart TD
    A[User Question] --> B[Question Embedding]
    B --> C[ChromaDB Similarity Search]
    C --> D[Relevant Document Chunks]
    D --> E[Context + Question]
    E --> F[LLM]
    F --> G[Grounded Answer]
    G --> H[Answer + Sources]
```    

## Features

- **PDF ingestion** — loads and parses PDFs page-by-page with metadata tracking
- **Semantic chunking** — splits documents into overlapping 1000-char chunks for better retrieval
- **Local embeddings** — runs HuggingFace `all-MiniLM-L6-v2` locally, no API key needed for embeddings
- **Persistent vector store** — ChromaDB stores embeddings on disk so re-indexing is only needed once
- **Source attribution** — every answer includes the source filenames it was derived from
- **Interactive CLI** — ask questions from the terminal with a simple REPL
- **Jupyter notebook** — step-by-step walkthrough of the full pipeline

## Project Structure

```
.
├── data/
│   └── sample/                 # Sample PDF documents
│       ├── adam_optimizer.pdf
│       ├── attention_is_all_you_need.pdf
│       ├── bert.pdf
│       ├── gpt3_few_shot_learners.pdf
│       └── resnet.pdf
├── notebooks/
│   └── RAG_Implementation.ipynb  # Interactive notebook walkthrough
├── src/
│   ├── app.py                  # CLI entry point
│   ├── chunking.py             # Document splitting
│   ├── embeddings.py           # HuggingFace embedding model
│   ├── generation.py           # LLM via OpenRouter
│   ├── ingestion.py            # PDF document loading
│   ├── rag_chain.py            # RAG prompt + retrieval chain
│   ├── retrieval.py            # Chroma retriever setup
│   └── vector_store.py         # ChromaDB vector store
├── .env                        # API keys (not committed)
├── requirements.txt
└── LICENSE                     # MIT
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/document-rag.git
cd document-rag
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get a free key at [openrouter.ai](https://openrouter.ai/keys).

### 3. Add your documents

Place PDF files in `data/sample/` (or update the path in `app.py`).

## Usage

### CLI

```bash
cd src
python app.py
```

Then type questions interactively:

```
Question: What is the transformer architecture?

Answer: The transformer is a model architecture introduced in "Attention Is All You Need"...

Sources:
  - attention_is_all_you_need.pdf
  - bert.pdf
```

Type `quit`, `exit`, or `q` to stop.

### Notebook

Open `notebooks/RAG_Implementation.ipynb` in Jupyter and run through each cell to see the pipeline step by step.

## How It Works

| Module | What it does |
|---|---|
| `ingestion.py` | Loads PDFs from a directory using `PyPDFLoader`, attaches source filenames as metadata |
| `chunking.py` | Splits pages into 1000-char chunks with 200-char overlap using `RecursiveCharacterTextSplitter` |
| `embeddings.py` | Creates `all-MiniLM-L6-v2` embeddings via HuggingFace (runs locally) |
| `vector_store.py` | Stores chunks + embeddings in a persistent ChromaDB collection |
| `retrieval.py` | Exposes a retriever that returns the top-4 most relevant chunks for a query |
| `generation.py` | Initializes a free OpenRouter chat model for answer generation |
| `rag_chain.py` | Ties it all together: retrieves context → formats prompt → calls LLM → returns answer + sources |

## Configuration

Key parameters you can adjust:

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `chunking.py` | `1000` | Max characters per chunk |
| `chunk_overlap` | `chunking.py` | `200` | Overlap between consecutive chunks |
| `model_name` | `embeddings.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `k` | `retrieval.py` | `4` | Number of chunks retrieved per query |
| `model` | `generation.py` | `inclusionai/ling-3.0-flash-vl:free` | OpenRouter LLM model |
| `temperature` | `generation.py` | `0` | LLM temperature (0 = deterministic) |

## License

MIT — see [LICENSE](LICENSE).

## References

The following research papers were used as data sources for this project:

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). **Deep Residual Learning for Image Recognition.**
   [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. Brown, T. B., et al. (2020). **Language Models are Few-Shot Learners.**
   [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**
   [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

4. Vaswani, A., et al. (2017). **Attention Is All You Need.**
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

5. Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization.**
   [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
