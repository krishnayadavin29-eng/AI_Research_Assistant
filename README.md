# 🔬 AI Research Assistant

A Streamlit application for asking questions about research papers and other documents. It combines document ingestion, chunking, Gemini embeddings, FAISS vector search, BM25 keyword search, reranking, and Gemini-powered answer generation. Responses are grounded in retrieved passages and include source citations where available.

> **Status:** This README describes the repository's current implementation. The app requires a Google Gemini API key to create embeddings and generate answers. Uploaded documents and the FAISS index are handled by the running app's filesystem; plan for persistent storage if deploying to an environment with ephemeral disks.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the app](#run-the-app)
- [Use the app](#use-the-app)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Data and privacy](#data-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Features

- Upload multiple research documents through the Streamlit interface.
- Read PDF, plain-text (`.txt`), and Markdown (`.md`) files.
- Split documents into overlapping text chunks and retain source metadata.
- Index chunks using Gemini embeddings and a local FAISS vector store.
- Combine semantic vector retrieval with BM25 keyword retrieval.
- Rewrite user questions into retrieval queries, then rerank candidate passages with a cross-encoder.
- Generate answers with Gemini using retrieved text as context; the prompt instructs the model to acknowledge when the context is insufficient.
- Show citations and source information alongside answers.
- Summarize the indexed document collection.
- Keep conversation history during the current Streamlit session and clear it from the sidebar.

## How it works

The main orchestration is implemented in `pipeline.py` as `ResearchPipeline`.

### Document ingestion

1. `loaders/document_loader.py` reads supported files and attaches source metadata.
2. `processing/text_splitter.py` splits loaded documents into chunks. The default chunk size is 800 characters with 150 characters of overlap.
3. `processing/embeddings.py` creates Gemini embedding instances.
4. `retrieval/vector_store.py` builds and saves a FAISS index.
5. A BM25 retriever is built for lexical keyword retrieval.

### Question answering

1. The question is rewritten into retrieval variants.
2. The hybrid retriever searches the FAISS and BM25 indexes.
3. Candidate passages are deduplicated and reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
4. The selected context and conversation history are sent to Gemini.
5. The UI displays the answer and formatted citations.

## Requirements

- Python 3.10 or newer is recommended.
- A Google Gemini API key with access to the configured model and embedding service.
- Internet access for Gemini API calls and for the first download of the sentence-transformers reranker model.

Python dependencies are listed in [`requirements.txt`](requirements.txt), including Streamlit, LangChain integrations, FAISS, sentence-transformers, PDF loading, and dotenv support.

## Setup

Clone the repository and move into the project directory:

```bash
git clone https://github.com/krishnayadavin29-eng/AI_Research_Assistant.git
cd AI_Research_Assistant
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure the Gemini API key

The application accepts the key in either of these ways:

1. Enter it in the **Google API Key** field in the app's sidebar; or
2. Set `GOOGLE_API_KEY` in the environment before launching the app.

For local development, you can create a `.env` file in the project root:

```dotenv
GOOGLE_API_KEY=your_google_gemini_api_key
```

`config.py` loads environment variables with `python-dotenv`. Keep `.env` private: do not commit API keys to GitHub. On a hosting provider, configure the key as a secret/environment variable rather than placing it in source code.

## Run the app

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) and may also provide a network URL.

## Use the app

1. Open the Streamlit URL in a browser.
2. Add your Gemini API key in the sidebar, or provide it through the environment.
3. In **Upload PDFs**, select one or more `.pdf`, `.txt`, or `.md` files.
4. Select **Index Uploaded Files** and wait for ingestion and indexing to finish.
5. Ask a question in the main chat area. The app will retrieve relevant passages and show an answer with sources.
6. Select **Summarise All Documents** to request a corpus summary.
7. Select **Clear Chat History** to reset the conversation for the current session.

The **Use Existing Index** tab attempts to load an index from the configured local index path. It is useful when that index exists on the same persistent filesystem; a fresh machine or ephemeral deployment will not automatically have the index.

## Project structure

```text
.
├── app.py                       # Streamlit user interface
├── config.py                    # Models, retrieval settings, and data paths
├── pipeline.py                  # Ingestion and question-answering orchestration
├── requirements.txt             # Python dependencies
├── chains/
│   ├── qa_chain.py              # Context-grounded Gemini answers
│   └── summary_chain.py         # Document collection summaries
├── data/
│   ├── papers/                  # Default location for local source documents
│   └── faiss_index/             # Default location for persisted FAISS index
├── loaders/
│   └── document_loader.py       # PDF, TXT, and Markdown loading
├── memory/
│   └── chat_memory.py           # Conversation history helper
├── processing/
│   ├── embeddings.py            # Gemini embeddings wrapper
│   └── text_splitter.py         # Document chunking
├── retrieval/
│   ├── bm25_search.py           # Keyword retrieval
│   ├── hybrid_retriever.py      # Combined retrieval
│   ├── reranker.py              # Cross-encoder reranking
│   ├── rewrite_query.py         # Query rewriting
│   └── vector_store.py          # FAISS index management
└── utils/
    └── citation.py              # Source and citation formatting
```

## Configuration

Default settings are in `config.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini chat model used for answers and related LLM tasks |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Maximum output tokens |
| `EMBEDDING_MODEL` | `models/embedding-001` | Gemini embedding model |
| `CHUNK_SIZE` | `800` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K_VECTOR` | `6` | Vector results requested per search |
| `TOP_K_BM25` | `6` | BM25 results requested per search |
| `TOP_K_FINAL` | `4` | Number of passages retained after reranking |
| `HYBRID_ALPHA` | `0.5` | Balance between vector and BM25 retrieval |
| `DATA_DIR` | `data/papers` | Default local document directory |
| `VECTOR_STORE_PATH` | `data/faiss_index` | Local FAISS persistence directory |

These are Python constants in `config.py`; edit them there to change the current configuration.

## Data and privacy

- Documents are processed by the application and their text is sent to the configured Gemini services when embeddings or answers are requested. Do not upload sensitive or restricted material unless you are authorized and the provider's data terms meet your requirements.
- The FAISS index is saved under `data/faiss_index` by default. Keep generated indexes and source documents out of public repositories unless you intend to publish them.
- Conversation memory is maintained by the app's in-memory session state; it is not a permanent chat-history database.
- Local files and indexes are not automatically durable on platforms with ephemeral filesystems. For a hosted multi-user or production deployment, use suitable persistent storage and review access controls, retention, and API-key management.

## Troubleshooting

### The app says the API key is missing or authentication fails

Set `GOOGLE_API_KEY` in your environment or enter a valid key in the sidebar. Confirm the key has access to the configured Gemini models and that it is not expired or restricted.

### Indexing or answer generation fails

Check your internet connection, API quota, and provider availability. Review the error shown by the app; PDF parsing, embedding requests, and model generation can fail independently.

### PDF text is missing or incomplete

Some PDFs contain scanned images rather than selectable text. The current loader uses PDF text extraction and does not provide OCR, so scanned documents may need OCR processing before upload.

### The reranker is slow on first use

The cross-encoder model may need to download the first time `sentence-transformers` is used. Allow network access and enough disk space; subsequent runs can use the local model cache.

### A saved index cannot be found

The index is loaded from `VECTOR_STORE_PATH` in `config.py` (default: `data/faiss_index`). Ensure the index files exist at that path in the current environment. Index files saved on another machine or a temporary deployment may not be available.

## Limitations

- Supported inputs are PDF, TXT, and Markdown; other formats are not loaded by the current document loader.
- The app depends on external Gemini services for embeddings and answer generation.
- The FAISS index and uploaded-file workflow are local-filesystem based, not a shared database-backed service.
- Retrieval and generated answers can be incomplete or incorrect. Check the cited source documents for important decisions and research claims.
- The repository currently provides a Streamlit app (`streamlit run app.py`). This README does not claim that a separate Vercel deployment entry point is included.

## Contributing

Bug reports and improvements are welcome. When reporting an issue, include the Python version, operating system, steps to reproduce, and the relevant error message. Never include API keys, private documents, or other secrets in an issue.

## License

No license file was visible in the repository when this README was prepared. Unless a license is added, do not assume the project is available for reuse, modification, or redistribution under an open-source license.
