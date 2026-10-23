# Private AI Knowledge Base

A local personal AI assistant that lets you upload documents and ask questions about their contents. Built with Python, Ollama, ChromaDB, FastAPI, and Streamlit.

## Overview

Private AI Knowledge Base is a Retrieval-Augmented Generation (RAG) project designed to help users search and understand their own documents using a locally running language model.

The goal is to keep document processing and AI inference on your own computer, reducing reliance on external AI services.

### Features

- **Document Q&A:** Ask questions about your indexed documents.
- **Local LLM:** Generate answers using Ollama and Qwen3.
- **Semantic search:** Create embeddings with `nomic-embed-text` and store them in ChromaDB.
- **Web interface:** Interact with the assistant through Streamlit.
- **REST API:** Use FastAPI endpoints to ask questions and manage documents.
- **Document uploads:** Upload TXT and PDF documents.
- **Scanned PDF support:** Use Tesseract OCR to extract text from scanned pages when needed.
- **Document management:** List indexed documents and remove their indexed content.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Ollama | Run language and embedding models locally |
| Qwen3 1.7B | Local language model |
| nomic-embed-text | Generate text embeddings |
| ChromaDB | Store and retrieve document embeddings |
| FastAPI | Provide REST API endpoints |
| Streamlit | Provide the web interface |
| PyMuPDF and pypdf | PDF processing |
| Tesseract OCR | Extract text from scanned PDF pages |

## Architecture

```text
                 User
                   |
             Streamlit UI
                   |
             FastAPI Backend
                   |
          Document Processing
             /           \
       Text Extraction   OCR
             \           /
              Text Chunks
                   |
          Embedding Model
        (nomic-embed-text)
                   |
               ChromaDB
                   |
          Relevant Context
                   |
             Qwen3 via Ollama
                   |
              Final Answer
```

## Project Structure

```text
private-ai-knowledge-base/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── rag/
│       ├── __init__.py
│       └── pipeline.py
├── data/
│   ├── data_science_notes.txt
│   ├── ml_notes.txt
│   ├── python_notes.txt
│   └── sample_notes.txt
├── .gitignore
├── requirements.txt
├── test_ollama.py
└── ui.py
```

Local uploads and the ChromaDB database are excluded from version control.

## Prerequisites

Before running the project, install:

- Python 3.12 or a compatible version
- Git
- [Ollama](https://ollama.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for scanned PDF support

The application also requires the Ollama models used by the project.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manishkr657/private-ai-knowledge-base.git
cd private-ai-knowledge-base
```

Because this repository is private, you must have GitHub access to clone it.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you use Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Ollama models

Make sure Ollama is installed and running, then execute:

```bash
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
```

Check the installed models:

```bash
ollama list
```

### 5. Configure Tesseract OCR

On Windows, install Tesseract OCR and verify that it works:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

The current application configuration uses this Windows installation path. If Tesseract is installed elsewhere, update the path in `app/main.py`.

## Running the Application

### Start the Streamlit interface

From the project root, activate your virtual environment and run:

```bash
python -m streamlit run ui.py
```

Open the local URL printed by Streamlit in your browser.

### Start the FastAPI backend

If you want to run the API separately:

```bash
python -m uvicorn app.main:app --reload
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

The API includes document upload, document listing, document deletion, and question-answering functionality.

## How RAG Works

Retrieval-Augmented Generation combines document retrieval with a language model.

1. A user uploads a document.
2. The application extracts its text.
3. The text is split into smaller chunks.
4. The embedding model converts the chunks into vectors.
5. ChromaDB stores those vectors for retrieval.
6. When the user asks a question, relevant chunks are retrieved.
7. The local language model uses the retrieved context to generate an answer.

This approach helps the assistant answer questions based on the user's own documents rather than relying only on the language model's pre-existing knowledge.

## Privacy and Data Storage

This project is designed for local use.

- The language and embedding models run through Ollama on the local machine.
- Uploaded documents are stored locally.
- ChromaDB stores indexed document data locally.
- The `.gitignore` excludes the Python virtual environment, uploaded documents, and configured local database directories.

**Important:** Review the code and configuration before using sensitive documents. Local execution does not automatically guarantee complete privacy or security.

## Hardware Considerations

The project is being developed on a resource-constrained Windows laptop. Small local models are used to make experimentation possible without a dedicated GPU.

Response speed depends on the CPU, available RAM, document size, model, and amount of retrieved context. OCR processing can be slow for large scanned PDFs.

## Current Development Goals

- Improve response time on low-resource hardware.
- Improve scanned PDF extraction and indexing.
- Expand support for multiple documents and topics.
- Improve retrieval quality and answer reliability.
- Improve the user interface and document management experience.

## Contributing

This is a learning project under active development. Suggestions and improvements are welcome.

## License

No license has been added yet. All rights are reserved by default unless a license is added to the repository.
