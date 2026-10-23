
from pathlib import Path
import hashlib

import chromadb
from ollama import embed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
CHROMA_PATH = DATA_PATH / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    text = text.strip()
    if not text:
        return []

    return [
        text[start : start + chunk_size]
        for start in range(0, len(text), chunk_size - overlap)
    ]


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name="personal_knowledge")



def index_text(file_path: Path, text: str) -> int:
    """Index extracted text while preserving other documents' records."""
    file_path = file_path.resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("The document is empty.")

    collection = get_collection()

    source_id = hashlib.sha256(
        str(file_path).lower().encode("utf-8")
    ).hexdigest()[:16]

    existing = collection.get(
        where={"source_id": source_id},
        include=["metadatas"],
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    result = embed(model=EMBEDDING_MODEL, input=chunks)

    chunk_ids = [
        f"{source_id}-{i}" for i in range(len(chunks))
    ]

    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=result.embeddings,
        metadatas=[
            {
                "source": file_path.name,
                "source_id": source_id,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ],
    )

    return len(chunks)


def index_file(file_path: Path) -> int:
    """Read and index a UTF-8 TXT file."""
    file_path = file_path.resolve()

    if file_path.suffix.lower() != ".txt":
        raise ValueError("Only TXT files are supported by index_file().")

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8-sig")
    return index_text(file_path, text)



def index_document() -> None:
    """Keep the original project-note indexing workflow."""
    document_path = DATA_PATH / "sample_notes.txt"
    count = index_file(document_path)
    print(f"Indexed {count} text chunk(s) from {document_path.name}.")



def list_documents() -> list[dict]:
    """List indexed files, including records from the older schema."""
    collection = get_collection()
    records = collection.get(include=["metadatas"])

    documents = {}

    for record_id, metadata in zip(records["ids"], records["metadatas"]):
        metadata = metadata or {}

        source = metadata.get("source", "unknown")
        source_id = metadata.get("source_id")

        # Older records may not have a source_id.
        if not source_id:
            source_id = f"legacy:{source}"

        if source_id not in documents:
            documents[source_id] = {
                "source_id": source_id,
                "filename": source,
                "chunks": 0,
            }

        documents[source_id]["chunks"] += 1

    return list(documents.values())



def search_knowledge(question: str, top_k: int = 3) -> list[dict]:
    result = embed(model=EMBEDDING_MODEL, input=question)

    collection = get_collection()
    total = collection.count()

    if total == 0:
        raise RuntimeError("No documents indexed. Index a document first.")

    matches = collection.query(
    query_embeddings=result.embeddings,
    n_results=min(top_k, total),
    include=["documents", "metadatas", "distances"],
    )

    documents = matches["documents"][0]
    metadatas = matches["metadatas"][0]
    distances = matches["distances"][0]

    return [
    {
        "text": document,
        "filename": (metadata or {}).get("source", "Unknown document"),
        "distance": distance,
    }
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    )
    ]



def remove_document(source_id: str) -> bool:
    """Remove only the chunks belonging to one indexed document."""
    collection = get_collection()

    if source_id.startswith("legacy:"):
        return False

    existing = collection.get(
        where={"source_id": source_id},
        include=["metadatas"],
    )

    if not existing["ids"]:
        return False

    collection.delete(ids=existing["ids"])
    return True


if __name__ == "__main__":
    index_document()
