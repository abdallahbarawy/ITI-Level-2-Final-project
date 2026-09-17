from threading import Lock
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


class Chunk(TypedDict):
    source: str
    chunk_index: int
    text: str
    distance: float


class RetrievalService:
    def __init__(self, settings: Settings, config: dict):
        if not (settings.vector_store_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Persisted Chroma store not found; run the notebook first")
        self.client = chromadb.PersistentClient(path=str(settings.vector_store_path))
        self.collection = self.client.get_collection(
            config["collection_name"], embedding_function=None
        )
        if self.collection.count() == 0:
            raise ValueError("The exported collection is empty")
        self.model = SentenceTransformer(config["embedding_model"], device="cpu")
        if self.model.get_embedding_dimension() != config["embedding_dim"]:
            raise ValueError("Embedding model dimension differs from the exported store")
        self.lock = Lock()

    def retrieve(self, question: str, k: int = 3) -> list[Chunk]:
        with self.lock:
            vector = self.model.encode([question], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Chunk(
                source=metadata["source"],
                chunk_index=metadata["chunk_index"],
                text=text,
                distance=float(distance),
            )
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
