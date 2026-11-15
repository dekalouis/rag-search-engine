import os
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, MOVIE_EMBEDDINGS_PATH, Movie, load_movies


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


EmbeddingArray = NDArray[Any]


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        self.document_map = {}
        lines: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            lines.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(lines, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                return embeddings
        return self.build_embeddings(documents)


def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
