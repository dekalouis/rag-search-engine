from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import Movie, load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self, documents: list[Movie] | None = None, model_name: str = "clip-ViT-B-32"
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            doc = self.documents[i]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "score": score,
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    movies = load_movies()
    searcher = MultimodalSearch(movies)
    return searcher.search_with_image(image_path)
