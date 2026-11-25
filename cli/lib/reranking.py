import json
import time
from functools import cache

from openai import OpenAIError
from sentence_transformers import CrossEncoder

from .query_enhancement import client, model
from .search_utils import SearchResult

RERANK_DELAY_SECONDS = 3
RERANK_MAX_TRIES = 3


@cache
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


def llm_score(query: str, doc: SearchResult) -> float:
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""

    for _ in range(RERANK_MAX_TRIES):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            return float((response.choices[0].message.content or "").strip())
        except (ValueError, OpenAIError):
            time.sleep(RERANK_DELAY_SECONDS)
    return 0.0


def rerank_individual(
    query: str, results: list[SearchResult], limit: int
) -> list[SearchResult]:
    for i, result in enumerate(results):
        if i > 0:
            time.sleep(RERANK_DELAY_SECONDS)
        result["metadata"]["rerank_score"] = llm_score(query, result)
    results.sort(key=lambda r: r["metadata"]["rerank_score"], reverse=True)
    return results[:limit]


def llm_rerank_batch(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[SearchResult]:
    if not documents:
        return []

    doc_map: dict[int, SearchResult] = {}
    doc_list: list[str] = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    ranking_text = (response.choices[0].message.content or "").strip()

    parsed_ids = json.loads(ranking_text)

    reranked: list[SearchResult] = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            doc = doc_map[doc_id]
            doc["metadata"]["batch_rank"] = i + 1
            reranked.append(doc)

    return reranked[:limit]


def cross_encoder_rerank(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[SearchResult]:
    pairs = [
        [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        for doc in documents
    ]
    scores = get_cross_encoder().predict(pairs)

    for doc, score in zip(documents, scores):
        doc["metadata"]["crossencoder_score"] = float(score)

    documents.sort(key=lambda x: x["metadata"]["crossencoder_score"], reverse=True)
    return documents[:limit]
