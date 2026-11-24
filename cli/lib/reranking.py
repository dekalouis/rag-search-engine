import time

from openai import OpenAIError

from .query_enhancement import client, model
from .search_utils import SearchResult

RERANK_DELAY_SECONDS = 3
RERANK_MAX_TRIES = 3


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
