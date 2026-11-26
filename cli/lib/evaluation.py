import json

from .query_enhancement import client, model
from .search_utils import SearchResult


def llm_judge_results(query: str, results: list[SearchResult]) -> list[int]:
    formatted_results: list[str] = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result['title']}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    text = (response.choices[0].message.content or "").strip()

    scores = json.loads(text)
    if len(scores) != len(results):
        raise ValueError("not the same length")
    return [int(s) for s in scores]
