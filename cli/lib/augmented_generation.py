from .hybrid_search import HybridSearch
from .query_enhancement import client, model
from .search_utils import RRF_K, SearchResult, load_movies


def generate_answer(query: str, results: list[SearchResult]) -> str:
    docs = ""
    for result in results:
        docs += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return (response.choices[0].message.content or "").strip()


def rag_command(query: str, limit: int = 5) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, RRF_K, limit)
    answer = generate_answer(query, results)
    return {"search_results": results, "answer": answer}


def summarize_results(query: str, results: list[SearchResult]) -> str:
    docs = ""
    for result in results:
        docs += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {docs}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return (response.choices[0].message.content or "").strip()


def summarize_command(query: str, limit: int = 5) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, RRF_K, limit)
    answer = summarize_results(query, results)
    return {"search_results": results, "summary": answer}


def answer_with_citations(query: str, results: list[SearchResult]) -> str:
    documents = ""
    for i, result in enumerate(results, 1):
        documents += f"[{i}] {result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return (response.choices[0].message.content or "").strip()


def citations_command(query: str, limit: int = 5) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, RRF_K, limit)
    answer = answer_with_citations(query, results)
    return {"search_results": results, "answer": answer}


def answer_question(question: str, results: list[SearchResult]) -> str:
    context = ""
    for result in results:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

    Question: {question}

    Documents:
    {context}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return (response.choices[0].message.content or "").strip()


def question_command(question: str, limit: int = 5) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(question, RRF_K, limit)
    answer = answer_question(question, results)
    return {"search_results": results, "answer": answer}
