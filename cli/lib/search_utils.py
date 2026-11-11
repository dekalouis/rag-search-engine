import json
import os
from typing import Any, TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


#
# class GoldenTestCase(TypedDict):
#     query: str
#     relevant_docs: list[str]
#
# class GoldenDataset(TypedDict):
#     test_cases: list[str]

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
# GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]
