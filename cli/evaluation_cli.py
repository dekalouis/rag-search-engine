import argparse

from lib.hybrid_search import HybridSearch
from lib.search_utils import RRF_K, load_golden_dataset, load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    dataset = load_golden_dataset()
    test_cases = dataset["test_cases"]
    searcher = HybridSearch(load_movies())
    print(f"k={limit}\n")
    for case in test_cases:
        results = searcher.rrf_search(case["query"], RRF_K, limit)
        titles = [r["title"] for r in results]
        relevant = case["relevant_docs"]
        hits = [t for t in titles if t in relevant]
        precision = len(hits) / len(titles) if titles else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {case['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(titles)}")
        print(f"  - Relevant: {', '.join(relevant)}")
        print()


if __name__ == "__main__":
    main()
