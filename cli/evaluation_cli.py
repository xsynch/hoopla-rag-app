import argparse
import json

from lib.searchutils import GOLDEN_DATASET, load_movies
from lib import hybrid_search
from lib import semantic_search

def main():
    golden_data = ""
    results = []
    
    relevant_titles = []
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument("--debug",type=bool,default=False,help="Print Debug Messages During Search")

    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATASET) as file:
        golden_data = json.load(file)

    # golden_relevant_titles = [''.join(x["relevant_docs"]) for x in golden_data["test_cases"]]
    # print(f"Golden Relevant titles: {golden_relevant_titles}")
    # return 
    documents = load_movies()
    semantic_searcher = semantic_search.SemanticSearch()
    semantic_searcher.load_or_create_embeddings(documents)
    hybrid_searcher = hybrid_search.HybridSearch(documents)
    print(f"k={limit}\n\n")
    test_cases = golden_data["test_cases"]
    for data in test_cases:
        titles = []
        golden_relevant_titles = []
        query = data["query"]
        for gold_title in data["relevant_docs"]:
            golden_relevant_titles.append(gold_title)
        results = hybrid_searcher.rrf_search(query=query,k=60,limit=limit)
        top_k_results = results[:limit]
        for values in top_k_results:
            titles.append(values[1]["title"])
            # print(f"{values[1]["title"] in golden_relevant_titles}")
        relevant_retrieved = sum(1 for title in titles if title in golden_relevant_titles)
        
        
        
        relevant_titles = [x for x in titles if x in golden_relevant_titles]
        # print(f"Relevant sum: {relevant_retrieved} with top k length: {len(top_k_results)}")
        precision = relevant_retrieved / len(top_k_results)
        recall = relevant_retrieved / len(set(data["relevant_docs"]))
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"""- Query: {query}\n  - Precision@{limit}:{precision: .4f}
  - Recall@{limit}: {recall:.4f}
  - F1 Score: {f1:.4f}
  - Retrieved: {', '.join(titles)}
  - Relevant {', '.join(relevant_titles)} """)
    # print(f"Found {len(results)} results")
    # print(results[0][1][1]["title"])
    



if __name__ == "__main__":
    main()