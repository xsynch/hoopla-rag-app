import argparse
from lib.hybrid_search import get_normalized_scores,get_results_weighted_scores

from lib.searchutils import (
    DEFAULT_ALPHA,
    DEFAULT_ALPHA_LIMIT
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize",help="Normalize BM25 and Semantic Scores")
    normalize_parser.add_argument("scores",nargs="+",help="List of scores to normalize")

    alpha_parserver = subparsers.add_parser("weighted-search",help="Perform weighted search operation")
    alpha_parserver.add_argument("query",type=str,help="Query to seearch on")
    alpha_parserver.add_argument("--alpha",type=float,help="Constant to use to dynamically control the weighing of scores")
    alpha_parserver.add_argument("--limit",type=int,default=DEFAULT_ALPHA_LIMIT,help="The amount of results to be returned")

    args = parser.parse_args()

    match args.command:
        case "normalize":            
            normalize_scores(args.scores)
        case "weighted-search":
            weighted_search(args.query,args.alpha,args.limit)
        case _:
            parser.print_help()
    
def normalize_scores(scores):
    if not scores:
        return
    get_normalized_scores(scores)

def weighted_search(query,alpha=DEFAULT_ALPHA,limit=DEFAULT_ALPHA_LIMIT):
    print(f"Getting results for {query}")
    get_results_weighted_scores(query,alpha,limit) 


if __name__ == "__main__":
    main()