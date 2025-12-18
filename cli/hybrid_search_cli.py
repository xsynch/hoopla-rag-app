import argparse
from lib.hybrid_search import get_normalized_scores,get_results_weighted_scores,get_rrf_search
from lib.searchutils import get_gemini_response, get_gemini_evaluation

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

    rrf_search_parser = subparsers.add_parser("rrf-search",help="Execute Hybrid Search using Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query",type=str,help="What to search for")
    rrf_search_parser.add_argument("--k",type=int,default=60,help="K to use for the search")
    rrf_search_parser.add_argument("--limit",type=int,default=5,help="Limit the amount of returned values")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell","rewrite","expand"],help="Query enhancement method",)
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual","batch","cross_encoder"],help="Query enhancement method",)
    rrf_search_parser.add_argument("--debug",default=False,type=bool,help="Print Debug messages as search happens")
    rrf_search_parser.add_argument("--evaluate",help="Evaluate the results of a search",action='store_true')
    

    args = parser.parse_args()

    match args.command:
        case "normalize":            
            normalize_scores(args.scores)
        case "weighted-search":
            weighted_search(args.query,args.alpha,args.limit)
        case "rrf-search":
            rrf_search(args.query,args.k,args.limit,args.enhance,args.rerank_method,args.evaluate,args.debug)
        case _:
            parser.print_help()
    
def normalize_scores(scores):
    if not scores:
        return
    get_normalized_scores(scores)

def weighted_search(query,alpha=DEFAULT_ALPHA,limit=DEFAULT_ALPHA_LIMIT):
    print(f"Getting results for {query}")
    get_results_weighted_scores(query,alpha,limit) 

def rrf_search(query,k,limit,enhanced_search,rerank_method,evaluate,debug=False):
    if not enhanced_search and rerank_method:
        get_rrf_search(query,k,limit,rerank_method=rerank_method,evaluate=evaluate,debug=debug)
    elif enhanced_search:
        enhanced_terms = get_gemini_response(enhanced_search,query)
        get_rrf_search(enhanced_terms,k,limit,rerank_method=rerank_method,evaluate=evaluate,debug=debug)
    else:        
        get_rrf_search(query,k,limit,rerank_method=rerank_method,evaluate=evaluate,debug=debug)



if __name__ == "__main__":
    main()