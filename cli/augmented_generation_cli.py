import argparse


from lib.augmented_genration import get_augmented_results, get_llm_summary, get_llm_citations, get_answers


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser("summarize", help="Search query for RAG")
    summary_parser.add_argument("query",type=str,help="query to use")
    summary_parser.add_argument("--limit",type=int,default=5,help="Number of entries to return")

    citations_parser = subparsers.add_parser("citations", help="Retrieve Citations for seaerch results")
    citations_parser.add_argument("query",type=str,help="query to use")
    citations_parser.add_argument("--limit",type=int,default=5,help="Number of entries to return")

    answer_parser = subparsers.add_parser("question", help="Retrieve Citations for seaerch results")
    answer_parser.add_argument("context",type=str,help="query to use")
    answer_parser.add_argument("--limit",type=int,default=5,help="Number of entries to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            run_augmented_query(query)
        case "summarize":
            query = args.query
            run_multidoc_summary(query,args.limit)
        case "citations":
            query = args.query
            get_results_citations(query,args.limit)
        case "question":
            get_answer_question(args.context,args.limit)
        case _:
            parser.print_help()


def run_augmented_query(query):
    get_augmented_results(query)

def run_multidoc_summary(query,limit):
    get_llm_summary(query,limit)

def get_results_citations(query,limit):
    get_llm_citations(query,limit)

def get_answer_question(question,limit):
    get_answers(question,limit)

if __name__ == "__main__":
    main()