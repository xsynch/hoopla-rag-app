#!/usr/bin/env python3

import argparse
from lib import semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    model_parser = subparsers.add_parser("verify", help="Create and Verify Model")

    embed_parser = subparsers.add_parser("embed_text",help="Enter Text for embedding")
    embed_parser.add_argument("text",type=str,help="Text to Embed")

    docembed_parser = subparsers.add_parser("verify_embeddings",help="Load and verify embeddings of movie data")

    embed_query_parter  = subparsers.add_parser("embedquery",help="Query Embeddings to conver query into vectors")
    embed_query_parter.add_argument("query",type=str,help="Query to change to vector")

    search_query_parser = subparsers.add_parser("search",help="Specify words to query in documents")
    search_query_parser.add_argument("query",help="Terms to search for in documents")    
    search_query_parser.add_argument("--limit",type=int, help="Limit the amount of returned answers, the default of 5")
    

    args = parser.parse_args()

            

    match args.command:
        case "verify":
            verify()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query(args.query)
        case "search":
            search(args.query, args.limit)
        
        case _:
            parser.print_help()

def verify():
    model = semantic_search.verify_model()

def embed_text(text):
    semantic_search.embed_text(text)
def verify_embeddings():
    semantic_search.verify_embeddings()
def embed_query(query_text):
    semantic_search.embed_query_text(query_text)
def search(query,limit):
    semantic_search.search(query,limit)

if __name__ == "__main__":
    main()
