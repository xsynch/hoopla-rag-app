#!/usr/bin/env python3

import argparse
from lib import semantic_search
from lib import chunkedsemanticsearch




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
    
    chunk_query_parser = subparsers.add_parser("chunk",help="Create a chunk for splitting text")
    chunk_query_parser.add_argument("text",type=str,help="Text to chunk")
    chunk_query_parser.add_argument("--chunk-size",default=200, type=int,help="Set the chunk size, default is 200")
    chunk_query_parser.add_argument("--overlap",type=int, default=0,help="Number of chunks  to create chunk that share words")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk",help="Create a chunk of sentences")
    semantic_chunk_parser.add_argument("text",type=str,help="Sentences to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size",type=int,default=4,help="Max Chunk Size")
    semantic_chunk_parser.add_argument("--overlap",type=int,default=0,help="Overlap for searches")

    build_chunk_embedding_parserver = subparsers.add_parser("embed_chunks",help="Build Chunk Embeddngs")
    search_chunked_parser = subparsers.add_parser("search_chunked",help="Search Chunks for Information")
    search_chunked_parser.add_argument("query",type=str,help="Text to look for in the chunks")
    search_chunked_parser.add_argument("--limit",type=int,default=5,help="Limit the amount of returned data from search")



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
        case "chunk":            
            chunk(args.text, args.chunk_size,args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text,args.max_chunk_size,args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query,args.limit)
        
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
def chunk(text, size, overlapsize):
   semantic_search.chunk_text(text,size,overlapsize)
def semantic_chunk(text,max_chunk_size,overlap):
    semantic_search.semantic_chunk(text,max_chunk_size,overlap)
    
def embed_chunks():
    chunkedsemanticsearch.embed_chunks()
def search_chunked(query,limit):
    chunkedsemanticsearch.search_chunked(query,limit)


if __name__ == "__main__":
    main()
