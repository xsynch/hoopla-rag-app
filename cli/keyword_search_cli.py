#!/usr/bin/env python3

import argparse
import json 

import string
import re 



from lib.invertedindex import InvertedIndex
# from lib import stems
from lib import searchutils




BM25_K1 = 1.5
BM25_B = 0.75

# from lib.searchutils import (
#     BM25_K1,
#     BM25_B
# )

JSON_FILE = "data/movies.json"
STOPWORD_FILE = "data/stopwords.txt"

indexed_movies = InvertedIndex() 

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    freq_search_parser = subparsers.add_parser("tf", help="Search for term frequency")
    freq_search_parser.add_argument("docid", type=str, help="Document ID to search in")
    freq_search_parser.add_argument("term", type=str, help="Term Frequency query")

    idf_parser = subparsers.add_parser("idf",help="Create and list Inverse Document Frequency")
    idf_parser.add_argument("query",help="Term to use")
    
    tfidf_parser = subparsers.add_parser("tfidf",help="Create and list Inverse Document Frequency")
    tfidf_parser.add_argument("docid",help="Document ID to use")
    tfidf_parser.add_argument("term",help="Term to use")
    
    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    

    search_parser = subparsers.add_parser("build", help="Build index of movies and create cache")
    
    

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # loadJson(JSON_FILE, args.query)
            search_in_index(args.query)
        case "tf":
            print(f"Searching term frequency for {args.term} in docid: {args.docid}")            
            term_frequency_search(args.docid, args.term)
        case "build":
            build_index()
        case "idf":
            print(f"Creating Inverse Document Frequency for {args.query}")
            idf_build(args.query)
        case "tfidf":
            print(f"Find the TF-IDF for {args.docid} and {args.term}")
            tfidf_build(args.docid, args.term)
        case "bm25idf":
            bm25idf_build(args.term)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term,args.b,args.k1)
        case "bm25search":
            bm25_search(args.query)

        case _:
            parser.print_help()


def loadJson(filename, movie_title):
    movie_data = ""
    new_searched_movie = ""
    
    cleaned_movie_search = removePunctuation(movie_title)
    for w in cleaned_movie_search.split():
        if  load_and_strip_stopwords(w) is not None:
            w = stems.get_stem_from_token(w)
            new_searched_movie = f"{new_searched_movie} {w}"
        
            
    print(f"New cleaned search: {new_searched_movie}")
        
    with open(filename) as f:
        movie_data = json.load(f)
    
    title_count = 1

    for movie in movie_data["movies"]:
        new_movie_title = ""
        cleaned_movie_title=removePunctuation(movie["title"])
        for w in cleaned_movie_title.split():
            if  load_and_strip_stopwords(w) is not None:
                w = stems.get_stem_from_token(w)
                new_movie_title = f"{new_movie_title} {w}"
        
        new_movie_title = new_movie_title.lower()
                 
        for word in new_searched_movie.split():       
            if(re.search(rf"{word.lower()}\w*" ,new_movie_title)):                
                print(f"{title_count}. {movie["title"]}")
                title_count = title_count + 1
    
def removePunctuation(word):
    translator = str.maketrans('','',string.punctuation) 
    cleaned_string = word.translate(translator)
    return cleaned_string


def load_and_strip_stopwords(data):
    stop_word_list = ""
    stop_word_lines = ""
    with open(STOPWORD_FILE) as f:
        stop_word_list = f.read()
        stop_word_lines = stop_word_list.splitlines()
    if data in stop_word_lines:
        return None 
    else:
        return data 

def build_index():
     
    indexed_movies.build(JSON_FILE)
    indexed_movies.save()

def search_in_index(search_title: str):
    indexed_movies.load()
    results = searchutils.get_movies_from_search(search_title, indexed_movies)
    if len(results) > 0:
        for r in results:
            print(r)
    else:
        print(f"No results found from search of {search_title}")
    # print(f"Results from search: {results}")
    
def term_frequency_search(doc_id:str, term: str):
    indexed_movies.load()
    results = indexed_movies.get_tf(doc_id, term)
    print(results)

def idf_build(term:str):
    
    idf = indexed_movies.get_idf(term)
    print(f"Inverse document frequency is {idf:.2f}")

def tfidf_build(docid:int,term:str):    
    tf_idf = indexed_movies.get_tfidf(int(docid),term)
    print(f"TF-IDF score of '{term}' in document '{docid}': {tf_idf:.2f}")

    
def bm25idf_build(term:str):    
    bm25idf = indexed_movies.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

def bm25_tf_command(doc_id:int, term:str, b, k1):
    bm25tf = indexed_movies.get_bm25_tf(doc_id,term,b,k1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

def bm25_search(query, limit=5):
    scores_list = indexed_movies.bm25_search(query, limit)
    num = 1
    for docid,title,score in scores_list:
        print(f"{num}. ({docid}) {title} - Score: {score})")
        num = num + 1




if __name__ == "__main__":
    main()
