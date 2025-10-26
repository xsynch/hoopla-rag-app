#!/usr/bin/env python3

import argparse
import json 
import string
import re 

from lib.invertedindex import InvertedIndex
from lib import stems
from lib import searchutils



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



if __name__ == "__main__":
    main()
