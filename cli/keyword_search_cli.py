#!/usr/bin/env python3

import argparse
import json 
import string
import re 



JSON_FILE = "data/movies.json"
STOPWORD_FILE = "data/stopwords.txt"

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            loadJson(JSON_FILE, args.query)
        case _:
            parser.print_help()


def loadJson(filename, movie_title):
    movie_data = ""
    new_searched_movie = ""
    
    cleaned_movie_search = removePunctuation(movie_title)
    for w in cleaned_movie_search.split():
        if  load_and_strip_stopwords(w) is not None:
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
    



if __name__ == "__main__":
    main()
