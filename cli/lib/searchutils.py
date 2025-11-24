import re 
import string 
from .invertedindex import InvertedIndex
from .stems import *
import json
import os 
BM25_K1 = 1.5
BM25_B = 0.75

MAX_CHUNK_SIZE=4
OVERLAP=0
SCORE_PRECISION = 3


DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

DEFAULT_ALPHA=0.5
DEFAULT_ALPHA_LIMIT=5



STOPWORD_FILE = "data/stopwords.txt"

CACHE_DIR = "cache"
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def get_movies_from_search(search_title, movieIndex:InvertedIndex):
    movie_data = ""
    new_searched_movie = ""
    results = []
    
    cleaned_movie_search = removePunctuation(search_title)
    for w in cleaned_movie_search.split():
        if  load_and_strip_stopwords(w) is not None:
            w = get_stem_from_token(w)
            new_searched_movie = f"{new_searched_movie} {w}"
    
        
            

    
    tokens = new_searched_movie.split()
    movieList = movieIndex.index.keys()
    # print(f"Searching {len(movieList)} movies for {new_searched_movie}")
    for movie in movieList:  
        movie_stem = get_stem_from_token(movie)      
        for t in tokens:                
            if(re.search(rf"{t.lower()}\w*" ,movie_stem.lower())):   
                movieInfo =   (movieIndex.get_documents(movie))[0]
                # print(f"match found: {movie}")
                if movieInfo not in results:           
                    results.append((movieIndex.get_documents(movie))[0])
            if len(results) >= 5:
                movie_and_id = []
                for found_movie in results:
                    title = movieIndex.docmap[found_movie]["title"]
                    movie_id = movieIndex.docmap[found_movie]["id"]
                    movie_and_id.append(f"{movie_id} {title}")
                return movie_and_id 
    for found_movie in results:
        title = movieIndex.docmap[found_movie]["title"]
        movie_id = movieIndex.docmap[found_movie]["id"]
        print(f"Movie Title: {title} with id: {movie_id}")
    return results
    #                                       
    #         if(re.search(rf"{t.lower()}\w*" ,movieIndex.get_documents(t))):                
    #             print(f"{title_count}. {movie["title"]}")            
            # print(f"matches: {(movieIndex.get_documents(t))[0]}")
            # results.append((movieIndex.get_documents(t))[0])
            
            
    
    



    # for movie in movie_data["movies"]:
    #     new_movie_title = ""
    #     cleaned_movie_title=removePunctuation(movie["title"])
    #     for w in cleaned_movie_title.split():
    #         if  load_and_strip_stopwords(w) is not None:
    #             w = stems.get_stem_from_token(w)
    #             new_movie_title = f"{new_movie_title} {w}"
        
    #     new_movie_title = new_movie_title.lower()
                 
    #     for word in new_searched_movie.split():       
    #         if(re.search(rf"{word.lower()}\w*" ,new_movie_title)):                
    #             print(f"{title_count}. {movie["title"]}")
    #             title_count = title_count + 1
    
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
    
def load_movies():
    with open("data/movies.json") as file:
        movie_file = json.load(file)
    return movie_file["movies"] 