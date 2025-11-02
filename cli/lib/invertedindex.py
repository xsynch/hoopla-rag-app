import json 
import pickle 
import os
import math 
from .cleantext import *
from collections import Counter, defaultdict
# from .stems import *
from nltk.stem import PorterStemmer
# from .searchutils import (
#     BM25_K1,
#     BM25_B

# )

BM25_K1 = 1.5
BM25_B = 0.75



CACHE_DIR ="cache"
CACHE_INDEX_FILE = "index.pkl"
CACHE_DOCMAP_FILE = "docmap.pkl"
CACHE_FREQ_FILE = "term_frequencies.pkl"
JSON_FILE = "data/movies.json"


class InvertedIndex():


    def __init__(self):
        self.docmap = {}
        self.index = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR,"doc_lengths.pkl")
    
    def __add_document(self, doc_id:int, text:str):
        tokens = tokenize_text(text)    
          
        # doc_id = str(doc_id)

        for word in set(tokens):                                      
            self.index[word].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)     
        self.doc_lengths[doc_id] = len(tokens)   
        


    def get_documents(self, term:str):  
        # print(f"Length of index is {len(self.index)}")   
        doc_ids = self.index.get(term, set())
        
        return sorted(list(doc_ids))
        
    
    def build(self, movie_file):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            movie_description = f"{movie["title"]} {movie["description"]}"
            self.__add_document(doc_id, movie_description)
            self.docmap[doc_id] = movie



    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/index.pkl", mode="wb",) as filename:
            pickle.dump(self.index,filename)
        with open("cache/docmap.pkl", mode="wb") as filename:
            pickle.dump(self.docmap,filename)
        with open(f"{CACHE_DIR}/{CACHE_FREQ_FILE}", mode="wb") as filename:
            pickle.dump(self.term_frequencies,filename)
        with open(self.doc_lengths_path, "wb") as filename:
            pickle.dump(self.doc_lengths,filename)

    def load(self):
        docmap_file = f"{CACHE_DIR}/{CACHE_DOCMAP_FILE}"
        index_file = f"{CACHE_DIR}/{CACHE_INDEX_FILE}"
        term_frequencies_file = f"{CACHE_DIR}/{CACHE_FREQ_FILE}"
        if os.path.exists(docmap_file):
            with open(docmap_file, mode="rb") as docfile:
                self.docmap = pickle.load(docfile)
        else:
            raise FileNotFoundError(f"Error loading {docmap_file}")
        if os.path.exists(index_file):
            with open(index_file, mode="rb") as indexfile:
                self.index = pickle.load(indexfile)
        else:
            raise FileNotFoundError(f"Error loading {index_file}")
        if os.path.exists(term_frequencies_file):
            with open(term_frequencies_file, mode="rb") as freqfile:
                self.term_frequencies = pickle.load(freqfile)
        else:
            raise FileNotFoundError(f"Error loading {term_frequencies_file}")
        with open(self.doc_lengths_path,"rb") as f:
            self.doc_lengths = pickle.load(f)
        
    def get_tf(self,doc_id: int, term: str) -> int:        
        tokens = tokenize_text(term)        
        if len(tokens) > 1:
            raise Exception(f"Searched for term should be one word")
        token = tokens[0]
        # print(f"Searching for {token} in {doc_id}")
        if doc_id not in self.term_frequencies.keys():
            print(f"Docid {doc_id} not found within the frequencies dict")            
            return 0
        items = self.term_frequencies.get(doc_id)          
        term_frequency = items[token]
        return term_frequency
        
        

        return term_frequency
    
    def get_idf(self, term: str) -> float:
        if len(self.docmap) == 0:
            self.load()
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)        
        term_doc_count = len(self.index[token])        
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tfidf(self,docid:int, term:str):
        if len(self.docmap) == 0:
            self.load()       
        tf = self.get_tf(docid,term)
        idf = self.get_idf(term)
        # print(f"Tf: {tf} and idf: {idf}")
        tfidf = tf * idf 
        return tfidf
    def get_bm25_idf(self, term:str):
        if len(self.docmap) == 0:
            self.load()
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        bm25idf = math.log((len(self.docmap) - len(self.index[token]) + 0.5)/ ((len(self.index[token]) + 0.5) ) + 1)
        return bm25idf
    def get_bm25_tf(self, doc_id: int, term: str, b = BM25_B, k1: float = BM25_K1):
        
        if len(self.docmap) == 0:
            self.load()
        tf = self.get_tf(doc_id,term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id]/ self.__get_avg_doc_length())
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        # return (tf *(k1 + 1))/(tf + k1)
        return tf_component
    
    def __get_avg_doc_length(self):
        if len(self.docmap) == 0:
            self.load()
        doc_count = len(self.docmap)
        total_doc_lengths = 0
        for d in self.doc_lengths.values():
            total_doc_lengths = total_doc_lengths + d 
        if doc_count == 0:
            return 0.0
        return total_doc_lengths / doc_count
    
    def bm25(self, doc_id, term):
        # self.load()
        bm25tf = self.get_bm25_tf(doc_id,term)
        bm25idf = self.get_bm25_idf(term)
        return bm25tf * bm25idf
    def bm25_search(self, query, limit):
        if len(self.docmap) == 0:
            self.load()
        tokens = tokenize_text(query)
        # print(f"Tokens: {tokens}")
        scores_dictionary = defaultdict(float)
        for token in tokens:
            # print(f"Processing {token}")
            doc_ids = self.index.get(token,[])
            # print(f"Total doc ids for {token}: {len(doc_ids)}")
            
            for doc in doc_ids:
                # score_total = 0                
                scores_dictionary[doc] = scores_dictionary[doc] + self.bm25(doc,token)
        # print(f"Score Dictionary: {scores_dictionary}")
        sorted_scores = sorted(scores_dictionary.items(), key=lambda item: item[1], reverse=True)
        results = []
        for d in range(len(sorted_scores[:limit])):
            docid = sorted_scores[d][0]
            movie_title = self.docmap[sorted_scores[d][0]]["title"]
            score = f"{sorted_scores[d][1]:.2f}"
            results.append([docid,movie_title,score])

            
        return results
            


def load_movies() -> list[dict]:
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    return data["movies"]

# def tokenize_text(words:str) -> list[str]:
    
#     words = removePunctuation(words)
#     valid_tokens = []
#     sentence = words.split()
#     for word in sentence:
#         if word:
#             valid_tokens.append(word)
#     stop_words = load_stop_words()
#     filtered_words = []
#     for word in valid_tokens:
#         if word not in stop_words:
#             filtered_words.append(word)
#     stemmer = PorterStemmer()
#     stemmed_words = []
#     for word in filtered_words:
#         stemmed_words.append(stemmer.stem(word))
    
    
#     return stemmed_words

def tokenize_text(text: str) -> list[str]:
    text = removePunctuation(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stop_words()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words