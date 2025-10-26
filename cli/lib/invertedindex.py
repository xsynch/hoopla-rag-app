from ast import mod
import json 
import pickle 
import os
from .cleantext import *
from collections import Counter
from .stems import *


CACHE_DIR ="cache"
CACHE_INDEX_FILE = "index.pkl"
CACHE_DOCMAP_FILE = "docmap.pkl"
CACHE_FREQ_FILE = "term_frequencies.pkl"

class InvertedIndex():
    index = {}
    docmap:dict[str,str] = {}

    def __init__(self):
        self.docmap = {}
        self.index = {}
        self.term_frequencies:dict[str,Counter] = {}
    
    def __add_document(self, doc_id:str, text:str):
        text = text.lower()
        tokens = text.split()
        
        doc_id = str(doc_id)

        for word in tokens:
            word = removePunctuation(word)                        
            word = get_stem_from_token(word)
            if word not in self.index.keys():                
                self.index[word] = {doc_id}                
            else:
                self.index[word].add(doc_id)
                
            if doc_id not in self.term_frequencies.keys():                
                self.term_frequencies[doc_id] = Counter([word])                
            else:
                self.term_frequencies[doc_id].update([word])        


    def get_documents(self, term:str):        
        return sorted(list(self.index[term.lower()]))
        
    
    def build(self, movie_file):
        with open(movie_file) as f:
            movie_data = json.load(f)
        
        for movie in movie_data["movies"]:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie



    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/index.pkl", mode="wb",) as filename:
            pickle.dump(self.index,filename)
        with open("cache/docmap.pkl", mode="wb") as filename:
            pickle.dump(self.docmap,filename)
        with open(f"{CACHE_DIR}/{CACHE_FREQ_FILE}", mode="wb") as filename:
            pickle.dump(self.term_frequencies,filename)

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
        
    def get_tf(self,doc_id: str, term: str) -> int:
        tokens = term.split()
        if len(tokens) > 1:
            raise Exception(f"Searched for term should be one word")
        # print(f"Searching for {term} in {doc_id}")
        if doc_id not in self.term_frequencies.keys():
            return 0
        items = self.term_frequencies[doc_id]        
        term_frequency = items[term]
        
        

        return term_frequency
            

    


