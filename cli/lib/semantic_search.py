import re
from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import json 

EMBEDDINGS_CACHE="movie_embeddings.npy"
CACHE_DIR="cache"

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None 
        self.documents = None 
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR,EMBEDDINGS_CACHE)
    
    def generate_embedding(self,text):
        if len(text) == 0 or text.isspace():
            raise ValueError("Please provide text")
        embeddings = self.model.encode([text])
        return embeddings[0]
    
    def build_embeddings(self, documents):
        doc_string = []
        if self.documents is None:
            self.documents = documents 
            for m in documents["movies"]:                          
                self.document_map[m["id"]] = m
        for key,val in self.document_map.items():
            doc_string.append(f"{val["title"]}: {val["description"]}")
        self.embeddings = self.model.encode(doc_string, show_progress_bar=True)
        np.save(self.embeddings_path,self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):        
        if self.documents is None:
            self.documents = documents 
            for m in documents["movies"]:                           
                self.document_map[m["id"]] = m
        if not os.path.isfile(self.embeddings_path):
            self.embeddings = self.build_embeddings(documents)
        else:
            self.embeddings = np.load(self.embeddings_path)
        # print(f"Length of embedings: {len(self.embeddings)} and documents {len(self.documents["movies"])}")
        if len(self.documents["movies"]) == len(self.embeddings):
            return self.embeddings
        
    def search(self, query, limit):
        
        similarity_score = []
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding_query = self.generate_embedding(query)
        
        for index, movie in enumerate(self.documents["movies"]):            
            # print(embed)
            similarity_score.append((cosine_similarity(embedding_query,self.embeddings[index]),movie))
            # print(f"{similarity_score}")
            # if len(similarity_score) > 0:
            #     return
        sorted_similarities = sorted(similarity_score,key=lambda x: x[0], reverse=True)
        return sorted_similarities[:limit]
        
                   

def verify_model():
    semantic_search = SemanticSearch() 
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movie_file = ""
    with open("data/movies.json") as file:
        movie_file = json.load(file)

    embeddings = semantic_search.load_or_create_embeddings(movie_file)
    print(f"Number of docs:   {len(movie_file)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query,limit=5):
    movie_file = ""
    semantic_search = SemanticSearch()
    with open("data/movies.json") as file:
        movie_file = json.load(file)

    semantic_search.load_or_create_embeddings(movie_file)
    results = semantic_search.search(query,limit)
    for result in results:
        print(f"{result[1]['title']}: (score: {result[0]})\n{result[1]['description']}")

def load_movies():
    with open("data/movies.json") as file:
        movie_file = json.load(file)
    return movie_file["movies"] 

def chunk_text(text:str,chunksize:int, overlapsize):
    split_data = text.split()
    
    chunked_text = chunk_data(text,chunksize,overlapsize)
    # idx = 0
    # while idx < len(split_data):
    #     if overlapsize > 0 and idx != 0:
    #         chunked_text.append(split_data[abs(idx - overlapsize):idx + chunksize])
    #     else:
    #         chunked_text.append(split_data[idx:idx + chunksize])
    #     idx = idx + chunksize
    # for index  in range(len(split_data)):
    #     if(index * chunksize) > len(split_data):
    #         break        
    #     if index == 0:
    #         chunked_text.append(split_data[:chunksize])            
    #     else:
    #         chunked_text.append(split_data[index * chunksize:(index * chunksize) +chunksize])


    print(f"Chunking {len(text)} characters")
    for i in range(len(chunked_text)):
        sentence = ' '.join(chunked_text[i])
        if len(sentence) > 0:
            print(f"{i+1}. {' '.join(chunked_text[i])}")


def chunk_data(text:str, chunksize:int, overlapsize:int):
    split_data = text.split()
    
    chunked_text = []
    idx = 0
    while idx < len(split_data):
        if overlapsize > 0 and idx != 0:
            chunked_text.append(split_data[abs(idx - overlapsize):idx + chunksize])
        else:
            chunked_text.append(split_data[idx:idx + chunksize])
        idx = idx + chunksize 
    return chunked_text

def chunk_sentences(text:str, chunksize:int, overlapsize:int):
    sentences = re.split(r"(?<=[.!?])\s+",text)
    # print(f"Sentences: {sentences}")
    chunked_text = []
    idx = 0
    while idx < len(sentences):
        chunk_sentences = sentences[idx: idx + chunksize]
        if chunked_text and len(chunk_sentences) <= overlapsize:
            break
        # if overlapsize > 0 and idx != 0:
        #     chunked_text.append(sentences[abs(idx - overlapsize):idx + chunksize])
        # else:
        #     chunked_text.append(sentences[idx:idx + chunksize])
        chunked_text.append(chunk_sentences)
        idx = idx + chunksize - overlapsize
    return chunked_text


def semantic_chunk(text:str, max_chunk_size:int, overlap:int):
    print(f"Semantically chunking {len(text)} characters")
    chunked_sentences = chunk_sentences(text,max_chunk_size,overlap)
    for i in range(len(chunked_sentences)):
        sentence = ' '.join(chunked_sentences[i])
        if len(sentence) > 0:
            print(f"{i+1}. {' '.join(chunked_sentences[i])}")