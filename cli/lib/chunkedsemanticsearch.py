from .semantic_search import SemanticSearch, chunk_sentences,cosine_similarity,semantic_chunk_2
from .searchutils import load_movies
import os 
import numpy as np
import json 

from .searchutils import(
     MAX_CHUNK_SIZE, OVERLAP,SCORE_PRECISION, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
     CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, DEFAULT_SEMANTIC_CHUNK_SIZE
)


CACHE_DIR = "cache"
CHUNK_METADATA_FILE_NAME="chunk_metadata.json"
CHUNK_EMBEDDINGS_FILE_NAME="chunk_embeddings.npy"

CHUNK_EMBEDDINGS_FILE = os.path.join(CACHE_DIR,CHUNK_EMBEDDINGS_FILE_NAME)
CHUNK_METADATA_FILE= os.path.join(CACHE_DIR,CHUNK_METADATA_FILE_NAME)

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk_2(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    # def build_chunk_embeddings(self, documents):
        
    #     all_chunks = []
    #     chunk_metadata_list = []
        
    #     self.documents = documents 
    #     self.document_map = {}
    #     for doc in documents:
    #         self.document_map[doc["id"]] = doc             
        
    #     for idx, m in enumerate(documents):      
                                
    #         text = m.get("description","")
    #         if not text.strip():
    #             continue 
            
            
            
    #         sentence_chunks = semantic_chunk_2(text,DEFAULT_CHUNK_SIZE,DEFAULT_CHUNK_OVERLAP)
            
    #         for i, sentence in enumerate(sentence_chunks):
    #             all_chunks.append(sentence)
    #             chunk_metadata_list.append( {"movie_idx" : idx,"chunk_idx":i,"total_chunks":len(sentence_chunks)})
        

    #     self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar=True)
    #     self.chunk_metadata = chunk_metadata_list
    #     np.save(CHUNK_EMBEDDINGS_FILE,self.chunk_embeddings)
    #     with open(CHUNK_METADATA_FILE,"w") as f:
    #         json.dump({"chunks": chunk_metadata_list, "total_chunks": len(all_chunks)}, f, indent=2)
    #     return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents 
        for m in documents:                          
            self.document_map[m["id"]] = m
        if not os.path.exists(CHUNK_EMBEDDINGS_FILE) or not os.path.exists(CHUNK_METADATA_FILE) :
            return self.build_chunk_embeddings(documents)
            
        else:
            
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_FILE)    
            
            with open(CHUNK_METADATA_FILE,"r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        chunk_scores = []
        movie_index_score = {}
        
        
        query_embeddings = self.generate_embedding(query)        
        for index, chunk_data in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embeddings,chunk_data)
            movie_index = self.chunk_metadata[index]["movie_idx"]            
            chunk_scores.append({"chunk_idx":index,"movie_idx":movie_index,"score":similarity_score})
        
        
        for chunk_score in chunk_scores:
            movie_index = chunk_score["movie_idx"]
            if movie_index not in movie_index_score or  chunk_score["score"] > movie_index_score[movie_index]:
                movie_index_score[movie_index] = chunk_score["score"]
        


        sorted_movie_score = sorted(movie_index_score.items(),key= lambda x: x[1],reverse=True)
        results = []
        for movie_idx,score in sorted_movie_score[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, SCORE_PRECISION),
                # "metadata": movie["metadata"] or {}
            }
            )

        return  results

def embed_chunks():
    documents = load_movies()
    ch_search = ChunkedSemanticSearch()
    embeddings = ch_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
        
        

def search_chunked(query,limit=5):
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    searcher.load_or_create_chunk_embeddings(movies)
    found_movies = searcher.search_chunks(query,limit)
    for i in range(len(found_movies)):
        TITLE = found_movies[i]["title"]
        SCORE = found_movies[i]["score"]
        DESCRIPTION = found_movies[i]["document"]
        print(f"\n{i}. {TITLE} (score: {SCORE:.4f})")
        print(f"   {DESCRIPTION}...")


