import os

from .searchutils import load_movies
from .invertedindex import InvertedIndex
from .chunkedsemanticsearch import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        # if not os.path.exists(self.idx.index_path):
        if not os.path.exists(self.idx.doc_lengths_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        results = {}
        bm_score_list = []
        bm25_results = self._bm25_search(query,limit*500)
        chunked_results = self.semantic_search.search_chunks(query,limit*500)
        # print(f"The length of bm25 results: {len(bm25_results)} and chunked results: {len(chunked_results)}")
        for r in range(len(bm25_results)):
            bm_score = bm25_results[r][2]
            bm_score_list.append(bm_score)
        bm_norm_scores = self.normalize_scores(bm_score_list)
            

        

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
    def normalize_scores(self,scores_list):
        scores_floats = [float(x) for x in scores_list]
        min_score = min(scores_floats)
        max_score = max(scores_floats)
        min_max_denominator = max_score - min_score
        results_list = []
        if min_score == max_score:            
            for num in scores_floats:
                results_list.append(1.0)
        else:
            for num in scores_floats:
                results_list.append((num - min_score)/min_max_denominator)
        return results_list
    
    
    
    
def get_normalized_scores(score_list):
    documents = load_movies()
    hybrid_searcher = HybridSearch(documents)
    scores = hybrid_searcher.normalize_scores(score_list)
    for score in scores:
       print(f"* {score:.4f}") 

def get_results_weighted_scores(query,alpha,limit):
    documents = load_movies()
    hybrid_searcher = HybridSearch(documents)
    hybrid_searcher.weighted_search(query,alpha,limit)

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score