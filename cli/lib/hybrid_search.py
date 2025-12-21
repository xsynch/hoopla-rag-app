import enum
import os
import json 

from .searchutils import get_gemini_response, load_movies, DEFAULT_K,DEFAULT_K_LIMIT, get_gemini_response_rerank, get_gemini_batch_rerank, get_gemini_evaluation
from .invertedindex import InvertedIndex
from .chunkedsemanticsearch import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder





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
        
        bm_score_list = []
        bm_score_standard = []
        ch_score_list = []
        ch_score_standard = []
        combined_results = {}
        
        bm25_results = self._bm25_search(query,limit*500)
        chunked_results = self.semantic_search.search_chunks(query,limit*500)
        # print(f"The length of bm25 results: {len(bm25_results)} and chunked results: {len(chunked_results)}")

        bm_score_list = [ x[2] for x in bm25_results]
        bm_norm_scores = self.normalize_scores(bm_score_list)
        for i,value in enumerate(bm25_results):
            bm_score_standard.append({
                "id": value[0],
                "title": value[1],
                "keyword_score":value[2],
                "normalized_score":bm_norm_scores[i],
            })

        ch_score_list = [chunk["score"] for chunk in chunked_results]
        ch_norm_scores = self.normalize_scores(ch_score_list)
        for i, value in enumerate(chunked_results):
            ch_score_standard.append({
                "id":value["id"],
                "title":value["title"],                
                "normalized_score": ch_norm_scores[i],
            })
        for ks in bm_score_standard:
            docid = ks["id"]
            document = self.semantic_search.document_map[docid]
            if docid not in combined_results:
                combined_results[docid] = {
                    "id":docid,
                    "title": ks["title"],
                    "document":document,
                    "bm25_score":0.0,
                    "semantic_score":0.0,
                    

                }
                combined_results[docid]["bm25_score"] = max(
                    ks["normalized_score"],combined_results[docid]["bm25_score"]
                )
        for chs in ch_score_standard:
            docid = chs["id"]
            document = self.semantic_search.document_map[docid]
            if docid not in combined_results:
                combined_results[docid] = {
                    "id":docid,
                    "title": chs["title"],
                    "document":document,
                    "semantic_score":0.0,
                    "bm25_score":0.0,
                    

                }
                combined_results[docid]["semantic_score"] = max(
                    chs["normalized_score"],combined_results[docid]["semantic_score"]
                )
            else:
                combined_results[docid]["semantic_score"] = chs["normalized_score"]
        
        for doc in combined_results:
            info = combined_results[doc]
            # print(f"{info["title"][:limit]}")
            combined_results[doc]["weighted_score"] = hybrid_score(info["bm25_score"],info["semantic_score"],alpha)
        # print(type(combined_results))
        combined_results = sorted(combined_results.items(),key=lambda x: x[1]["weighted_score"],reverse=True)
        # print(type(combined_results))
        return combined_results[:limit]
        # print(f"Number of keyword scores: {len(bm_norm_scores)} and Number of chunk scores: {len(ch_norm_scores)} ")
    def rrf_search(self, query, k=DEFAULT_K, limit=DEFAULT_K_LIMIT,debug=False):  
        if debug:
            print(f"Original Query: {query}")      
        
        bm_score_standard = []
        
        ch_score_standard = []
        combined_results = {}
        
        bm25_results = self._bm25_search(query,limit*500)
        chunked_results = self.semantic_search.search_chunks(query,limit*500)
        

        # bm_score_list = [ x[2] for x in bm25_results]
        # bm_norm_scores = self.normalize_scores(bm_score_list)
        for i,value in enumerate(bm25_results):
            bm_score_standard.append({
                "id": value[0],
                "title": value[1],
                "bm25_rank":i,
                "rrf_score": rrf_score(i,k),
                
            })

        # ch_score_list = [chunk["score"] for chunk in chunked_results]
        # ch_norm_scores = self.normalize_scores(ch_score_list)
        for i, value in enumerate(chunked_results):
            ch_score_standard.append({
                "id":value["id"],
                "title":value["title"],                
                "semantic_rank": i,
                "rrf_score": rrf_score(i,k)
            })
        for i,ks in enumerate(bm_score_standard):
            docid = ks["id"]
            document = self.semantic_search.document_map[docid]
            if docid not in combined_results:
                combined_results[docid] = {
                    "id":docid,
                    "title": ks["title"],
                    "document":document,
                    "bm25_rank":ks["bm25_rank"],
                    "rrf_score": ks["rrf_score"],
                    

                }
                # combined_results[docid]["bm25_score"] = max(
                #     ks["normalized_score"],combined_results[docid]["bm25_score"]
                # )
        for i,chs in enumerate(ch_score_standard):
            docid = chs["id"]
            document = self.semantic_search.document_map[docid]
            if docid not in combined_results:
                combined_results[docid] = {
                    "id":docid,
                    "title": chs["title"],
                    "document":document,
                    "semantic_rank":chs["semantic_rank"],
                    "rrf_score": chs["rrf_score"],
                    "bm25_rank":0,
                    

                }

            else:
                combined_results[docid]["rrf_score"] = chs["rrf_score"] + combined_results[docid]["rrf_score"]
                combined_results[docid]["semantic_rank"] = chs["semantic_rank"]
        
        # for doc in combined_results:
        #     info = combined_results[doc]
        #     # print(f"{info["title"][:limit]}")
        #     combined_results[doc]["weighted_score"] = hybrid_score(info["bm25_score"],info["semantic_score"],alpha)
        # print(type(combined_results))
        combined_results = sorted(combined_results.items(),key=lambda x: x[1]["rrf_score"],reverse=True)
        print("RRF top titles:", [r[1]["title"] for r in combined_results[:5]])
        # print(type(combined_results))
        return combined_results[:limit]
    
    def get_semantic_keyword_results(keyword_results,semantic_results):
        combined_results = []
        return combined_results

    

    
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
    results = hybrid_searcher.weighted_search(query,alpha,limit)
    # print(results)
    for i in range(len(results)):
        title = results[i][1]["title"]
        hybrid_score = results[i][1]["weighted_score"]
        description = results[i][1]["document"]["description"][:100]
        bm25_score = results[i][1]["bm25_score"]
        semantic_score = results[i][1]["semantic_score"]
        print(f"{i+1}. {title}\nHybrid Score: {hybrid_score: .3f}\nBM25: {bm25_score: .3f} Semantic: {semantic_score: .3f}\n{description}")

def hybrid_score(bm25_score, semantic_score, alpha=0.5):    
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def get_rrf_search(query,k,limit,evaluate=None,rerank_method=None,debug=False):
    final_results = []

    if rerank_method:
        limit = limit * 5
    documents = load_movies()
    hybrid_searcher = HybridSearch(documents)
    results = hybrid_searcher.rrf_search(query,k,limit)   
    if rerank_method == "individual":
        print(f"Reranking the results")
        for i in range(len(results)):
            rerank_score = float(get_gemini_response_rerank(query,results[i][1]["document"]))        
            results[i][1]["rerank_score"] = rerank_score
        results  = sorted(results,key=lambda x: x[1]["rerank_score"],reverse=True )[:limit//5]        
        
    elif rerank_method == "batch":   
        doc_list = []          
        for i in range(len(results)):
            doc_list.append(results[i][1]["document"])
        batch_results = json.loads(get_batch_rerank(query,doc_list))
        # print(f"Batch Results: {batch_results} json? {json.loads(batch_results)}")
        # return 
        for i in range(len(results)):
            if results[i][1]["document"]["id"] in batch_results:
                print(f"{results[i][1]["document"]["id"]} is being updated")
                results[i][1]["rerank_score"] = batch_results.index(results[i][1]["document"]["id"]) +1           
        
        results  = sorted(results,key=lambda x: x[1]["rerank_score"],reverse=False )[:limit//5]  
    elif rerank_method == "cross_encoder":
        doc_list = []          
        for i in range(len(results)):
            doc_list.append(results[i][1]["document"])
        scores = get_crossencoder_rerank(query,doc_list)
        for i in range(len(scores)):
            results[i][1]["cross_encoder_score"] = scores[i]
        results  = sorted(results,key=lambda x: x[1]["cross_encoder_score"],reverse=True )[:limit//5]
    
    for i in range(len(results)):
        title = results[i][1]["title"]
        rrf_score = results[i][1]["rrf_score"]
        description = results[i][1]["document"]["description"][:100]
        bm25_rank = results[i][1]["bm25_rank"]
        semantic_rank = results[i][1]["semantic_rank"]
        rerank_score = results[i][1].get("rerank_score","")
        cross_encoder_score = results[i][1].get("cross_encoder_score","")
        final_results.append(f"{i+1}. {title}\nRRF Score: {rrf_score: .3f}\nBM25 Rank: {bm25_rank} Semantic Rank: {semantic_rank}")
        if rerank_method is None:
            print(f"{i+1}. {title}\nRRF Score: {rrf_score: .3f}\nBM25 Rank: {bm25_rank} Semantic Rank: {semantic_rank}\n{description}")
        elif rerank_method == "indivicual" or rerank_method == "batch":
            print(f"{i+1}. {title}\nRerank Score: {rerank_score: .3f}\nRRF Score: {rrf_score: .3f}\nBM25 Rank: {bm25_rank} Semantic Rank: {semantic_rank}\n{description}")
        elif rerank_method == "cross_encoder":
            print(f"{i+1}. {title}\nCross Encoder Score: {cross_encoder_score: .3f}\nRRF Score: {rrf_score: .3f}\nBM25 Rank: {bm25_rank} Semantic Rank: {semantic_rank}\n{description}")
    if evaluate:
        print(f"Evaluating the results and ranking them for '{query}'")        
        responses = get_gemini_evaluation(query, final_results)
        
        # print(f"{responses} of type {type(responses)}")
        # print(f"Length of final_results: {len(final_results)} Length of responses: {len(responses)}: ")
        for  i in range(len(responses)):
            movie_title = final_results[i].split("\n")[0]
            # print(f"{movie_title}")
            print(f"{movie_title} {responses[i]}/3")


def get_batch_rerank(query,documents):
    batch_results = get_gemini_batch_rerank(query,documents)
    return batch_results

def get_crossencoder_rerank(query,documents)->list:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
    

    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    return scores