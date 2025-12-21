import os 
import json 
from dotenv import load_dotenv
from google import genai 

from .searchutils import get_gemini_response, load_movies, DEFAULT_K,DEFAULT_K_LIMIT, get_gemini_response_rerank, get_gemini_batch_rerank, get_gemini_evaluation

from .hybrid_search import HybridSearch


def get_augmented_results(query):
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query,limit=5)
    document_list = []
    print(f"Search Results:")
    for i in range(len(results)):
        document_list.append(results[i][1]["document"])
        print(f"""
    - {results[i][1]["document"]["title"]}""")
    prompt = return_rag_prompt(query,document_list)
    rag_response = get_rag_response(prompt)
    print(f"The response is: {rag_response}")


def get_rag_response(prompt):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")    

    client = genai.Client(api_key=api_key)    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)        
    return response.text

def return_rag_prompt(query, documents):
    return  f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{documents}

Provide a comprehensive answer that addresses the query:"""

def get_llm_summary(query,limit=5):
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query,limit)
    document_list = []
    print(f"Search Results:")
    for i in range(len(results)):
        document_list.append(results[i][1]["document"]["title"])
        print(f"""
    - {results[i][1]["document"]["title"]}""")

    summary = get_llm_data(query,document_list)
    print(f"\nLLM Summary:\n{summary}")
        
def get_llm_data(query,results):
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")    

    client = genai.Client(api_key=api_key)    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)        
    return response.text



def get_llm_citations(query,limit=5):
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query,limit)
    document_list = []
    print(f"Search Results:")
    for i in range(len(results)):
        document_list.append(results[i][1]["document"])
        print(f"""
    - {results[i][1]["document"]["title"]}""")
    # summary = get_results_citations(query,document_list)
    # print(f"LLM Answer:\n{summary}")


def get_results_citations(query,documents):
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")    

    client = genai.Client(api_key=api_key)    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)  
    # print(response)      
    return response.text

    