import re 
import string
from time import sleep 
from .invertedindex import InvertedIndex
from .stems import *
import json
import os 

from dotenv import load_dotenv
from google import genai 


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

DEFAULT_K=60
DEFAULT_K_LIMIT=5



STOPWORD_FILE = "data/stopwords.txt"

CACHE_DIR = "cache"
DATA_DIR = "data"
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

GOLDEN_DATASET = os.path.join(DATA_DIR,"golden_dataset.json")

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

def get_gemini_response(method,query,formatted_results=None):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = ""
    print(f"Using key {api_key[:6]}...")
    match method:
        case "spelling":
            prompt = get_spelling_prompt(query) 
        case "rewrite":
            prompt = get_rewrite_prompt(query)
        case "expand":
            prompt = get_expand_prompt(query)  
        # case "rank_relevance":
        #     # sleep(180)        
        #     prompt = get_relevancerank_prompt(query,formatted_results=formatted_results)
        #     print(f"{prompt}")
        #     return 


    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=prompt)

    
    if response.text != query:
        print( f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")
    return response.text


def get_gemini_response_rerank(query,document):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # print(f"Reranking document: {document}")
    rerank_prompt = get_rerank_results(query,document)
    # print(f"Rerank Prompt: {rerank_prompt}")


    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=rerank_prompt)
    sleep(3)
    # print(f"Response: {response}")
    return response.text


def get_gemini_batch_rerank(query,doc_list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    
    rerank_prompt = get_batch_rerank_prompt(query,doc_list)
    


    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=rerank_prompt)
    
    # print(f"Response: {response}")
    #the response is not coming back with valid json so I need to strip ```json from the front and ``` from the back
    response_result = response.text.lstrip("```json").rstrip("```")
    return response_result

def get_gemini_evaluation(query,formatted_results):
    
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    
    
    
    evaluation_prompt = get_relevancerank_prompt(query,formatted_results)



    client = genai.Client(api_key=api_key)
    # print(f"{evaluation_prompt}")
    
    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=evaluation_prompt)
        

    return json.loads(response.text)



def get_rewrite_prompt(query):
    return(f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""   ) 
    
def get_spelling_prompt(query):
    return f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

def get_expand_prompt(query):
    return f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

def get_rerank_results(query,doc):
    prompt =  f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    return prompt


def get_batch_rerank_prompt(query,document_list):
   return  f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{document_list}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. Do not wrap the list in ```json For example:

[75, 12, 34, 2, 1]
"""

def get_relevancerank_prompt(query, formatted_results):
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""