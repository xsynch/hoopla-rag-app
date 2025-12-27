import os 
import json 
from dotenv import load_dotenv
from google import genai 
import mimetypes
import types

from .searchutils import get_gemini_response, load_movies, DEFAULT_K,DEFAULT_K_LIMIT, get_gemini_response_rerank, get_gemini_batch_rerank, get_gemini_evaluation

from .hybrid_search import HybridSearch

def get_image_results(image,query):
    mime = ""
    image_contents = ""
    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    
    


    if os.path.exists(image):
        mime,_ = mimetypes.guess_type(image)
        mime = mime or "image/jpeg"
        print(f"Mime type: {mime}")        
        with open(image,"rb") as file:
            image_contents = file.read()        
    else:
        print(f"{image} not found")
        return
    parts = [prompt, genai.types.Part.from_bytes(data=image_contents, mime_type=mime),query ]
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")    

    client = genai.Client(api_key=api_key)    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=parts)        
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")



