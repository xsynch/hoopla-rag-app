import os 
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

from lib.searchutils import load_movies

# from transformers import AutoModel

# # Get a model
# model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)

# # Encode text
# text_embeddings = model.encode_text(["a photo of a cat", "a dog playing fetch"])

# # Encode images
# image = Image.open("cat.jpg")
# image_embeddings = model.encode_image([image])

# # Calculate similarity
# similarity = torch.cosine_similarity(text_embeddings, image_embeddings)


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32",documents=None):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None 
        self.documents = documents
        self._create_texts()

    def embed_image(self,image):
        image_data = ""
        if os.path.exists(image):
            image_data = Image.open(image)              
            self.embeddings = self.model.encode([image_data])
        else:
            return None 
        return self.embeddings[0]
    
    def _create_texts(self):
        # doc_string = []
                
        for val in self.documents:
            self.texts.append(f"{val["title"]}: {val["description"]}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def search_with_image(self,image_path):
        similarity_list = []
        image_embeddings = self.embed_image(image_path) 
        
            

def verify_image_embedding(image):
    mm_search = MultimodalSearch()    
    image_embeddings = mm_search.embed_image(image)
    if len(image_embeddings) >0:
        print(f"Embedding shape: {image_embeddings.shape[0]} dimensions")
    else:
        print("Error generating image embeddings")

def image_search_command(image_path):
    movies = load_movies()
    mm_search = MultimodalSearch(documents=movies)