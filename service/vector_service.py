from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from fastapi import HTTPException
from PIL import Image
import numpy as np, os
import logging
import torch
import io
# DEVICE = "cude" if torch.cuda.is_available() else "cpu"
vectors = {}
# for img_path in os.listdir("images/"):
#     emb = model.encode(Image.open(f"images/{img_path}"))
#     vectors[img_path] = emb
np.save("embeddings.npy", vectors)
class VectorService:
    def __init__(self):
        logging.info("Model is loading...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logging.info("model load successfully")

    def image_to_vector(self, image_data: bytes) -> list[float]:
        """convert image bytes to feature vector"""
        try: 
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            #process image with clip processor
            inputs = self.processor(images=image, return_tensors="pt")

            #move inputs to the same device as model
            inputs = { k: v.to(self.device) for k, v in inputs.items() }

            #get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            #convert to numpy array and normalize
            vector = image_features.cpu().numpy()[0]
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()
        except Exception as e:
            logging.error(f"error processing image with CLIP: {str(e)}")
            raise HTTPException(status_code=400, detail=f"image processing error {str(e)}")

vector_service = VectorService()
