from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from PIL import Image

from dotenv import load_dotenv
import os


from transformers import ViTModel, ViTFeatureExtractor
import torch
import torch.nn.functional as F


app = FastAPI()
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
load_dotenv()

PINECONE_API_KEY = os.getenv('API_KEY')
INDEX = os.getenv('INDEX_NAME')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX)

# Input schema
class ImageRequest(BaseModel):
    id:int
    image:Image

class GetTopKSimiliar(BaseModel):
    topk:int
    image:Image

def get_embdeddigs(Image_patch):
    inputs = feature_extractor(images=Image_patch, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output[0]
        return F.normalize(embeddings, dim=0)

@app.get("/")
def ItsWorking():
    return {"message":"API is running"}

@app.post("/push")
def push_embeddings(request:ImageRequest):
    try:
        Image_patch = request.image
        embeddigs = get_embdeddigs(Image_patch)

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/get")
def get_top_similiar(request:GetTopKSimiliar):
    try:
        Image_patch = request.image
        embeddigs = get_embdeddigs(Image_patch)
    
    except Exception as e:
        return {"error":str(e)}