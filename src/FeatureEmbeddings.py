import torch
import torch.nn.functional as F
from PIL import Image

class FeatureEmbeddings:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
    
    def get_embeddings(self, Image_patch, device):
        """
        Get embeddings for the given features using the model.
        
        :param features: List of features to get embeddings for.
        :return: List of embeddings corresponding to the features.
        """
        inputs = self.feature_extractor(images=Image_patch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output[0]  # or mean pooling
            return F.normalize(embedding, dim=0).cpu()