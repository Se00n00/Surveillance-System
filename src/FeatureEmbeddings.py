class FeatureEmbeddings:
    def __init__(self, model):
        self.model = model
    
    def get_embeddings(self, features):
        """
        Get embeddings for the given features using the model.
        
        :param features: List of features to get embeddings for.
        :return: List of embeddings corresponding to the features.
        """
        return self.model.encode(features, convert_to_tensor=True)