import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

class VectorDB:
    def __init__(self, top_k, search_method="cosine"):
        """
        ### parameters
        top_k: int - The number of nearest neighbors to return.

        search_method: str - *cosine*, *euclidean*, or *manhattan*.
        """

        self.collection = None
        self.top_k = top_k
        self.search_distance = search_method
    
    def add(self, vector:np.ndarray):
        """
        Adds a vector to the collection.
        
        *params*
        vector: np.ndarray - The vector to add.
        """
        assert vector.ndim == 2, "Vector must be a 2D array."
        if self.collection.size == 0:
            self.collection = vector.reshape(1, -1)
        else:
            self.collection = np.vstack((self.collection, vector.reshape(1, -1)))
    
    def search(self, query:np.ndarray):
        match self.search_distance:
            case "cosine":
                return cosine_similarity(self.collection, query.reshape(1, -1)).flatten().argsort()[:self.top_k]
            case "euclidean":
                return euclidean_distances(self.collection, query.reshape(1, -1)).flatten().argsort()[:self.top_k]
            case "manhattan":
                return manhattan_distances(self.collection, query.reshape(1, -1)).flatten().argsort()[:self.top_k]
            case _:
                raise ValueError("Unsupported search method. Use 'cosine', 'euclidean', or 'manhattan'.")
