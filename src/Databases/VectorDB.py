import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances


class VectorDB:
    def __init__(self, top_k, meta_data:None,  search_method="cosine"):  # TODO INIT: JSON ?, Device ?
        """
        ### parameters
        top_k: int - The number of nearest neighbors to return.

        search_method: str - *cosine*, *euclidean*, or *manhattan*.
        """

        self.collection = np.empty((0, 0), dtype=np.float32) # TODO: variable dtype
        self.top_k = top_k
        self.search_distance = search_method
        self.have_meta_data = False
        if meta_data:
            self.have_meta_data = True
            self.meta_data = pd.DataFrame(meta_data)
            self.num_rows = 0
    
    def add(self, vector:np.ndarray, meta_data_vector:np.ndarray=None):
        """
        Adds a vector to the collection.
        
        *params*
        vector: np.ndarray - The vector to add.
        """
        assert vector.ndim == 2, "Vector must be a 2D array."   # TODO: vector.shape[1] == collecton.shape[1]
        if self.collection.size == 0:
            self.collection = vector.reshape(1, -1)
        else:
            self.collection = np.vstack((self.collection, vector.reshape(1, -1)))
        
        if self.have_meta_data:
            self.meta_data.loc[self.num_rows] = meta_data_vector    # TODO: assert about its 1D nature
            self.num_rows += 1
    
    def search(self, query:np.ndarray):
        # TODO: Check Query's Dimension and ndim
        similarity = None
        match self.search_distance:
            case "cosine":
                similarity =  cosine_similarity(self.collection, query).flatten().argsort()[:self.top_k]
            case "euclidean":
                similarity =  euclidean_distances(self.collection, query).flatten().argsort()[:self.top_k]
            case "manhattan":
                similarity =  manhattan_distances(self.collection, query).flatten().argsort()[:self.top_k]
            case _:
                raise ValueError("Unsupported search method. Use 'cosine', 'euclidean', or 'manhattan'.")

        return self.meta_data.iloc[similarity]