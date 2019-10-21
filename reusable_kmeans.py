# coding: utf-8
from faiss import Kmeans, Clustering, ClusteringParameters, vector_float_to_array


class ReusableKMeans(Kmeans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = None
    def train(self, x):
        if self.index is None:
            return super().train(x)
        else:
            n, d = x.shape
            assert d == self.d
            clus = Clustering(d, self.k, self.cp)
            clus.train(x, self.index)
            centroids = vector_float_to_array(clus.centroids)
            self.centroids = centroids.reshape(self.k, d)
            self.obj = vector_float_to_array(clus.obj)
            return self.obj[-1] if self.obj.size > 0 else 0.0
            
