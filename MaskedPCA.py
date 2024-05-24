from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class MaskedPCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, mask=None):  
        # mask should contain selected cols. Suppose it is boolean to avoid code overhead
        super().__init__()
        self.n_components = n_components
        self.mask = mask

    def fit(self, X, y = None):
        self.pca = PCA(n_components=self.n_components)
        mask = self.mask
        mask = self.mask if self.mask is not None else slice(None)
        self.pca.fit(X[:, mask])
        self.n_samples_ = self.pca.n_samples_
        self.components_ = self.pca.components_
        self.n_components_ = self.pca.n_components_
        self.explained_variance_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.singular_values_ = self.pca.singular_values_
        return self

    def transform(self, X, y = None):
        mask = self.mask if self.mask is not None else slice(None)
        pca_transformed = self.pca.transform(X[:, mask])
        if self.mask is not None:
            remaining_cols = X[:, ~mask]
            return np.hstack([remaining_cols, pca_transformed])
        else:
            return pca_transformed
        
        
        