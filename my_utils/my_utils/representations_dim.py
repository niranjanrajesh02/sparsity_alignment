import numpy as np
from sklearn.decomposition import PCA


def compute_effective_dim(activations):
    # activations: numpy array of shape (num_samples, num_features)
    pca = PCA()
    pca.fit(activations)
    eigenspectrum = pca.explained_variance_ # eigenvalues of the covariance matrix
    effective_dim = (eigenspectrum.sum() ** 2) / (eigenspectrum ** 2).sum()
    return effective_dim


def compute_pca_dim(activations, variance_threshold=0.95):
    # activations: numpy array of shape (num_samples, num_features)
    pca = PCA()
    pca.fit(activations)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    pca_dim = np.searchsorted(cumulative_variance, variance_threshold) + 1 # +1 because searchsorted returns the index where the threshold would be inserted to maintain order
    return pca_dim

## Effective Dim (PCA) ##
# https://github.com/EricElmoznino/encoder_dimensionality/blob/main/custom_model_tools/eigenspectrum.py
# pca.fit(activations)
# eigenspectrum = pca.explained_variance_
# eigspec.sum() ** 2 / (eigspec**2).sum()



## Capacity Dim ##
# https://github.com/EricElmoznino/encoder_dimensionality/blob/main/lib/manifold_geometry.py


## Intrinsic Dim (2nn) ##

