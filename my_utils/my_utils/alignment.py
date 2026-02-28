import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau


def compute_RSA(X,Y, dist_metric='correlation', corr_metric='pearson'):
    # X and Y are (n_samples, n_features) matrices of activations for two systems

    # Preprocess X and Y: zero-mean and unit norm (does not matter for corr dist)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    
    if dist_metric=='correlation':
      X_dist=1-np.corrcoef(X)
      Y_dist=1-np.corrcoef(Y)
    elif dist_metric=='euclidean':
      X_dist=squareform(pdist(X, metric='euclidean'))
      Y_dist=squareform(pdist(Y, metric='euclidean'))

    X_dist_flat = X_dist[np.triu_indices(X_dist.shape[0], k=1)]
    Y_dist_flat = Y_dist[np.triu_indices(Y_dist.shape[0], k=1)]
    
    valid_indices = ~np.isnan(X_dist_flat) & ~np.isnan(Y_dist_flat)
    X_dist_flat = X_dist_flat[valid_indices]
    Y_dist_flat = Y_dist_flat[valid_indices]

    if len(X_dist_flat) < 2:
        rsa_corr= [np.nan]
        
    else:
      if corr_metric=='pearson':
        rsa_corr = pearsonr(X_dist_flat, Y_dist_flat)[0]
      elif corr_metric=='spearman':
        rsa_corr = spearmanr(X_dist_flat, Y_dist_flat)[0]
      elif corr_metric=='kendall':
        rsa_corr = kendalltau(X_dist_flat, Y_dist_flat)[0]
  
    return rsa_corr


def vectorized_correlation(X,Y):
    # compute correlation between columns of X and Y
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    numerator = X.T @ Y
    denominator = np.linalg.norm(X, axis=0)[:, None] * np.linalg.norm(Y, axis=0)[None, :] + 1e-8


    return numerator / denominator