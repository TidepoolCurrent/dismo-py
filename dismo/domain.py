"""
Domain - Gower distance species distribution model.

The Domain algorithm computes the Gower distance between environmental
variables at any location and those at any of the known locations of 
occurrence ('training sites').

This implementation matches R's dismo::domain algorithm.

Reference:
    Carpenter, G., A.N. Gillison, and J. Winter. 1993. 
    DOMAIN: a flexible modelling procedure for mapping potential 
    distributions of plants and animals. Biodiversity and Conservation 2:667-680.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


class Domain:
    """
    Domain model using Gower distance.
    
    Matches R's dismo::domain implementation.
    
    Algorithm (from R dismo):
    1. For each variable, compute normalized distance to each training point
    2. For each variable, take minimum distance across training points
    3. Take maximum distance across variables (limiting factor)
    4. Suitability = 1 - max_distance (truncated to [0, 1])
    
    Parameters
    ----------
    None (threshold parameter removed to match R)
        
    Attributes
    ----------
    presence_data_ : ndarray
        Environmental values at presence locations
    ranges_ : ndarray
        Range (max - min) for each variable
    variables_ : list
        Names of environmental variables
    """
    
    def __init__(self):
        self.presence_data_ = None
        self.ranges_ = None
        self.variables_ = None
        self._fitted = False
    
    def fit(self, X: Union['pd.DataFrame', NDArray]) -> 'Domain':
        """
        Fit the Domain model to presence data.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at presence locations.
            Shape (n_samples, n_variables).
            
        Returns
        -------
        self : Domain
            Fitted model
        """
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            self.variables_ = list(X.columns)
            X = X.values
        else:
            X = np.asarray(X)
            self.variables_ = [f'var{i}' for i in range(X.shape[1])]
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.presence_data_ = X.copy()
        
        # Calculate ranges for Gower distance normalization
        self.ranges_ = np.ptp(X, axis=0)  # max - min for each column
        # Avoid division by zero
        self.ranges_ = np.where(self.ranges_ == 0, 1, self.ranges_)
        
        self._fitted = True
        return self
    
    def predict(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Predict habitat suitability for new locations.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at prediction locations.
            Must have same variables as training data.
            
        Returns
        -------
        predictions : ndarray
            Suitability scores between 0 and 1.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X[self.variables_].values
        else:
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_vars = X.shape[1]
        n_train = len(self.presence_data_)
        
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            test_point = X[i]
            
            # For each variable, calculate MEAN normalized distance to all training points
            # Then convert to score: 1 - mean_distance
            scores_per_var = np.zeros(n_vars)
            
            for j in range(n_vars):
                # Normalized distances to all training points for this variable
                dists = np.abs(self.presence_data_[:, j] - test_point[j]) / self.ranges_[j]
                mean_dist = np.mean(dists)
                # Truncate distance to max 1
                mean_dist = min(mean_dist, 1.0)
                scores_per_var[j] = 1 - mean_dist
            
            # Minimum across variables (limiting factor)
            predictions[i] = np.min(scores_per_var)
        
        return predictions
    
    def predict_distance(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Return raw Gower distances instead of suitability.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at prediction locations.
            
        Returns
        -------
        distances : ndarray
            Maximum normalized distance to training points.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X[self.variables_].values
        else:
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_vars = X.shape[1]
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            test_point = X[i]
            mean_dist_per_var = np.zeros(n_vars)
            
            for j in range(n_vars):
                dists = np.abs(self.presence_data_[:, j] - test_point[j]) / self.ranges_[j]
                mean_dist_per_var[j] = np.mean(dists)
            
            distances[i] = np.max(mean_dist_per_var)
        
        return distances


# Keep the old gower_distance function for backwards compatibility
def gower_distance(x: NDArray, y: NDArray, ranges: NDArray) -> float:
    """
    Calculate Gower distance between two points.
    
    Gower distance normalizes each variable by its range,
    making variables comparable regardless of scale.
    """
    safe_ranges = np.where(ranges == 0, 1, ranges)
    diffs = np.abs(x - y) / safe_ranges
    return np.mean(diffs)
