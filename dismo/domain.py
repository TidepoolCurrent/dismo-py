"""
Domain - Gower distance species distribution model.

Domain calculates suitability based on Gower distance to the 
nearest presence point in environmental space.

Reference:
    Carpenter, G., A.N. Gillison, and J. Winter. 1993. 
    DOMAIN: a flexible modelling procedure for mapping potential 
    distributions of plants and animals. Biodiversity and Conservation 2:667-680.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def gower_distance(x: NDArray, y: NDArray, ranges: NDArray) -> float:
    """
    Calculate Gower distance between two points.
    
    Gower distance normalizes each variable by its range,
    making variables comparable regardless of scale.
    
    Parameters
    ----------
    x, y : ndarray
        Two points to compare (1D arrays)
    ranges : ndarray
        Range (max - min) for each variable
        
    Returns
    -------
    float
        Gower distance (0 = identical, 1 = maximally different)
    """
    # Avoid division by zero
    safe_ranges = np.where(ranges == 0, 1, ranges)
    diffs = np.abs(x - y) / safe_ranges
    return np.mean(diffs)


class Domain:
    """
    Domain model using Gower distance.
    
    Domain predicts suitability based on environmental similarity
    to presence locations. For each prediction point, it calculates
    the minimum Gower distance to any presence point. Lower distance
    means higher suitability.
    
    Parameters
    ----------
    threshold : float, optional
        Maximum Gower distance considered suitable (default 0.05).
        Points with distance > threshold get suitability 0.
        
    Attributes
    ----------
    presence_data_ : ndarray
        Environmental values at presence locations
    ranges_ : ndarray
        Range (max - min) for each variable
    variables_ : list
        Names of environmental variables
        
    Examples
    --------
    >>> from dismo import Domain
    >>> import numpy as np
    >>> 
    >>> # Training data (presence locations)
    >>> presence = np.array([
    ...     [15, 800],   # temp, precip
    ...     [16, 850],
    ...     [14, 780]
    ... ])
    >>> 
    >>> model = Domain()
    >>> model.fit(presence)
    >>> 
    >>> # Predict at new locations
    >>> new_sites = np.array([
    ...     [15.5, 820],  # Similar to presences
    ...     [25, 500]     # Very different
    ... ])
    >>> predictions = model.predict(new_sites)
    
    Notes
    -----
    The Domain model was one of the earliest SDM methods (Carpenter et al. 1993).
    It's simple but effective for presence-only data when you expect
    the species to occur in environmentally similar locations.
    
    Unlike Bioclim (which uses envelopes), Domain considers the 
    actual proximity to known presences in environmental space.
    """
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
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
        
        self._fitted = True
        return self
    
    def predict(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Predict habitat suitability for new locations.
        
        Suitability is based on minimum Gower distance to any
        presence point. Distance is converted to suitability as:
        
            suitability = max(0, 1 - distance/threshold)
        
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
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Find minimum Gower distance to any presence
            min_dist = float('inf')
            for pres in self.presence_data_:
                dist = gower_distance(X[i], pres, self.ranges_)
                if dist < min_dist:
                    min_dist = dist
            
            # Convert distance to suitability
            # Closer = higher suitability
            if min_dist <= self.threshold:
                predictions[i] = 1 - (min_dist / self.threshold)
            else:
                predictions[i] = 0
        
        return predictions
    
    def predict_distance(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Return raw Gower distances instead of suitability.
        
        Useful for analysis or custom thresholding.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at prediction locations.
            
        Returns
        -------
        distances : ndarray
            Minimum Gower distance to any presence point.
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
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            min_dist = float('inf')
            for pres in self.presence_data_:
                dist = gower_distance(X[i], pres, self.ranges_)
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist
        
        return distances
