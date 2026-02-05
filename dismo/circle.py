"""
Circle - Geographic distance species distribution model.

The Circle algorithm computes suitability based on geographic distance
from the nearest known occurrence. Simple but useful for range estimation.

Reference:
    Hijmans, R.J. and C.H. Graham, 2006. The ability of climate envelope 
    models to predict the effect of climate change on species distributions.
    Global Change Biology 12:2272-2281.
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


def haversine_distance(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
    radius: float = 6371.0
) -> float:
    """
    Calculate great-circle distance between two points.
    
    Parameters
    ----------
    lon1, lat1 : float
        First point (degrees)
    lon2, lat2 : float
        Second point (degrees)
    radius : float
        Earth radius in km (default 6371)
        
    Returns
    -------
    float
        Distance in km
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return radius * c


class Circle:
    """
    Circle model based on geographic distance.
    
    Predicts suitability based on distance to the nearest known 
    occurrence. Points within a threshold distance are considered
    suitable, with suitability decreasing linearly with distance.
    
    Parameters
    ----------
    threshold : float
        Maximum distance (km) considered suitable. Default uses
        the maximum inter-point distance in training data.
    use_haversine : bool
        If True, use great-circle distance. If False, use Euclidean
        distance in degrees (faster but less accurate).
        
    Attributes
    ----------
    presence_coords_ : ndarray
        Coordinates (lon, lat) of presence locations
    threshold_ : float
        Distance threshold used for predictions
        
    Examples
    --------
    >>> from dismo import Circle
    >>> import numpy as np
    >>> 
    >>> # Presence locations (lon, lat)
    >>> presence = np.array([
    ...     [-122.4, 37.8],  # San Francisco
    ...     [-118.2, 34.1],  # Los Angeles
    ...     [-117.2, 32.7]   # San Diego
    ... ])
    >>> 
    >>> model = Circle(threshold=100)  # 100 km threshold
    >>> model.fit(presence)
    >>> 
    >>> # Test points
    >>> test = np.array([
    ...     [-121.0, 36.0],  # Near SF
    ...     [-110.0, 40.0]   # Utah (far)
    ... ])
    >>> predictions = model.predict(test)
    """
    
    def __init__(self, threshold: Optional[float] = None, use_haversine: bool = True):
        self.threshold = threshold
        self.use_haversine = use_haversine
        self.presence_coords_ = None
        self.threshold_ = None
        self._fitted = False
    
    def fit(self, X: Union['pd.DataFrame', NDArray]) -> 'Circle':
        """
        Fit the Circle model to presence coordinates.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Presence locations as (longitude, latitude) pairs.
            Shape (n_samples, 2).
            
        Returns
        -------
        self : Circle
            Fitted model
        """
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)
        
        if X.shape[1] != 2:
            raise ValueError("Input must have 2 columns (lon, lat)")
        
        self.presence_coords_ = X.copy()
        
        # Calculate threshold if not provided
        if self.threshold is None:
            # Use maximum inter-point distance
            max_dist = 0
            n = len(X)
            for i in range(n):
                for j in range(i + 1, n):
                    if self.use_haversine:
                        d = haversine_distance(X[i, 0], X[i, 1], X[j, 0], X[j, 1])
                    else:
                        d = np.sqrt((X[i, 0] - X[j, 0])**2 + (X[i, 1] - X[j, 1])**2)
                    max_dist = max(max_dist, d)
            self.threshold_ = max_dist
        else:
            self.threshold_ = self.threshold
        
        self._fitted = True
        return self
    
    def predict(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Predict habitat suitability for new locations.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Prediction locations as (longitude, latitude) pairs.
            
        Returns
        -------
        predictions : ndarray
            Suitability scores between 0 and 1.
            1 = at a presence location
            0 = beyond threshold distance
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            lon, lat = X[i, 0], X[i, 1]
            
            # Find minimum distance to any presence
            min_dist = float('inf')
            for pres in self.presence_coords_:
                if self.use_haversine:
                    d = haversine_distance(lon, lat, pres[0], pres[1])
                else:
                    d = np.sqrt((lon - pres[0])**2 + (lat - pres[1])**2)
                min_dist = min(min_dist, d)
            
            # Convert to suitability (linear decay)
            if min_dist >= self.threshold_:
                predictions[i] = 0
            else:
                predictions[i] = 1 - (min_dist / self.threshold_)
        
        return predictions
    
    def predict_distance(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Return minimum distance to any presence location.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Prediction locations as (longitude, latitude) pairs.
            
        Returns
        -------
        distances : ndarray
            Distance in km (if use_haversine) or degrees.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)
        
        n_samples = X.shape[0]
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            lon, lat = X[i, 0], X[i, 1]
            
            min_dist = float('inf')
            for pres in self.presence_coords_:
                if self.use_haversine:
                    d = haversine_distance(lon, lat, pres[0], pres[1])
                else:
                    d = np.sqrt((lon - pres[0])**2 + (lat - pres[1])**2)
                min_dist = min(min_dist, d)
            
            distances[i] = min_dist
        
        return distances
