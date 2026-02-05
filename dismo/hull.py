"""
Convex Hull - Geographic range species distribution model.

The Convex Hull algorithm defines the species range as the smallest
convex polygon containing all known occurrences.

This is a simple geometric approach often used for range mapping.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, List, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def convex_hull(points: NDArray) -> NDArray:
    """
    Compute convex hull of a set of points using Graham scan.
    
    Parameters
    ----------
    points : ndarray
        Points as (n, 2) array of (x, y) coordinates
        
    Returns
    -------
    hull : ndarray
        Vertices of convex hull in counter-clockwise order
    """
    points = np.asarray(points)
    n = len(points)
    
    if n < 3:
        return points
    
    # Find lowest point (and leftmost if tie)
    start_idx = 0
    for i in range(1, n):
        if points[i, 1] < points[start_idx, 1]:
            start_idx = i
        elif points[i, 1] == points[start_idx, 1] and points[i, 0] < points[start_idx, 0]:
            start_idx = i
    
    start = points[start_idx]
    
    # Sort points by polar angle with respect to start
    def polar_angle(p):
        dx = p[0] - start[0]
        dy = p[1] - start[1]
        return np.arctan2(dy, dx)
    
    # Get indices sorted by polar angle
    other_points = [i for i in range(n) if i != start_idx]
    other_points.sort(key=lambda i: (polar_angle(points[i]), 
                                      -np.sqrt((points[i, 0] - start[0])**2 + 
                                              (points[i, 1] - start[1])**2)))
    
    # Graham scan
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    hull = [start_idx]
    for i in other_points:
        while len(hull) > 1 and cross(points[hull[-2]], points[hull[-1]], points[i]) <= 0:
            hull.pop()
        hull.append(i)
    
    return points[hull]


def point_in_polygon(x: float, y: float, polygon: NDArray) -> bool:
    """
    Test if a point is inside a polygon using ray casting.
    
    Parameters
    ----------
    x, y : float
        Point coordinates
    polygon : ndarray
        Polygon vertices as (n, 2) array
        
    Returns
    -------
    bool
        True if point is inside polygon
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


class ConvexHull:
    """
    Convex Hull geographic range model.
    
    Defines suitable habitat as the convex hull containing all
    known occurrence points. Can optionally include a buffer zone.
    
    Parameters
    ----------
    buffer : float
        Buffer distance (in same units as coordinates) around hull.
        Default 0 (no buffer).
        
    Attributes
    ----------
    hull_vertices_ : ndarray
        Vertices of the convex hull
    presence_coords_ : ndarray
        Original presence coordinates
        
    Examples
    --------
    >>> from dismo import ConvexHull
    >>> import numpy as np
    >>> 
    >>> # Presence locations
    >>> presence = np.array([
    ...     [-122.4, 37.8],
    ...     [-118.2, 34.1],
    ...     [-121.0, 38.5],
    ...     [-119.0, 35.5]
    ... ])
    >>> 
    >>> model = ConvexHull()
    >>> model.fit(presence)
    >>> 
    >>> # Test points
    >>> test = np.array([
    ...     [-120.0, 36.0],  # Inside hull
    ...     [-110.0, 40.0]   # Outside hull
    ... ])
    >>> predictions = model.predict(test)
    """
    
    def __init__(self, buffer: float = 0):
        self.buffer = buffer
        self.hull_vertices_ = None
        self.presence_coords_ = None
        self._fitted = False
    
    def fit(self, X: Union['pd.DataFrame', NDArray]) -> 'ConvexHull':
        """
        Fit the ConvexHull model to presence coordinates.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Presence locations as (longitude, latitude) pairs.
            Shape (n_samples, 2). Requires at least 3 points.
            
        Returns
        -------
        self : ConvexHull
            Fitted model
        """
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)
        
        if X.shape[1] != 2:
            raise ValueError("Input must have 2 columns (lon, lat)")
        
        if len(X) < 3:
            raise ValueError("Need at least 3 points for convex hull")
        
        self.presence_coords_ = X.copy()
        self.hull_vertices_ = convex_hull(X)
        
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
            Binary suitability (1 = inside hull, 0 = outside).
            With buffer > 0, points near the hull get partial scores.
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
            x, y = X[i, 0], X[i, 1]
            
            if point_in_polygon(x, y, self.hull_vertices_):
                predictions[i] = 1.0
            elif self.buffer > 0:
                # Check distance to hull edges
                min_dist = self._distance_to_hull(x, y)
                if min_dist <= self.buffer:
                    predictions[i] = 1 - (min_dist / self.buffer)
        
        return predictions
    
    def _distance_to_hull(self, x: float, y: float) -> float:
        """Calculate minimum distance from point to hull edges."""
        min_dist = float('inf')
        n = len(self.hull_vertices_)
        
        for i in range(n):
            p1 = self.hull_vertices_[i]
            p2 = self.hull_vertices_[(i + 1) % n]
            
            # Distance from point to line segment
            dist = self._point_to_segment_distance(x, y, p1[0], p1[1], p2[0], p2[1])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(self, px: float, py: float,
                                   x1: float, y1: float,
                                   x2: float, y2: float) -> float:
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def get_hull_polygon(self) -> NDArray:
        """
        Get the convex hull vertices.
        
        Returns
        -------
        vertices : ndarray
            Hull vertices in order, shape (n, 2)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.hull_vertices_.copy()
