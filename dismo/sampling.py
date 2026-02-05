"""
Background point sampling utilities.

SDMs need background/pseudo-absence points for model fitting.
These functions help generate them appropriately.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Optional, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def randomPoints(
    n: int,
    extent: Tuple[float, float, float, float],
    exclusion_buffer: Optional[NDArray] = None,
    buffer_distance: float = 0.0
) -> NDArray:
    """
    Generate random background points within an extent.
    
    Parameters
    ----------
    n : int
        Number of points to generate
    extent : tuple
        (xmin, xmax, ymin, ymax) bounding box
    exclusion_buffer : ndarray, optional
        Points to exclude (presence locations). Shape (n, 2).
    buffer_distance : float
        Minimum distance from exclusion points (in same units as extent)
        
    Returns
    -------
    points : ndarray
        Random points, shape (n, 2) as [x, y] coordinates
        
    Examples
    --------
    >>> extent = (-125, -115, 32, 42)  # California
    >>> bg_points = randomPoints(1000, extent)
    >>> print(bg_points.shape)
    (1000, 2)
    """
    xmin, xmax, ymin, ymax = extent
    
    if exclusion_buffer is None or buffer_distance == 0:
        # Simple random sampling
        x = np.random.uniform(xmin, xmax, n)
        y = np.random.uniform(ymin, ymax, n)
        return np.column_stack([x, y])
    
    # Sample with exclusion buffer
    points = []
    max_attempts = n * 10
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        
        # Check distance to exclusion points
        dists = np.sqrt((exclusion_buffer[:, 0] - x)**2 + 
                       (exclusion_buffer[:, 1] - y)**2)
        
        if np.all(dists >= buffer_distance):
            points.append([x, y])
        
        attempts += 1
    
    if len(points) < n:
        # Fill remaining with random points (no buffer check)
        remaining = n - len(points)
        x = np.random.uniform(xmin, xmax, remaining)
        y = np.random.uniform(ymin, ymax, remaining)
        points.extend(np.column_stack([x, y]).tolist())
    
    return np.array(points[:n])


def gridSample(
    extent: Tuple[float, float, float, float],
    resolution: float
) -> NDArray:
    """
    Generate a regular grid of points.
    
    Parameters
    ----------
    extent : tuple
        (xmin, xmax, ymin, ymax) bounding box
    resolution : float
        Grid cell size
        
    Returns
    -------
    points : ndarray
        Grid points, shape (n, 2) as [x, y] coordinates
    """
    xmin, xmax, ymin, ymax = extent
    
    x = np.arange(xmin + resolution/2, xmax, resolution)
    y = np.arange(ymin + resolution/2, ymax, resolution)
    
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def targetGroupBackground(
    presence: NDArray,
    all_occurrences: NDArray,
    n: int
) -> NDArray:
    """
    Sample background from target-group occurrences.
    
    Target-group background uses occurrences of related species
    (e.g., all species in a genus) to account for sampling bias.
    
    Parameters
    ----------
    presence : ndarray
        Focal species presence locations, shape (n_pres, 2)
    all_occurrences : ndarray
        All target-group occurrences, shape (n_all, 2)
    n : int
        Number of background points to sample
        
    Returns
    -------
    background : ndarray
        Sampled background points, shape (n, 2)
        
    Notes
    -----
    This method assumes sampling bias is similar across the target group,
    so using target-group locations as background helps control for it.
    
    Reference:
        Phillips et al. (2009). Sample selection bias and presence-only 
        distribution models. Ecological Applications 19:181-197.
    """
    # Remove focal species presences from pool
    # (Simple distance-based exclusion)
    mask = np.ones(len(all_occurrences), dtype=bool)
    
    for pres in presence:
        dists = np.sqrt((all_occurrences[:, 0] - pres[0])**2 + 
                       (all_occurrences[:, 1] - pres[1])**2)
        mask &= (dists > 1e-6)  # Exclude exact matches
    
    pool = all_occurrences[mask]
    
    if len(pool) < n:
        # Not enough points, return all with replacement
        indices = np.random.choice(len(pool), n, replace=True)
    else:
        indices = np.random.choice(len(pool), n, replace=False)
    
    return pool[indices]


def spatialThin(
    points: NDArray,
    min_distance: float
) -> NDArray:
    """
    Spatially thin points to reduce clustering.
    
    Removes points that are too close together, keeping a 
    roughly even spatial distribution.
    
    Parameters
    ----------
    points : ndarray
        Input points, shape (n, 2)
    min_distance : float
        Minimum distance between retained points
        
    Returns
    -------
    thinned : ndarray
        Thinned points, shape (m, 2) where m <= n
        
    Notes
    -----
    Uses a greedy algorithm: iterate through points in random order,
    keeping each point only if it's far enough from all kept points.
    """
    if len(points) == 0:
        return points
    
    # Random order to avoid spatial bias
    order = np.random.permutation(len(points))
    points = points[order]
    
    kept = [points[0]]
    
    for point in points[1:]:
        dists = np.sqrt((np.array(kept)[:, 0] - point[0])**2 + 
                       (np.array(kept)[:, 1] - point[1])**2)
        
        if np.all(dists >= min_distance):
            kept.append(point)
    
    return np.array(kept)
