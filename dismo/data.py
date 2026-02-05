"""
Data acquisition wrappers.

Provides simple interfaces to common biodiversity data sources.
Wraps existing Python packages where available.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, List, Union, Tuple


def gbif(
    species: str,
    limit: int = 500,
    country: Optional[str] = None,
    geometry: Optional[str] = None,
    has_coordinate: bool = True,
    **kwargs
) -> Dict:
    """
    Query GBIF for species occurrence records.
    
    This is a wrapper around pygbif. Install with: pip install pygbif
    
    Parameters
    ----------
    species : str
        Species name (e.g., "Pycnopodia helianthoides")
    limit : int
        Maximum records to return (default 500)
    country : str, optional
        Two-letter country code (e.g., "US", "CA")
    geometry : str, optional
        WKT geometry to search within
    has_coordinate : bool
        Only return georeferenced records (default True)
    **kwargs
        Additional arguments passed to pygbif.occurrences.search
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'coords': ndarray of (lon, lat) pairs
        - 'data': full response data
        - 'count': number of records
        
    Examples
    --------
    >>> result = gbif("Pinus ponderosa", limit=100, country="US")
    >>> print(f"Found {result['count']} records")
    >>> coords = result['coords']  # Use directly with SDM models
    
    Raises
    ------
    ImportError
        If pygbif is not installed
    """
    try:
        from pygbif import occurrences
    except ImportError:
        raise ImportError(
            "pygbif not installed. Install with: pip install pygbif"
        )
    
    # Build search parameters
    search_params = {
        'scientificName': species,
        'limit': limit,
        'hasCoordinate': has_coordinate,
    }
    
    if country:
        search_params['country'] = country
    if geometry:
        search_params['geometry'] = geometry
    
    search_params.update(kwargs)
    
    # Query GBIF
    response = occurrences.search(**search_params)
    
    # Extract coordinates
    records = response.get('results', [])
    coords = []
    
    for rec in records:
        lon = rec.get('decimalLongitude')
        lat = rec.get('decimalLatitude')
        if lon is not None and lat is not None:
            coords.append([lon, lat])
    
    return {
        'coords': np.array(coords) if coords else np.array([]).reshape(0, 2),
        'data': response,
        'count': len(coords),
    }


def inat(
    species: str,
    limit: int = 500,
    place_id: Optional[int] = None,
    quality_grade: str = "research",
    **kwargs
) -> Dict:
    """
    Query iNaturalist for species occurrence records.
    
    Uses the iNaturalist API directly (no external package needed).
    
    Parameters
    ----------
    species : str
        Species name (e.g., "Pycnopodia helianthoides")
    limit : int
        Maximum records to return (default 500)
    place_id : int, optional
        iNaturalist place ID to filter by
    quality_grade : str
        Quality filter: "research", "needs_id", or "casual"
    **kwargs
        Additional API parameters
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'coords': ndarray of (lon, lat) pairs
        - 'data': full response data
        - 'count': number of records
        
    Examples
    --------
    >>> result = inat("Pycnopodia helianthoides", limit=100)
    >>> print(f"Found {result['count']} records")
    """
    import urllib.request
    import urllib.parse
    import json
    
    base_url = "https://api.inaturalist.org/v1/observations"
    
    params = {
        'taxon_name': species,
        'per_page': min(limit, 200),  # API max is 200
        'quality_grade': quality_grade,
        'geo': 'true',
    }
    
    if place_id:
        params['place_id'] = place_id
    
    params.update(kwargs)
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        return {
            'coords': np.array([]).reshape(0, 2),
            'data': {'error': str(e)},
            'count': 0,
        }
    
    # Extract coordinates
    records = data.get('results', [])
    coords = []
    
    for rec in records:
        loc = rec.get('location')
        if loc:
            # iNat returns "lat,lon" string
            parts = loc.split(',')
            if len(parts) == 2:
                lat, lon = float(parts[0]), float(parts[1])
                coords.append([lon, lat])  # Return as lon, lat
    
    return {
        'coords': np.array(coords) if coords else np.array([]).reshape(0, 2),
        'data': data,
        'count': len(coords),
    }


def worldclim(
    var: str = "bio",
    res: float = 10,
    path: Optional[str] = None
) -> str:
    """
    Get WorldClim climate data.
    
    This function provides download URLs for WorldClim data.
    For actual raster handling, use rasterio or xarray.
    
    Parameters
    ----------
    var : str
        Variable: "bio" (bioclimatic), "tmin", "tmax", "prec", "tavg"
    res : float
        Resolution in arc-minutes: 0.5, 2.5, 5, or 10
    path : str, optional
        Directory to save files (if downloading)
        
    Returns
    -------
    str
        Download URL for the requested data
        
    Notes
    -----
    WorldClim 2.1 data: https://www.worldclim.org/data/worldclim21.html
    
    For programmatic download and loading:
    >>> import rasterio
    >>> url = worldclim("bio", res=10)
    >>> # Download and open with rasterio
    """
    res_map = {
        0.5: "30s",
        2.5: "2.5m", 
        5: "5m",
        10: "10m",
    }
    
    if res not in res_map:
        raise ValueError(f"Resolution must be one of {list(res_map.keys())}")
    
    res_str = res_map[res]
    base = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base"
    
    var_map = {
        "bio": "wc2.1",
        "tmin": "wc2.1",
        "tmax": "wc2.1",
        "prec": "wc2.1",
        "tavg": "wc2.1",
    }
    
    if var not in var_map:
        raise ValueError(f"Variable must be one of {list(var_map.keys())}")
    
    filename = f"wc2.1_{res_str}_{var}.zip"
    url = f"{base}/{filename}"
    
    return url
