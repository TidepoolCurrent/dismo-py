"""
Bioclimatic Variables (biovars)

Calculate the 19 standard bioclimatic variables from monthly 
temperature and precipitation data.

These are the same variables used in WorldClim and commonly
used for species distribution modeling.

Reference:
    https://www.worldclim.org/data/bioclim.html
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple


def biovars(
    tmin: NDArray,
    tmax: NDArray,
    prec: NDArray
) -> Dict[str, float]:
    """
    Calculate 19 bioclimatic variables from monthly climate data.
    
    Parameters
    ----------
    tmin : ndarray
        Monthly minimum temperatures (12 values, °C)
    tmax : ndarray
        Monthly maximum temperatures (12 values, °C)
    prec : ndarray
        Monthly precipitation (12 values, mm)
        
    Returns
    -------
    dict
        Dictionary with keys bio1-bio19 and their values
        
    Examples
    --------
    >>> # Monthly data for a location
    >>> tmin = [2, 3, 5, 8, 12, 15, 17, 17, 14, 10, 6, 3]
    >>> tmax = [8, 10, 14, 18, 23, 27, 30, 29, 25, 19, 13, 9]
    >>> prec = [80, 70, 65, 55, 45, 30, 20, 25, 40, 60, 75, 85]
    >>> bio = biovars(tmin, tmax, prec)
    >>> print(f"Mean Annual Temp: {bio['bio1']:.1f}°C")
    
    Notes
    -----
    The 19 bioclimatic variables are:
    
    Temperature:
    - bio1: Annual Mean Temperature
    - bio2: Mean Diurnal Range (mean of monthly max-min)
    - bio3: Isothermality (bio2/bio7 × 100)
    - bio4: Temperature Seasonality (std dev × 100)
    - bio5: Max Temperature of Warmest Month
    - bio6: Min Temperature of Coldest Month
    - bio7: Temperature Annual Range (bio5-bio6)
    - bio8: Mean Temperature of Wettest Quarter
    - bio9: Mean Temperature of Driest Quarter
    - bio10: Mean Temperature of Warmest Quarter
    - bio11: Mean Temperature of Coldest Quarter
    
    Precipitation:
    - bio12: Annual Precipitation
    - bio13: Precipitation of Wettest Month
    - bio14: Precipitation of Driest Month
    - bio15: Precipitation Seasonality (CV)
    - bio16: Precipitation of Wettest Quarter
    - bio17: Precipitation of Driest Quarter
    - bio18: Precipitation of Warmest Quarter
    - bio19: Precipitation of Coldest Quarter
    """
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    prec = np.asarray(prec, dtype=float)
    
    if len(tmin) != 12 or len(tmax) != 12 or len(prec) != 12:
        raise ValueError("All inputs must have 12 monthly values")
    
    # Mean temperature per month
    tavg = (tmin + tmax) / 2
    
    # --- Temperature variables ---
    
    # bio1: Annual Mean Temperature
    bio1 = np.mean(tavg)
    
    # bio2: Mean Diurnal Range
    bio2 = np.mean(tmax - tmin)
    
    # bio5: Max Temperature of Warmest Month
    bio5 = np.max(tmax)
    
    # bio6: Min Temperature of Coldest Month
    bio6 = np.min(tmin)
    
    # bio7: Temperature Annual Range
    bio7 = bio5 - bio6
    
    # bio3: Isothermality
    bio3 = (bio2 / bio7) * 100 if bio7 != 0 else 0
    
    # bio4: Temperature Seasonality (standard deviation × 100)
    bio4 = np.std(tavg) * 100
    
    # --- Precipitation variables ---
    
    # bio12: Annual Precipitation
    bio12 = np.sum(prec)
    
    # bio13: Precipitation of Wettest Month
    bio13 = np.max(prec)
    
    # bio14: Precipitation of Driest Month
    bio14 = np.min(prec)
    
    # bio15: Precipitation Seasonality (coefficient of variation)
    bio15 = (np.std(prec) / np.mean(prec)) * 100 if np.mean(prec) > 0 else 0
    
    # --- Quarter-based variables ---
    # A quarter is 3 consecutive months (wrapping around)
    
    def quarter_sum(data, start):
        """Sum of 3 consecutive months starting at index."""
        return data[start] + data[(start+1) % 12] + data[(start+2) % 12]
    
    def quarter_mean(data, start):
        """Mean of 3 consecutive months starting at index."""
        return quarter_sum(data, start) / 3
    
    # Find quarters
    quarter_prec = [quarter_sum(prec, i) for i in range(12)]
    quarter_tavg = [quarter_mean(tavg, i) for i in range(12)]
    
    wettest_q = np.argmax(quarter_prec)
    driest_q = np.argmin(quarter_prec)
    warmest_q = np.argmax(quarter_tavg)
    coldest_q = np.argmin(quarter_tavg)
    
    # bio8: Mean Temperature of Wettest Quarter
    bio8 = quarter_mean(tavg, wettest_q)
    
    # bio9: Mean Temperature of Driest Quarter
    bio9 = quarter_mean(tavg, driest_q)
    
    # bio10: Mean Temperature of Warmest Quarter
    bio10 = quarter_mean(tavg, warmest_q)
    
    # bio11: Mean Temperature of Coldest Quarter
    bio11 = quarter_mean(tavg, coldest_q)
    
    # bio16: Precipitation of Wettest Quarter
    bio16 = quarter_sum(prec, wettest_q)
    
    # bio17: Precipitation of Driest Quarter
    bio17 = quarter_sum(prec, driest_q)
    
    # bio18: Precipitation of Warmest Quarter
    bio18 = quarter_sum(prec, warmest_q)
    
    # bio19: Precipitation of Coldest Quarter
    bio19 = quarter_sum(prec, coldest_q)
    
    return {
        'bio1': bio1,
        'bio2': bio2,
        'bio3': bio3,
        'bio4': bio4,
        'bio5': bio5,
        'bio6': bio6,
        'bio7': bio7,
        'bio8': bio8,
        'bio9': bio9,
        'bio10': bio10,
        'bio11': bio11,
        'bio12': bio12,
        'bio13': bio13,
        'bio14': bio14,
        'bio15': bio15,
        'bio16': bio16,
        'bio17': bio17,
        'bio18': bio18,
        'bio19': bio19,
    }


def biovars_from_grids(
    tmin_grids: NDArray,
    tmax_grids: NDArray,
    prec_grids: NDArray
) -> Dict[str, NDArray]:
    """
    Calculate bioclimatic variables for gridded data.
    
    Parameters
    ----------
    tmin_grids : ndarray
        Monthly minimum temperature grids, shape (12, rows, cols)
    tmax_grids : ndarray
        Monthly maximum temperature grids, shape (12, rows, cols)
    prec_grids : ndarray
        Monthly precipitation grids, shape (12, rows, cols)
        
    Returns
    -------
    dict
        Dictionary with keys bio1-bio19, each containing a 2D grid
    """
    if tmin_grids.shape[0] != 12:
        raise ValueError("First dimension must be 12 (months)")
    
    rows, cols = tmin_grids.shape[1], tmin_grids.shape[2]
    
    result = {f'bio{i}': np.zeros((rows, cols)) for i in range(1, 20)}
    
    for r in range(rows):
        for c in range(cols):
            tmin = tmin_grids[:, r, c]
            tmax = tmax_grids[:, r, c]
            prec = prec_grids[:, r, c]
            
            bio = biovars(tmin, tmax, prec)
            for key, val in bio.items():
                result[key][r, c] = val
    
    return result
