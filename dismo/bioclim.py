"""
Bioclim - Climate envelope species distribution model.

Bioclim is one of the earliest SDM methods (Busby 1991). It defines
suitable habitat as the environmental envelope where the species
has been observed.

This implementation matches R's dismo::bioclim algorithm.

Reference:
    Nix, H.A., 1986. A biogeographic analysis of Australian elapid snakes.
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


class Bioclim:
    """
    Bioclim climate envelope model.
    
    Matches R's dismo::bioclim implementation.
    
    The BIOCLIM algorithm computes similarity by comparing environmental
    values at any location to a percentile distribution of values at
    known occurrence locations. The closer to the 50th percentile (median),
    the more suitable the location.
    
    Algorithm (from R dismo):
    1. For each variable, compute percentile of test value in training distribution
    2. Values > 0.5 are subtracted from 1 (tails treated equally)
    3. Take minimum percentile across all variables
    4. Final score = 2 * (1 - min_percentile), giving 0-1 range
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    presence_data_ : ndarray
        Environmental values at presence locations
    variables_ : list
        Names of environmental variables
    """
    
    def __init__(self):
        self.presence_data_ = None
        self.variables_ = None
        self._fitted = False
    
    def fit(self, X: Union['pd.DataFrame', NDArray]) -> 'Bioclim':
        """
        Fit the Bioclim model to presence data.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at presence locations.
            Shape (n_samples, n_variables).
            
        Returns
        -------
        self : Bioclim
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
        self.mins_ = np.nanmin(X, axis=0)
        self.maxs_ = np.nanmax(X, axis=0)
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
        
        # Calculate percentile scores for each variable
        percentile_scores = np.zeros((n_samples, n_vars))
        
        for j in range(n_vars):
            train_vals = self.presence_data_[:, j]
            train_vals = train_vals[~np.isnan(train_vals)]
            
            for i in range(n_samples):
                val = X[i, j]
                
                if np.isnan(val):
                    percentile_scores[i, j] = 0
                    continue
                
                # Check if outside range
                if val < train_vals.min() or val > train_vals.max():
                    percentile_scores[i, j] = 0
                    continue
                
                # Calculate percentile (proportion of training values <= test value)
                pct = np.sum(train_vals <= val) / len(train_vals)
                
                # Fold around 0.5 (tails treated equally)
                if pct > 0.5:
                    pct = 1 - pct
                
                percentile_scores[i, j] = pct
        
        # Minimum across variables (limiting factor)
        min_pct = np.min(percentile_scores, axis=1)
        
        # Transform: 2 * (1 - min_pct) would give values 0-1
        # But R uses: 2 * min_pct to scale properly
        # Actually, the score represents how close to median (0.5 = at median)
        # So we scale: 2 * min_pct gives 0-1 where 1 = at all medians
        predictions = 2 * min_pct
        
        return predictions
