"""
Bioclim - Climate envelope species distribution model.

Bioclim is one of the earliest SDM methods (Busby 1991). It defines
suitable habitat as the environmental envelope where the species
has been observed.

For each variable, it calculates the percentile of the prediction
location relative to the training presence distribution.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, TYPE_CHECKING

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


class Bioclim:
    """
    Bioclim climate envelope model.
    
    Bioclim predicts habitat suitability based on how well new locations
    fall within the environmental envelope defined by presence locations.
    
    For each environmental variable, the score is based on how far the
    value is from the median of the presence distribution, measured in
    percentiles. The overall score is the minimum across all variables.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    mins_ : ndarray
        Minimum value for each variable in training data
    maxs_ : ndarray
        Maximum value for each variable in training data
    medians_ : ndarray
        Median value for each variable
    percentiles_ : dict
        Percentile values for each variable (2.5, 5, 10, 25, 50, 75, 90, 95, 97.5)
    variables_ : list
        Names of environmental variables
        
    Examples
    --------
    >>> import pandas as pd
    >>> from dismo import Bioclim
    >>> 
    >>> # Training data (presence locations with environmental values)
    >>> presence = pd.DataFrame({
    ...     'bio1': [15, 16, 14, 17, 15],  # Mean temperature
    ...     'bio12': [800, 900, 750, 850, 820]  # Annual precipitation
    ... })
    >>> 
    >>> model = Bioclim()
    >>> model.fit(presence)
    >>> 
    >>> # Predict at new locations
    >>> new_sites = pd.DataFrame({
    ...     'bio1': [15, 25, 10],
    ...     'bio12': [800, 500, 900]
    ... })
    >>> predictions = model.predict(new_sites)
    
    References
    ----------
    Busby, J.R. (1991). BIOCLIM - a bioclimate analysis and prediction system.
    In: Margules, C.R. & Austin, M.P. (eds) Nature conservation: cost effective
    biological surveys and data analysis, pp. 64-68. CSIRO, Melbourne.
    """
    
    def __init__(self):
        self.mins_ = None
        self.maxs_ = None
        self.medians_ = None
        self.percentiles_ = None
        self.variables_ = None
        self._fitted = False
    
    def fit(self, X: Union['pd.DataFrame', NDArray]) -> 'Bioclim':
        """
        Fit the Bioclim model to presence data.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at presence locations.
            If DataFrame, column names are used as variable names.
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
        
        # Calculate envelope statistics
        self.mins_ = np.nanmin(X, axis=0)
        self.maxs_ = np.nanmax(X, axis=0)
        self.medians_ = np.nanmedian(X, axis=0)
        
        # Calculate percentiles for each variable
        pcts = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
        self.percentiles_ = {}
        for i, var in enumerate(self.variables_):
            self.percentiles_[var] = {
                p: np.nanpercentile(X[:, i], p) for p in pcts
            }
        
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
            1 = optimal (at median), 0 = outside envelope.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            # Ensure same variables in same order
            X = X[self.variables_].values
        else:
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_vars = X.shape[1]
        
        # Calculate score for each variable
        scores = np.zeros((n_samples, n_vars))
        
        for i in range(n_vars):
            scores[:, i] = self._score_variable(X[:, i], i)
        
        # Bioclim score is minimum across variables (limiting factor)
        predictions = np.min(scores, axis=1)
        
        return predictions
    
    def _score_variable(self, values: NDArray, var_idx: int) -> NDArray:
        """
        Calculate Bioclim score for one variable.
        
        Score is based on percentile position:
        - At median (50th percentile): score = 1
        - At 2.5th or 97.5th percentile: score = 0.05
        - Outside envelope: score = 0
        
        The score represents how "central" the value is to the
        training distribution.
        """
        var = self.variables_[var_idx]
        pcts = self.percentiles_[var]
        median = self.medians_[var_idx]
        
        scores = np.zeros(len(values))
        
        for j, val in enumerate(values):
            if np.isnan(val):
                scores[j] = np.nan
                continue
            
            # Outside envelope
            if val < self.mins_[var_idx] or val > self.maxs_[var_idx]:
                scores[j] = 0
                continue
            
            # Calculate percentile score
            if val <= median:
                # Below median: score based on lower tail
                # Interpolate between percentiles
                if val <= pcts[2.5]:
                    scores[j] = 0.05 * (val - self.mins_[var_idx]) / (pcts[2.5] - self.mins_[var_idx] + 1e-10)
                elif val <= pcts[5]:
                    scores[j] = 0.05 + 0.05 * (val - pcts[2.5]) / (pcts[5] - pcts[2.5] + 1e-10)
                elif val <= pcts[10]:
                    scores[j] = 0.10 + 0.15 * (val - pcts[5]) / (pcts[10] - pcts[5] + 1e-10)
                elif val <= pcts[25]:
                    scores[j] = 0.25 + 0.25 * (val - pcts[10]) / (pcts[25] - pcts[10] + 1e-10)
                else:  # <= median
                    scores[j] = 0.50 + 0.50 * (val - pcts[25]) / (pcts[50] - pcts[25] + 1e-10)
            else:
                # Above median: score based on upper tail
                if val >= pcts[97.5]:
                    scores[j] = 0.05 * (self.maxs_[var_idx] - val) / (self.maxs_[var_idx] - pcts[97.5] + 1e-10)
                elif val >= pcts[95]:
                    scores[j] = 0.05 + 0.05 * (pcts[97.5] - val) / (pcts[97.5] - pcts[95] + 1e-10)
                elif val >= pcts[90]:
                    scores[j] = 0.10 + 0.15 * (pcts[95] - val) / (pcts[95] - pcts[90] + 1e-10)
                elif val >= pcts[75]:
                    scores[j] = 0.25 + 0.25 * (pcts[90] - val) / (pcts[90] - pcts[75] + 1e-10)
                else:  # > median, < 75th
                    scores[j] = 0.50 + 0.50 * (pcts[75] - val) / (pcts[75] - pcts[50] + 1e-10)
        
        return np.clip(scores, 0, 1)
