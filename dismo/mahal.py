"""
Mahalanobis distance species distribution model.

Mahalanobis distance accounts for correlations between variables,
unlike Euclidean distance which treats variables as independent.

Reference:
    Farber, O. and R. Kadmon, 2003. Assessment of alternative approaches 
    for bioclimatic modeling with special emphasis on the Mahalanobis distance. 
    Ecological Modelling, 160:115-130.
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


class Mahalanobis:
    """
    Mahalanobis distance SDM model.
    
    Calculates habitat suitability based on Mahalanobis distance
    from the centroid of presence locations. Unlike Euclidean distance,
    Mahalanobis distance accounts for correlations between variables.
    
    Parameters
    ----------
    threshold : float, optional
        Chi-squared quantile threshold for suitability cutoff.
        Default 0.95 (95% of chi-squared distribution).
        
    Attributes
    ----------
    mean_ : ndarray
        Centroid of presence data
    cov_ : ndarray
        Covariance matrix of presence data
    cov_inv_ : ndarray
        Inverse of covariance matrix
    variables_ : list
        Names of environmental variables
        
    Examples
    --------
    >>> from dismo import Mahalanobis
    >>> import numpy as np
    >>> 
    >>> # Presence data with correlated variables
    >>> presence = np.array([
    ...     [15, 800],
    ...     [16, 850],
    ...     [14, 780],
    ...     [17, 900],
    ...     [15.5, 820]
    ... ])
    >>> 
    >>> model = Mahalanobis()
    >>> model.fit(presence)
    >>> 
    >>> # Predict at new locations
    >>> new_sites = np.array([[15.5, 825], [25, 500]])
    >>> suitability = model.predict(new_sites)
    
    Notes
    -----
    Mahalanobis distance is defined as:
        D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
    
    where μ is the centroid and Σ is the covariance matrix.
    
    Under multivariate normality, D² follows a chi-squared distribution
    with degrees of freedom equal to the number of variables.
    """
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.mean_ = None
        self.cov_ = None
        self.cov_inv_ = None
        self.chi2_cutoff_ = None
        self.variables_ = None
        self._fitted = False
    
    def fit(self, X: Union['pd.DataFrame', NDArray]) -> 'Mahalanobis':
        """
        Fit the Mahalanobis model to presence data.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at presence locations.
            Shape (n_samples, n_variables).
            Requires n_samples > n_variables for covariance estimation.
            
        Returns
        -------
        self : Mahalanobis
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
        
        n_samples, n_vars = X.shape
        
        if n_samples <= n_vars:
            raise ValueError(
                f"Need more samples ({n_samples}) than variables ({n_vars}) "
                "for covariance estimation"
            )
        
        # Calculate centroid and covariance
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        
        # Handle 1D case
        if self.cov_.ndim == 0:
            self.cov_ = np.array([[self.cov_]])
        
        # Use pseudoinverse for numerical stability
        # This handles near-singular covariance matrices better
        self.cov_inv_ = np.linalg.pinv(self.cov_)
        
        # Chi-squared cutoff for threshold
        from scipy.stats import chi2
        self.chi2_cutoff_ = chi2.ppf(self.threshold, df=n_vars)
        
        self._fitted = True
        return self
    
    def predict(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Predict habitat suitability for new locations.
        
        Suitability is based on Mahalanobis distance, converted to
        a 0-1 scale using the chi-squared distribution.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Environmental values at prediction locations.
            
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
        
        # Calculate Mahalanobis distance squared
        d2 = self.mahalanobis_distance_squared(X)
        
        # Convert to suitability using chi-squared CDF
        # P(D² < d²) gives probability of being this close or closer
        from scipy.stats import chi2
        n_vars = len(self.mean_)
        
        # Suitability = 1 - P(D² < d²)
        # High distance = low suitability
        suitability = 1 - chi2.cdf(d2, df=n_vars)
        
        return suitability
    
    def mahalanobis_distance_squared(self, X: NDArray) -> NDArray:
        """
        Calculate squared Mahalanobis distance.
        
        Parameters
        ----------
        X : ndarray
            Points to calculate distance for. Shape (n, p).
            
        Returns
        -------
        d2 : ndarray
            Squared Mahalanobis distances. Shape (n,).
        """
        diff = X - self.mean_
        
        # D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        # For each row: d2[i] = diff[i] @ cov_inv @ diff[i].T
        left = diff @ self.cov_inv_
        d2 = np.sum(left * diff, axis=1)
        
        return d2
    
    def mahalanobis_distance(self, X: Union['pd.DataFrame', NDArray]) -> NDArray:
        """
        Calculate Mahalanobis distance (not squared).
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Points to calculate distance for.
            
        Returns
        -------
        d : ndarray
            Mahalanobis distances.
        """
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X = X[self.variables_].values
        else:
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return np.sqrt(self.mahalanobis_distance_squared(X))
