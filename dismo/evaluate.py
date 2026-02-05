"""
Model evaluation utilities.

Provides evaluate() function similar to R dismo::evaluate().
For more comprehensive evaluation, use enmeval-py.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Tuple, Any


def evaluate(
    presence: NDArray,
    absence: NDArray,
    model: Any = None,
    predictions_pres: NDArray = None,
    predictions_abs: NDArray = None
) -> Dict:
    """
    Evaluate SDM model performance.
    
    Similar to R's dismo::evaluate(). For comprehensive evaluation
    with cross-validation, use enmeval-py.
    
    Parameters
    ----------
    presence : ndarray
        Presence data (environmental values or coordinates)
    absence : ndarray
        Absence/background data
    model : object, optional
        Fitted model with predict() method
    predictions_pres : ndarray, optional
        Pre-computed predictions at presence locations
    predictions_abs : ndarray, optional
        Pre-computed predictions at absence locations
        
    Returns
    -------
    dict
        Evaluation metrics:
        - 'auc': Area Under ROC Curve
        - 'cor': Correlation between predictions
        - 'pcor': Point-biserial correlation
        - 'prevalence': Presence prevalence
        - 'threshold': Optimal threshold (max TSS)
        - 'sensitivity': True positive rate at threshold
        - 'specificity': True negative rate at threshold
        - 'tss': True Skill Statistic at threshold
        - 'kappa': Cohen's Kappa at threshold
        
    Examples
    --------
    >>> from dismo import Bioclim, evaluate
    >>> 
    >>> # Fit model
    >>> model = Bioclim().fit(presence_env)
    >>> 
    >>> # Evaluate
    >>> result = evaluate(presence_env, background_env, model=model)
    >>> print(f"AUC: {result['auc']:.3f}")
    """
    # Get predictions
    if predictions_pres is None:
        if model is None:
            raise ValueError("Either model or predictions must be provided")
        predictions_pres = model.predict(presence)
    
    if predictions_abs is None:
        if model is None:
            raise ValueError("Either model or predictions must be provided")
        predictions_abs = model.predict(absence)
    
    n_pres = len(predictions_pres)
    n_abs = len(predictions_abs)
    
    # AUC (Wilcoxon-Mann-Whitney)
    auc = _calc_auc(predictions_pres, predictions_abs)
    
    # Prevalence
    prevalence = n_pres / (n_pres + n_abs)
    
    # Find optimal threshold (maximizing TSS)
    all_preds = np.concatenate([predictions_pres, predictions_abs])
    thresholds = np.unique(np.percentile(all_preds, np.arange(0, 101, 5)))
    
    best_tss = -1
    best_threshold = 0.5
    best_sens = 0
    best_spec = 0
    
    for thresh in thresholds:
        sens = np.mean(predictions_pres >= thresh)
        spec = np.mean(predictions_abs < thresh)
        tss = sens + spec - 1
        
        if tss > best_tss:
            best_tss = tss
            best_threshold = thresh
            best_sens = sens
            best_spec = spec
    
    # Kappa at best threshold
    tp = np.sum(predictions_pres >= best_threshold)
    tn = np.sum(predictions_abs < best_threshold)
    fp = np.sum(predictions_abs >= best_threshold)
    fn = np.sum(predictions_pres < best_threshold)
    
    total = tp + tn + fp + fn
    po = (tp + tn) / total  # Observed agreement
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2)  # Expected
    
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    
    # Point-biserial correlation
    labels = np.concatenate([np.ones(n_pres), np.zeros(n_abs)])
    all_preds = np.concatenate([predictions_pres, predictions_abs])
    
    # Pearson correlation as proxy
    if np.std(all_preds) > 0:
        cor = np.corrcoef(labels, all_preds)[0, 1]
    else:
        cor = 0
    
    return {
        'auc': auc,
        'cor': cor,
        'prevalence': prevalence,
        'threshold': best_threshold,
        'sensitivity': best_sens,
        'specificity': best_spec,
        'tss': best_tss,
        'kappa': kappa,
        'n_presence': n_pres,
        'n_absence': n_abs,
    }


def _calc_auc(presence_pred: NDArray, background_pred: NDArray) -> float:
    """Calculate AUC using Wilcoxon-Mann-Whitney statistic."""
    n_pres = len(presence_pred)
    n_bg = len(background_pred)
    
    if n_pres == 0 or n_bg == 0:
        return np.nan
    
    # Count concordant pairs
    concordant = 0
    ties = 0
    
    for p in presence_pred:
        concordant += np.sum(background_pred < p)
        ties += np.sum(background_pred == p)
    
    auc = (concordant + 0.5 * ties) / (n_pres * n_bg)
    return auc


def threshold(
    presence_pred: NDArray,
    absence_pred: NDArray,
    method: str = "tss"
) -> float:
    """
    Find optimal prediction threshold.
    
    Parameters
    ----------
    presence_pred : ndarray
        Predictions at presence locations
    absence_pred : ndarray
        Predictions at absence locations
    method : str
        Method for threshold selection:
        - "tss": Maximize True Skill Statistic (default)
        - "sensitivity": First threshold with sensitivity >= 0.9
        - "specificity": First threshold with specificity >= 0.9
        - "equal_sens_spec": Where sensitivity ≈ specificity
        
    Returns
    -------
    float
        Optimal threshold value
    """
    all_preds = np.concatenate([presence_pred, absence_pred])
    thresholds = np.unique(np.percentile(all_preds, np.arange(0, 101, 2)))
    
    if method == "tss":
        best_score = -1
        best_thresh = 0.5
        
        for thresh in thresholds:
            sens = np.mean(presence_pred >= thresh)
            spec = np.mean(absence_pred < thresh)
            tss = sens + spec - 1
            
            if tss > best_score:
                best_score = tss
                best_thresh = thresh
        
        return best_thresh
    
    elif method == "sensitivity":
        for thresh in sorted(thresholds, reverse=True):
            if np.mean(presence_pred >= thresh) >= 0.9:
                return thresh
        return thresholds[0]
    
    elif method == "specificity":
        for thresh in sorted(thresholds):
            if np.mean(absence_pred < thresh) >= 0.9:
                return thresh
        return thresholds[-1]
    
    elif method == "equal_sens_spec":
        best_diff = float('inf')
        best_thresh = 0.5
        
        for thresh in thresholds:
            sens = np.mean(presence_pred >= thresh)
            spec = np.mean(absence_pred < thresh)
            diff = abs(sens - spec)
            
            if diff < best_diff:
                best_diff = diff
                best_thresh = thresh
        
        return best_thresh
    
    else:
        raise ValueError(f"Unknown method: {method}")
