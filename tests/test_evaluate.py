"""Tests for model evaluation."""

import pytest
import numpy as np
from dismo import evaluate, threshold, Bioclim


class TestEvaluate:
    """Test model evaluation."""
    
    def test_perfect_model(self):
        """Perfect separation should give AUC = 1."""
        pres_pred = np.array([0.9, 0.95, 0.85, 0.92])
        abs_pred = np.array([0.1, 0.05, 0.15, 0.08])
        
        result = evaluate(
            presence=None, absence=None,
            predictions_pres=pres_pred,
            predictions_abs=abs_pred
        )
        
        assert result['auc'] == 1.0
    
    def test_random_model(self):
        """Random predictions should give AUC ≈ 0.5."""
        np.random.seed(42)
        pres_pred = np.random.uniform(0, 1, 100)
        abs_pred = np.random.uniform(0, 1, 100)
        
        result = evaluate(
            presence=None, absence=None,
            predictions_pres=pres_pred,
            predictions_abs=abs_pred
        )
        
        # Should be close to 0.5 (within 0.1)
        assert 0.4 < result['auc'] < 0.6
    
    def test_returns_all_metrics(self):
        """Should return all expected metrics."""
        pres_pred = np.array([0.7, 0.8, 0.6])
        abs_pred = np.array([0.3, 0.4, 0.2])
        
        result = evaluate(
            presence=None, absence=None,
            predictions_pres=pres_pred,
            predictions_abs=abs_pred
        )
        
        expected_keys = ['auc', 'cor', 'prevalence', 'threshold',
                         'sensitivity', 'specificity', 'tss', 'kappa']
        
        for key in expected_keys:
            assert key in result
    
    def test_with_model(self):
        """Should work with a fitted model."""
        np.random.seed(42)
        
        # Train data
        train = np.random.normal(15, 2, (50, 2))
        train[:, 1] = train[:, 0] * 50 + np.random.normal(0, 20, 50)
        
        # Background data (different distribution)
        bg = np.random.uniform(5, 25, (100, 2))
        bg[:, 1] = np.random.uniform(200, 1000, 100)
        
        model = Bioclim()
        model.fit(train)
        
        result = evaluate(train, bg, model=model)
        
        assert 'auc' in result
        assert 0 <= result['auc'] <= 1


class TestThreshold:
    """Test threshold selection."""
    
    def test_tss_method(self):
        """TSS method should find good threshold."""
        pres_pred = np.array([0.7, 0.8, 0.9, 0.75, 0.85])
        abs_pred = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
        
        thresh = threshold(pres_pred, abs_pred, method="tss")
        
        # Should be between presence and absence predictions
        assert 0.3 < thresh < 0.7
    
    def test_sensitivity_method(self):
        """Sensitivity method should ensure high sensitivity."""
        pres_pred = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        abs_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.45])
        
        thresh = threshold(pres_pred, abs_pred, method="sensitivity")
        
        # Should give sensitivity >= 0.9
        sens = np.mean(pres_pred >= thresh)
        assert sens >= 0.9 or thresh == min(pres_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
