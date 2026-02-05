"""Tests for Mahalanobis distance model."""

import pytest
import numpy as np
from dismo import Mahalanobis


class TestMahalanobis:
    """Test Mahalanobis distance SDM model."""
    
    def test_fit_basic(self):
        """Test basic fitting."""
        np.random.seed(42)
        # Generate correlated data
        mean = [15, 800]
        cov = [[4, 50], [50, 2500]]
        data = np.random.multivariate_normal(mean, cov, 50)
        
        model = Mahalanobis()
        model.fit(data)
        
        assert model._fitted
        assert model.mean_.shape == (2,)
        assert model.cov_.shape == (2, 2)
        assert model.cov_inv_.shape == (2, 2)
    
    def test_predict_at_centroid(self):
        """Points at centroid should have highest suitability."""
        np.random.seed(42)
        data = np.random.multivariate_normal([15, 800], [[4, 0], [0, 2500]], 50)
        
        model = Mahalanobis()
        model.fit(data)
        
        # Centroid should have high suitability
        centroid = model.mean_.reshape(1, -1)
        pred = model.predict(centroid)
        
        assert pred[0] > 0.9, f"Centroid should have high suitability, got {pred[0]}"
    
    def test_predict_far_point(self):
        """Points far from centroid should have low suitability."""
        np.random.seed(42)
        data = np.random.multivariate_normal([15, 800], [[4, 0], [0, 2500]], 50)
        
        model = Mahalanobis()
        model.fit(data)
        
        # Very far point
        far = np.array([[100, 5000]])
        pred = model.predict(far)
        
        assert pred[0] < 0.1, f"Far point should have low suitability, got {pred[0]}"
    
    def test_accounts_for_correlation(self):
        """Mahalanobis should account for variable correlations."""
        np.random.seed(42)
        # Strongly correlated: high temp = high precip
        mean = [15, 800]
        cov = [[4, 100], [100, 2500]]  # Positive correlation
        data = np.random.multivariate_normal(mean, cov, 100)
        
        model = Mahalanobis()
        model.fit(data)
        
        # Point that matches correlation direction (slightly high temp + precip)
        correlated = np.array([[16, 850]])  # Follows correlation
        # Point that violates correlation (high temp, low precip)
        uncorrelated = np.array([[16, 750]])  # Against correlation
        
        pred_corr = model.predict(correlated)
        pred_uncorr = model.predict(uncorrelated)
        
        # Correlated point should be more suitable
        assert pred_corr[0] > pred_uncorr[0], \
            f"Correlated {pred_corr[0]:.3f} should > uncorrelated {pred_uncorr[0]:.3f}"
    
    def test_mahalanobis_distance(self):
        """Test raw distance output."""
        np.random.seed(42)
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
        
        model = Mahalanobis()
        model.fit(data)
        
        # Origin should have distance ~0
        origin = np.array([[0, 0]])
        dist = model.mahalanobis_distance(origin)
        
        assert dist[0] < 1.0, f"Origin should have small distance, got {dist[0]}"
    
    def test_requires_enough_samples(self):
        """Should error if n_samples <= n_variables."""
        # 2 samples, 3 variables - can't estimate covariance
        data = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        model = Mahalanobis()
        with pytest.raises(ValueError, match="Need more samples"):
            model.fit(data)
    
    def test_singular_covariance(self):
        """Should handle singular covariance matrices."""
        # Perfectly correlated (singular cov)
        data = np.array([
            [1, 2],
            [2, 4],
            [3, 6],
            [4, 8],
            [5, 10]
        ])
        
        model = Mahalanobis()
        model.fit(data)  # Should add regularization
        
        assert model._fitted
        # Should still be able to predict
        pred = model.predict(np.array([[3, 6]]))
        assert not np.isnan(pred[0])


class TestMahalanobisRParity:
    """Test parity with R dismo::mahal."""
    
    @pytest.mark.skip(reason="R reference data not yet generated")
    def test_predict_parity(self):
        """Compare predictions to R dismo."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
