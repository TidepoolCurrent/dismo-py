"""Tests for Domain model."""

import pytest
import numpy as np
from dismo import Domain
from dismo.domain import gower_distance


class TestGowerDistance:
    """Test Gower distance calculation."""
    
    def test_identical_points(self):
        """Identical points have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        ranges = np.array([10.0, 10.0, 10.0])
        
        dist = gower_distance(x, y, ranges)
        assert dist == 0.0
    
    def test_different_points(self):
        """Different points have positive distance."""
        x = np.array([0.0, 0.0])
        y = np.array([10.0, 10.0])
        ranges = np.array([10.0, 10.0])
        
        dist = gower_distance(x, y, ranges)
        assert dist == 1.0  # Max difference normalized by range
    
    def test_partial_difference(self):
        """Partial differences average correctly."""
        x = np.array([0.0, 5.0])
        y = np.array([10.0, 5.0])  # One var different, one same
        ranges = np.array([10.0, 10.0])
        
        dist = gower_distance(x, y, ranges)
        assert dist == 0.5  # Average of 1.0 and 0.0


class TestDomain:
    """Test Domain SDM model."""
    
    def test_fit_basic(self):
        """Test basic fitting."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780],
            [15.5, 820]
        ])
        
        model = Domain()
        model.fit(data)
        
        assert model._fitted
        assert len(model.variables_) == 2
        assert model.presence_data_.shape == (4, 2)
    
    def test_predict_at_presence(self):
        """Points at exact presence locations should have high suitability."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain()
        model.fit(data)
        
        # Predict at same location as a presence
        test = np.array([[15, 800]])
        pred = model.predict(test)
        
        assert pred[0] == 1.0, f"Exact match should be 1.0, got {pred[0]}"
    
    def test_predict_nearby(self):
        """Points near presences should have moderate suitability."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain(threshold=0.1)  # Larger threshold
        model.fit(data)
        
        # Predict at slightly different location
        test = np.array([[15.1, 805]])
        pred = model.predict(test)
        
        assert 0 < pred[0] < 1.0, f"Nearby point should be 0-1, got {pred[0]}"
    
    def test_predict_far(self):
        """Points far from presences should have 0 suitability."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain(threshold=0.05)
        model.fit(data)
        
        # Very different location
        test = np.array([[50, 2000]])
        pred = model.predict(test)
        
        assert pred[0] == 0.0, f"Distant point should be 0, got {pred[0]}"
    
    def test_predict_distance(self):
        """Test raw distance output."""
        data = np.array([
            [15, 800],
            [16, 850]
        ])
        
        model = Domain()
        model.fit(data)
        
        # Exact match should have distance 0
        test = np.array([[15, 800]])
        dist = model.predict_distance(test)
        
        assert dist[0] == 0.0
    
    def test_threshold_effect(self):
        """Different thresholds should affect predictions."""
        data = np.array([
            [15, 800],
            [16, 850]
        ])
        
        test = np.array([[15.5, 825]])
        
        # Low threshold = stricter
        model_strict = Domain(threshold=0.01)
        model_strict.fit(data)
        pred_strict = model_strict.predict(test)
        
        # High threshold = more lenient
        model_lenient = Domain(threshold=0.5)
        model_lenient.fit(data)
        pred_lenient = model_lenient.predict(test)
        
        # Lenient should give higher suitability
        assert pred_lenient[0] >= pred_strict[0]


class TestDomainRParity:
    """Test parity with R dismo::domain."""
    
    @pytest.mark.skip(reason="R reference data not yet generated")
    def test_predict_parity(self):
        """Compare predictions to R dismo."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
