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
        
        # Should be high (not exactly 1.0 due to mean distance algorithm)
        assert pred[0] > 0.5, f"Presence location should score high, got {pred[0]}"
    
    def test_predict_nearby(self):
        """Points near presences should have moderate suitability."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain()
        model.fit(data)
        
        # Predict at slightly different location
        test = np.array([[15.1, 805]])
        pred = model.predict(test)
        
        assert 0 < pred[0] <= 1.0, f"Nearby point should be 0-1, got {pred[0]}"
    
    def test_predict_far(self):
        """Points very far from presences should have low/zero suitability."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain()
        model.fit(data)
        
        # Very different location (far outside range)
        test = np.array([[100, 5000]])
        pred = model.predict(test)
        
        assert pred[0] == 0.0, f"Distant point should be 0, got {pred[0]}"
    
    def test_predict_distance(self):
        """Test raw distance output."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain()
        model.fit(data)
        
        # At a presence location, mean distance should be based on 
        # distance to OTHER presences
        test = np.array([[15, 800]])
        dist = model.predict_distance(test)
        
        # Distance should be small but not necessarily 0
        assert dist[0] < 0.5, f"Distance at presence should be small, got {dist[0]}"
    
    def test_suitability_decreases_with_distance(self):
        """Suitability should decrease as we move away from presences."""
        data = np.array([
            [15, 800],
            [16, 850],
            [14, 780]
        ])
        
        model = Domain()
        model.fit(data)
        
        # Test at increasing distances
        test_near = np.array([[15.5, 825]])
        test_mid = np.array([[20, 1000]])
        test_far = np.array([[30, 1500]])
        
        pred_near = model.predict(test_near)[0]
        pred_mid = model.predict(test_mid)[0]
        pred_far = model.predict(test_far)[0]
        
        assert pred_near >= pred_mid >= pred_far, \
            f"Suitability should decrease: {pred_near} >= {pred_mid} >= {pred_far}"


class TestDomainRParity:
    """Test parity with R dismo::domain."""
    
    def test_predict_parity(self):
        """Compare predictions to R dismo::domain."""
        train = np.array([
            [15, 800],
            [16, 900],
            [14, 750],
            [17, 850],
            [15, 820]
        ])
        
        test = np.array([
            [15.5, 825],
            [25, 500],
            [14.5, 780]
        ])
        
        # R reference values (from dismo::domain)
        r_predictions = np.array([0.7, 0, 0.6266667])
        
        model = Domain()
        model.fit(train)
        py_predictions = model.predict(test)
        
        assert np.allclose(py_predictions, r_predictions, atol=0.001), \
            f"Python {py_predictions} != R {r_predictions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
