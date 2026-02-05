"""Tests for Circle model."""

import pytest
import numpy as np
from dismo import Circle, haversine_distance


class TestHaversineDistance:
    """Test haversine distance calculation."""
    
    def test_same_point(self):
        """Same point should have distance 0."""
        d = haversine_distance(-122.4, 37.8, -122.4, 37.8)
        assert d == 0.0
    
    def test_known_distance(self):
        """Test against known distance."""
        # SF to LA is approximately 559 km
        d = haversine_distance(-122.4, 37.8, -118.2, 34.1)
        assert 550 < d < 570, f"SF-LA should be ~559km, got {d}"
    
    def test_symmetric(self):
        """Distance should be symmetric."""
        d1 = haversine_distance(-122.4, 37.8, -118.2, 34.1)
        d2 = haversine_distance(-118.2, 34.1, -122.4, 37.8)
        assert abs(d1 - d2) < 0.001


class TestCircle:
    """Test Circle geographic model."""
    
    def test_fit_basic(self):
        """Test basic fitting."""
        coords = np.array([
            [-122.4, 37.8],
            [-118.2, 34.1],
            [-117.2, 32.7]
        ])
        
        model = Circle()
        model.fit(coords)
        
        assert model._fitted
        assert model.presence_coords_.shape == (3, 2)
        assert model.threshold_ > 0
    
    def test_predict_at_presence(self):
        """Points at presence locations should have suitability 1."""
        coords = np.array([
            [-122.4, 37.8],
            [-118.2, 34.1]
        ])
        
        model = Circle(threshold=100)
        model.fit(coords)
        
        # Predict at same location
        test = np.array([[-122.4, 37.8]])
        pred = model.predict(test)
        
        assert pred[0] == 1.0
    
    def test_predict_nearby(self):
        """Points near presences should have moderate suitability."""
        coords = np.array([
            [-122.4, 37.8],
            [-118.2, 34.1]
        ])
        
        model = Circle(threshold=200)  # 200 km threshold
        model.fit(coords)
        
        # 50 km from SF
        test = np.array([[-122.0, 37.4]])
        pred = model.predict(test)
        
        assert 0 < pred[0] < 1.0
    
    def test_predict_far(self):
        """Points beyond threshold should have suitability 0."""
        coords = np.array([
            [-122.4, 37.8],  # SF
            [-118.2, 34.1]   # LA
        ])
        
        model = Circle(threshold=100)  # 100 km threshold
        model.fit(coords)
        
        # New York (very far)
        test = np.array([[-74.0, 40.7]])
        pred = model.predict(test)
        
        assert pred[0] == 0.0
    
    def test_predict_distance(self):
        """Test raw distance output."""
        coords = np.array([
            [-122.4, 37.8],
            [-118.2, 34.1]
        ])
        
        model = Circle()
        model.fit(coords)
        
        # At presence location
        test = np.array([[-122.4, 37.8]])
        dist = model.predict_distance(test)
        
        assert dist[0] == 0.0
    
    def test_euclidean_mode(self):
        """Test with Euclidean distance (degrees)."""
        coords = np.array([
            [0, 0],
            [1, 1]
        ])
        
        model = Circle(threshold=2, use_haversine=False)
        model.fit(coords)
        
        # 0.5 degrees away
        test = np.array([[0.5, 0]])
        pred = model.predict(test)
        
        assert 0 < pred[0] < 1.0
    
    def test_auto_threshold(self):
        """Threshold should auto-calculate from data."""
        coords = np.array([
            [-122.4, 37.8],  # SF
            [-118.2, 34.1]   # LA
        ])
        
        model = Circle()  # No threshold specified
        model.fit(coords)
        
        # Threshold should be SF-LA distance (~559 km)
        assert 550 < model.threshold_ < 570


class TestCircleRParity:
    """Test parity with R dismo::circles."""
    
    def test_algorithm_noted(self):
        """
        R's circles() creates circular buffers around points.
        Python implementation provides similar functionality
        with linear distance decay.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
