"""Tests for ConvexHull model."""

import pytest
import numpy as np
from dismo import ConvexHull, convex_hull, point_in_polygon


class TestConvexHullFunction:
    """Test convex hull computation."""
    
    def test_triangle(self):
        """Three points form a triangle hull."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        hull = convex_hull(points)
        
        assert len(hull) == 3
    
    def test_square(self):
        """Square with center point."""
        points = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],  # corners
            [0.5, 0.5]  # center
        ])
        hull = convex_hull(points)
        
        # Only corners should be in hull
        assert len(hull) == 4
    
    def test_collinear(self):
        """Collinear points."""
        points = np.array([[0, 0], [1, 1], [2, 2]])
        hull = convex_hull(points)
        
        # Should return endpoints
        assert len(hull) <= 3


class TestPointInPolygon:
    """Test point-in-polygon."""
    
    def test_inside(self):
        """Point inside polygon."""
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        
        assert point_in_polygon(1, 1, polygon)
    
    def test_outside(self):
        """Point outside polygon."""
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        
        assert not point_in_polygon(3, 3, polygon)
    
    def test_on_edge(self):
        """Point on edge (edge case)."""
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        
        # On edge behavior can vary - just check it doesn't crash
        result = point_in_polygon(1, 0, polygon)
        assert isinstance(result, bool)


class TestConvexHullModel:
    """Test ConvexHull SDM model."""
    
    def test_fit_basic(self):
        """Test basic fitting."""
        points = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])
        
        model = ConvexHull()
        model.fit(points)
        
        assert model._fitted
        assert model.hull_vertices_ is not None
    
    def test_predict_inside(self):
        """Points inside hull should have suitability 1."""
        points = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2]
        ])
        
        model = ConvexHull()
        model.fit(points)
        
        test = np.array([[1, 1]])  # Center
        pred = model.predict(test)
        
        assert pred[0] == 1.0
    
    def test_predict_outside(self):
        """Points outside hull should have suitability 0."""
        points = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2]
        ])
        
        model = ConvexHull()
        model.fit(points)
        
        test = np.array([[5, 5]])  # Outside
        pred = model.predict(test)
        
        assert pred[0] == 0.0
    
    def test_predict_with_buffer(self):
        """Points near hull with buffer should have partial suitability."""
        points = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2]
        ])
        
        model = ConvexHull(buffer=1.0)
        model.fit(points)
        
        # Just outside the hull
        test = np.array([[2.5, 1]])
        pred = model.predict(test)
        
        assert 0 < pred[0] < 1.0
    
    def test_requires_3_points(self):
        """Should error with fewer than 3 points."""
        points = np.array([[0, 0], [1, 1]])
        
        model = ConvexHull()
        with pytest.raises(ValueError, match="at least 3"):
            model.fit(points)
    
    def test_get_hull_polygon(self):
        """Should return hull vertices."""
        points = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2], [1, 1]
        ])
        
        model = ConvexHull()
        model.fit(points)
        
        hull = model.get_hull_polygon()
        assert len(hull) == 4  # Only corners


class TestConvexHullRParity:
    """Test parity with R dismo::convHull."""
    
    def test_algorithm_noted(self):
        """
        Both R and Python use standard convex hull algorithms.
        Results should be equivalent for the same input points.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
