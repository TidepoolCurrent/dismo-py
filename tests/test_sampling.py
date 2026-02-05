"""Tests for background sampling utilities."""

import pytest
import numpy as np
from dismo import randomPoints, gridSample, targetGroupBackground, spatialThin


class TestRandomPoints:
    """Test random background point generation."""
    
    def test_basic_sampling(self):
        """Generate points within extent."""
        extent = (0, 10, 0, 10)
        points = randomPoints(100, extent)
        
        assert points.shape == (100, 2)
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 0] <= 10)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 1] <= 10)
    
    def test_with_buffer(self):
        """Points should avoid exclusion buffer."""
        extent = (0, 10, 0, 10)
        exclusion = np.array([[5, 5]])  # Exclude center
        
        points = randomPoints(100, extent, 
                             exclusion_buffer=exclusion, 
                             buffer_distance=2)
        
        # Check most points are outside buffer
        dists = np.sqrt((points[:, 0] - 5)**2 + (points[:, 1] - 5)**2)
        assert np.mean(dists >= 2) > 0.8  # Most points outside
    
    def test_geographic_extent(self):
        """Test with realistic geographic extent."""
        extent = (-125, -115, 32, 42)  # California
        points = randomPoints(1000, extent)
        
        assert points.shape == (1000, 2)
        assert np.all(points[:, 0] >= -125)
        assert np.all(points[:, 0] <= -115)


class TestGridSample:
    """Test regular grid sampling."""
    
    def test_basic_grid(self):
        """Generate a regular grid."""
        extent = (0, 10, 0, 10)
        points = gridSample(extent, resolution=1)
        
        # 10x10 grid
        assert len(points) == 100
        assert points.shape[1] == 2
    
    def test_grid_spacing(self):
        """Check grid spacing is correct."""
        extent = (0, 10, 0, 10)
        points = gridSample(extent, resolution=2)
        
        # 5x5 grid
        assert len(points) == 25
        
        # Check spacing
        x_unique = np.unique(points[:, 0])
        spacing = np.diff(x_unique)
        assert np.allclose(spacing, 2.0)


class TestTargetGroupBackground:
    """Test target-group background sampling."""
    
    def test_excludes_focal(self):
        """Should exclude focal species occurrences."""
        focal = np.array([[0, 0], [1, 1]])
        all_occ = np.array([
            [0, 0],  # Focal (exclude)
            [1, 1],  # Focal (exclude)
            [2, 2],
            [3, 3],
            [4, 4]
        ])
        
        bg = targetGroupBackground(focal, all_occ, n=3)
        
        assert len(bg) == 3
        # Check none are at focal locations
        for point in bg:
            assert not (np.allclose(point, [0, 0]) or np.allclose(point, [1, 1]))
    
    def test_sample_count(self):
        """Should return requested number of points."""
        focal = np.array([[0, 0]])
        all_occ = np.random.uniform(0, 10, (100, 2))
        
        bg = targetGroupBackground(focal, all_occ, n=50)
        assert len(bg) == 50


class TestSpatialThin:
    """Test spatial thinning."""
    
    def test_removes_close_points(self):
        """Should remove points that are too close."""
        # Cluster of points
        points = np.array([
            [0, 0],
            [0.1, 0.1],  # Too close to first
            [5, 5],
            [5.1, 5.1],  # Too close to third
            [10, 10]
        ])
        
        thinned = spatialThin(points, min_distance=1.0)
        
        # Should keep roughly 3 points (one from each cluster)
        assert len(thinned) <= 3
    
    def test_preserves_spacing(self):
        """Retained points should be well-spaced."""
        np.random.seed(42)
        points = np.random.uniform(0, 100, (1000, 2))
        
        thinned = spatialThin(points, min_distance=5.0)
        
        # Check all pairs are at least min_distance apart
        for i in range(len(thinned)):
            for j in range(i + 1, len(thinned)):
                dist = np.sqrt((thinned[i, 0] - thinned[j, 0])**2 + 
                              (thinned[i, 1] - thinned[j, 1])**2)
                assert dist >= 5.0, f"Points {i} and {j} too close: {dist}"
    
    def test_empty_input(self):
        """Handle empty input."""
        points = np.array([]).reshape(0, 2)
        thinned = spatialThin(points, min_distance=1.0)
        assert len(thinned) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
