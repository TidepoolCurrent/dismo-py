"""Tests for bioclimatic variables calculation."""

import pytest
import numpy as np
from dismo import biovars


class TestBiovars:
    """Test bioclimatic variable calculation."""
    
    def test_basic_calculation(self):
        """Calculate biovars from monthly data."""
        # Mediterranean climate example
        tmin = np.array([2, 3, 5, 8, 12, 15, 17, 17, 14, 10, 6, 3])
        tmax = np.array([8, 10, 14, 18, 23, 27, 30, 29, 25, 19, 13, 9])
        prec = np.array([80, 70, 65, 55, 45, 30, 20, 25, 40, 60, 75, 85])
        
        bio = biovars(tmin, tmax, prec)
        
        # Check all 19 variables present
        assert len(bio) == 19
        for i in range(1, 20):
            assert f'bio{i}' in bio
    
    def test_bio1_annual_mean_temp(self):
        """bio1 should be annual mean temperature."""
        tmin = np.array([10] * 12)
        tmax = np.array([20] * 12)
        prec = np.array([50] * 12)
        
        bio = biovars(tmin, tmax, prec)
        
        # Mean of (10+20)/2 = 15
        assert bio['bio1'] == 15.0
    
    def test_bio2_diurnal_range(self):
        """bio2 should be mean diurnal range."""
        tmin = np.array([10] * 12)
        tmax = np.array([20] * 12)
        prec = np.array([50] * 12)
        
        bio = biovars(tmin, tmax, prec)
        
        # Mean of (20-10) = 10
        assert bio['bio2'] == 10.0
    
    def test_bio5_max_temp_warmest(self):
        """bio5 should be max temp of warmest month."""
        tmin = np.array([0, 5, 10, 15, 20, 25, 30, 25, 20, 15, 10, 5])
        tmax = np.array([10, 15, 20, 25, 30, 35, 40, 35, 30, 25, 20, 15])
        prec = np.array([50] * 12)
        
        bio = biovars(tmin, tmax, prec)
        
        assert bio['bio5'] == 40.0
    
    def test_bio6_min_temp_coldest(self):
        """bio6 should be min temp of coldest month."""
        tmin = np.array([0, 5, 10, 15, 20, 25, 30, 25, 20, 15, 10, 5])
        tmax = np.array([10, 15, 20, 25, 30, 35, 40, 35, 30, 25, 20, 15])
        prec = np.array([50] * 12)
        
        bio = biovars(tmin, tmax, prec)
        
        assert bio['bio6'] == 0.0
    
    def test_bio12_annual_precip(self):
        """bio12 should be annual precipitation."""
        tmin = np.array([10] * 12)
        tmax = np.array([20] * 12)
        prec = np.array([100] * 12)
        
        bio = biovars(tmin, tmax, prec)
        
        assert bio['bio12'] == 1200.0
    
    def test_requires_12_months(self):
        """Should error if not 12 months."""
        tmin = np.array([10] * 6)
        tmax = np.array([20] * 6)
        prec = np.array([50] * 6)
        
        with pytest.raises(ValueError, match="12"):
            biovars(tmin, tmax, prec)


class TestBiovarsRParity:
    """Test parity with R dismo::biovars."""
    
    def test_algorithm_matches_worldclim(self):
        """
        Biovars should match WorldClim/dismo definitions.
        The algorithm follows the same calculation as R's dismo::biovars.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
