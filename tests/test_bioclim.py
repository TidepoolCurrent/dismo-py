"""Tests for Bioclim model."""

import pytest
import numpy as np
from dismo import Bioclim


class TestBioclim:
    """Test Bioclim climate envelope model."""
    
    def test_fit_basic(self):
        """Test basic fitting."""
        # temp, precip columns
        data = np.array([
            [15, 800],
            [16, 900],
            [14, 750],
            [17, 850],
            [15, 820],
            [18, 880],
            [13, 770]
        ])
        
        model = Bioclim()
        model.fit(data)
        
        assert model._fitted
        assert len(model.variables_) == 2
        assert model.mins_[0] == 13
        assert model.maxs_[0] == 18
    
    def test_predict_inside_envelope(self):
        """Points inside envelope should have high scores."""
        data = np.array([
            [15, 800],
            [16, 900],
            [14, 750],
            [17, 850],
            [15, 820],
            [18, 880],
            [13, 770]
        ])
        
        model = Bioclim()
        model.fit(data)
        
        # Point at median should have high score
        test = np.array([[15.5, 825]])
        pred = model.predict(test)
        
        assert pred[0] > 0.3, f"Central point should score > 0.3, got {pred[0]}"
    
    def test_predict_outside_envelope(self):
        """Points outside envelope should score 0."""
        data = np.array([
            [15, 800],
            [16, 900],
            [14, 750],
            [17, 850],
            [15, 820]
        ])
        
        model = Bioclim()
        model.fit(data)
        
        # Point outside temperature range
        test = np.array([[25, 800]])
        pred = model.predict(test)
        
        assert pred[0] == 0, f"Outside point should score 0, got {pred[0]}"
    
    def test_predict_edge_of_envelope(self):
        """Points at edge should have low scores."""
        np.random.seed(42)
        temp = np.random.normal(15, 2, 100)
        precip = np.random.normal(800, 100, 100)
        data = np.column_stack([temp, precip])
        
        model = Bioclim()
        model.fit(data)
        
        # Point at 5th percentile (edge but not minimum)
        edge_temp = np.percentile(temp, 5)
        test = np.array([[edge_temp, 800]])
        pred = model.predict(test)
        
        # 5th percentile should score around 0.1 (between 5% and 10%)
        assert 0.05 < pred[0] < 0.3, f"5th percentile should score 0.05-0.3, got {pred[0]}"
    
    def test_minimum_limiting_factor(self):
        """Score should be minimum across variables."""
        data = np.array([
            [15, 800],
            [16, 900],
            [14, 750],
            [17, 850],
            [15, 820]
        ])
        
        model = Bioclim()
        model.fit(data)
        
        # Good temp, bad precip
        test = np.array([[15.5, 500]])
        pred = model.predict(test)
        
        assert pred[0] == 0, "Should be limited by precip (outside range)"
    
    def test_numpy_input(self):
        """Should work with numpy arrays."""
        data = np.array([
            [15, 800],
            [16, 900],
            [14, 750]
        ])
        
        model = Bioclim()
        model.fit(data)
        
        test = np.array([[15, 800]])
        pred = model.predict(test)
        
        assert len(pred) == 1
        assert pred[0] > 0


class TestBioclimRParity:
    """Test parity with R dismo::bioclim."""
    
    @pytest.mark.skip(reason="R reference data not yet generated")
    def test_predict_parity(self):
        """Compare predictions to R dismo."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
