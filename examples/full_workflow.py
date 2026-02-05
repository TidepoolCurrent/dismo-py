#!/usr/bin/env python3
"""
Full SDM Workflow Example

Demonstrates using dismo-py, enmeval-py, and coordinatecleaner-py together
for a complete species distribution modeling workflow.

Requirements:
    pip install numpy
    # Optional: pip install scipy pandas

Author: TidepoolCurrent
"""

import numpy as np
import sys
sys.path.insert(0, '..')  # For running from examples/

# =============================================================================
# STEP 1: Generate Sample Data
# =============================================================================
print("=" * 60)
print("STEP 1: Generate Sample Data")
print("=" * 60)

np.random.seed(42)

# Simulate occurrence records (lon, lat, temp, precip)
# Species prefers moderate temp (15°C) and moderate precip (800mm)
n_records = 100

# True presence locations (clustered around optimal conditions)
true_lon = np.random.normal(-120, 2, n_records)
true_lat = np.random.normal(37, 1.5, n_records)
true_temp = np.random.normal(15, 2, n_records)
true_precip = np.random.normal(800, 100, n_records)

# Add some problematic records for cleaning demo
# Record at (0, 0)
problem_lon = np.array([0, -120.5, -120.5])
problem_lat = np.array([0, 37.5, 37.5])  # Last two are duplicates
problem_temp = np.array([15, 15, 15])
problem_precip = np.array([800, 800, 800])

lon = np.concatenate([true_lon, problem_lon])
lat = np.concatenate([true_lat, problem_lat])
temp = np.concatenate([true_temp, problem_temp])
precip = np.concatenate([true_precip, problem_precip])

print(f"Total records: {len(lon)}")
print(f"Sample coordinates: ({lon[0]:.2f}, {lat[0]:.2f})")

# =============================================================================
# STEP 2: Clean Coordinates
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Clean Coordinates (coordinatecleaner-py)")
print("=" * 60)

try:
    from coordinatecleaner import clean_coordinates, cc_zero, cc_dupl
    
    # Run cleaning tests
    valid, details = clean_coordinates(lon, lat, 
                                       tests=['val', 'zero', 'dupl'],
                                       return_details=True)
    
    print(f"Records passing all tests: {np.sum(valid)}/{len(valid)}")
    print(f"  - Invalid coordinates: {np.sum(~details['val'])}")
    print(f"  - At (0,0): {np.sum(~details['zero'])}")
    print(f"  - Duplicates: {np.sum(~details['dupl'])}")
    
    # Keep only clean records
    lon_clean = lon[valid]
    lat_clean = lat[valid]
    temp_clean = temp[valid]
    precip_clean = precip[valid]
    
    print(f"\nClean records retained: {len(lon_clean)}")
    
except ImportError:
    print("coordinatecleaner-py not installed, skipping cleaning step")
    lon_clean, lat_clean = lon, lat
    temp_clean, precip_clean = temp, precip

# =============================================================================
# STEP 3: Fit SDM Models
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Fit SDM Models (dismo-py)")
print("=" * 60)

from dismo import Bioclim, Domain, Circle, ConvexHull

# Environmental data for SDMs
env_data = np.column_stack([temp_clean, precip_clean])
coord_data = np.column_stack([lon_clean, lat_clean])

# Fit multiple models
print("\nFitting Bioclim...")
bioclim = Bioclim()
bioclim.fit(env_data)
print(f"  Fitted with {len(env_data)} presence records")

print("\nFitting Domain...")
domain = Domain()
domain.fit(env_data)
print(f"  Fitted with {len(env_data)} presence records")

print("\nFitting Circle (geographic)...")
circle = Circle(threshold=200)  # 200 km
circle.fit(coord_data)
print(f"  Threshold: {circle.threshold_:.0f} km")

print("\nFitting ConvexHull (geographic)...")
hull = ConvexHull()
hull.fit(coord_data)
print(f"  Hull vertices: {len(hull.hull_vertices_)}")

# =============================================================================
# STEP 4: Generate Predictions
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Generate Predictions")
print("=" * 60)

# Create test points
test_env = np.array([
    [15, 800],   # Optimal
    [15, 850],   # Near optimal
    [20, 600],   # Suboptimal
    [30, 400],   # Outside range
])

test_coords = np.array([
    [-120, 37],    # In range
    [-115, 35],    # Near range
    [-100, 40],    # Far from range
])

print("\nEnvironmental predictions:")
print("  Test point      | Bioclim | Domain")
print("  " + "-" * 40)
bc_pred = bioclim.predict(test_env)
dom_pred = domain.predict(test_env)
for i, (t, p) in enumerate(test_env):
    print(f"  ({t:.0f}°C, {p:.0f}mm) | {bc_pred[i]:.3f}   | {dom_pred[i]:.3f}")

print("\nGeographic predictions:")
print("  Test point       | Circle | Hull")
print("  " + "-" * 40)
circ_pred = circle.predict(test_coords)
hull_pred = hull.predict(test_coords)
for i, (lo, la) in enumerate(test_coords):
    print(f"  ({lo:.0f}, {la:.0f})     | {circ_pred[i]:.3f}  | {hull_pred[i]:.3f}")

# =============================================================================
# STEP 5: Evaluate Models
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Evaluate Models (enmeval-py)")
print("=" * 60)

try:
    from enmeval import calc_auc, calc_cbi
    
    # Generate pseudo-absence (background) data
    np.random.seed(123)
    bg_temp = np.random.uniform(5, 35, 500)
    bg_precip = np.random.uniform(200, 1500, 500)
    bg_env = np.column_stack([bg_temp, bg_precip])
    
    # Get predictions for presence and background
    pres_pred = bioclim.predict(env_data)
    bg_pred = bioclim.predict(bg_env)
    
    # Calculate AUC
    auc = calc_auc(pres_pred, bg_pred)
    print(f"\nBioclim AUC: {auc:.3f}")
    
    # Calculate CBI
    from enmeval.boyce import calc_cbi
    cbi = calc_cbi(pres_pred, bg_pred)
    print(f"Bioclim CBI: {cbi:.3f}")
    
    if auc > 0.7:
        print("\n✓ Model performs better than random (AUC > 0.7)")
    
except ImportError:
    print("enmeval-py not installed, skipping evaluation step")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("WORKFLOW COMPLETE")
print("=" * 60)
print("""
This workflow demonstrated:
1. Data cleaning with coordinatecleaner-py
2. Multiple SDM algorithms with dismo-py
3. Model evaluation with enmeval-py

For real applications:
- Use actual occurrence data from GBIF/iNaturalist
- Use real environmental layers (WorldClim, etc.)
- Apply spatial cross-validation (enmeval partitioning)
- Compare multiple models and use ensemble predictions

See also:
- https://github.com/TidepoolCurrent/dismo-py
- https://github.com/TidepoolCurrent/enmeval-py
- https://github.com/TidepoolCurrent/coordinatecleaner-py
""")
