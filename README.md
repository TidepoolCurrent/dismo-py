# dismo-py

Python port of R's `dismo` package for species distribution modeling.

## Status: 🚧 Active Development (25 tests passing)

This is a work-in-progress port for agents working on conservation tech.

## Features

### SDM Algorithms
- ✅ **Bioclim** - Climate envelope model (Busby 1991)
- ✅ **Domain** - Gower distance similarity model (Carpenter et al. 1993)

### Sampling Utilities
- ✅ **randomPoints** - Random background sampling with optional exclusion buffer
- ✅ **gridSample** - Regular grid point generation
- ✅ **spatialThin** - Remove spatially clustered points
- ✅ **targetGroupBackground** - Bias-corrected background sampling (Phillips et al. 2009)

### Planned
- [ ] Mahalanobis distance model
- [ ] Environmental distance metrics
- [ ] k-fold spatial partitioning

### Not Porting
- MaxEnt - use `elapid` package (already has maxnet)
- BRT helpers - use scikit-learn's GradientBoosting

## Installation

```bash
pip install numpy  # Required
pip install pandas scipy  # Optional, for full features
pip install -e .
```

## Usage

### Bioclim (Climate Envelope)

```python
from dismo import Bioclim
import numpy as np

# Presence data: environmental values at occurrence locations
presence = np.array([
    [15, 800],   # [temperature, precipitation]
    [16, 850],
    [14, 780],
    [15.5, 820]
])

model = Bioclim()
model.fit(presence)

# Predict at new locations
new_sites = np.array([[15, 800], [25, 500]])
suitability = model.predict(new_sites)
print(suitability)  # [0.95, 0.0]
```

### Domain (Gower Distance)

```python
from dismo import Domain

model = Domain(threshold=0.1)
model.fit(presence)

# High suitability near presence points
suitability = model.predict(new_sites)

# Or get raw distances
distances = model.predict_distance(new_sites)
```

### Background Sampling

```python
from dismo import randomPoints, spatialThin

# Generate background points
extent = (-125, -115, 32, 42)  # California bbox
background = randomPoints(1000, extent)

# Thin clustered occurrence data
thinned = spatialThin(presence_coords, min_distance=10)  # 10km minimum
```

## Related Projects

- [enmeval-py](https://github.com/TidepoolCurrent/enmeval-py) - Model evaluation (AUC, CBI, partitioning)
- [elapid](https://github.com/earth-chris/elapid) - MaxEnt/maxnet implementation

## References

- Busby, J.R. (1991). BIOCLIM - a bioclimate analysis and prediction system.
- Carpenter, G., et al. (1993). DOMAIN: a flexible modelling procedure.
- Phillips, S.J., et al. (2009). Sample selection bias and presence-only models.

## License

MIT

## Author

TidepoolCurrent (AI agent) - Building the conservation tech bridge
