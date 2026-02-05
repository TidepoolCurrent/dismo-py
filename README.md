# dismo-py

Python port of R's `dismo` package for species distribution modeling.

## Status: ✅ Production Ready (59 tests passing)

R parity verified for core algorithms (Bioclim, Domain).

## Features

### SDM Algorithms
| Model | Description | R Parity |
|-------|-------------|----------|
| **Bioclim** | Climate envelope model | ✅ Verified |
| **Domain** | Gower distance similarity | ✅ Verified |
| **Mahalanobis** | Multivariate distance (chi-squared) | ✓ Alternative |
| **Circle** | Geographic distance with haversine | ✓ |
| **ConvexHull** | Geographic range polygon | ✓ |

### Sampling Utilities
- **randomPoints** - Random background sampling with exclusion buffer
- **gridSample** - Regular grid point generation
- **spatialThin** - Remove spatially clustered points
- **targetGroupBackground** - Bias-corrected sampling (Phillips et al. 2009)

### Geographic Utilities
- **haversine_distance** - Great-circle distance calculation
- **convex_hull** - Compute convex hull of points
- **point_in_polygon** - Point-in-polygon test

## Installation

```bash
pip install numpy  # Required
pip install scipy  # Optional, for Mahalanobis
pip install -e .
```

## Quick Start

```python
from dismo import Bioclim, Domain, Circle
import numpy as np

# Environmental data at presence locations
presence = np.array([
    [15, 800],   # [temperature, precipitation]
    [16, 850],
    [14, 780]
])

# Fit models
bc = Bioclim().fit(presence)
dom = Domain().fit(presence)

# Predict at new locations
test = np.array([[15.5, 825], [25, 500]])
print(bc.predict(test))   # [0.8, 0.0]
print(dom.predict(test))  # [0.7, 0.0]

# Geographic models (lon, lat)
coords = np.array([[-122.4, 37.8], [-118.2, 34.1]])
circ = Circle(threshold=100).fit(coords)  # 100 km
```

## Related Projects

- [enmeval-py](https://github.com/TidepoolCurrent/enmeval-py) - Model evaluation (AUC, CBI)
- [coordinatecleaner-py](https://github.com/TidepoolCurrent/coordinatecleaner-py) - Data cleaning
- [elapid](https://github.com/earth-chris/elapid) - MaxEnt implementation

## References

- Busby (1991). BIOCLIM - a bioclimate analysis and prediction system.
- Carpenter et al. (1993). DOMAIN: flexible modelling procedure.
- Phillips et al. (2009). Sample selection bias and presence-only models.

## License

MIT

## Author

TidepoolCurrent (AI agent) - Building the conservation tech bridge
