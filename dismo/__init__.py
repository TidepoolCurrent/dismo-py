"""
dismo-py: Species Distribution Modeling

Python port of R's dismo package.
"""

from .bioclim import Bioclim
from .domain import Domain
from .mahal import Mahalanobis
from .circle import Circle, haversine_distance
from .hull import ConvexHull, convex_hull, point_in_polygon
from .sampling import randomPoints, gridSample, targetGroupBackground, spatialThin
from .biovars import biovars, biovars_from_grids
from .data import gbif, inat, worldclim
from .evaluate import evaluate, threshold

__version__ = "0.1.0"
__all__ = [
    # SDM Models
    "Bioclim", 
    "Domain",
    "Mahalanobis",
    "Circle",
    "ConvexHull",
    # Geographic utilities
    "haversine_distance",
    "convex_hull",
    "point_in_polygon",
    # Sampling
    "randomPoints",
    "gridSample", 
    "targetGroupBackground",
    "spatialThin",
    # Climate data
    "biovars",
    "biovars_from_grids",
    # Data acquisition
    "gbif",
    "inat",
    "worldclim",
    # Evaluation
    "evaluate",
    "threshold",
]
