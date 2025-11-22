from __future__ import annotations

# Re-export public API for pygcm.ecology
from .adapter import EcologyAdapter
from .phyto import PhytoManager
from .population import PopulationManager
from .spectral import (
    SpectralBands,
    band_weights_from_mode,
    default_leaf_reflectance,
    make_bands,
)
from .types import WeatherDaily, WeatherInstant

__all__ = [
    "EcologyAdapter",
    "PopulationManager",
    "SpectralBands",
    "make_bands",
    "band_weights_from_mode",
    "default_leaf_reflectance",
    "WeatherInstant",
    "WeatherDaily",
    "PhytoManager",
]
