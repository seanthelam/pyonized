"""
geometry.py

3D coordinate and grid management.

v0.1: Galactic (l, b, d) <-> Cartesian (x, y, z) transforms,
plus a simple regular sightline sampler. Galactocentric conversion is
included since density models are naturally written in galactocentric
cylindrical coordinates (R,z).

Conventions
-----------
- l, b in degrees (Galactic longitude / latitude)
- d in kpc (heliocentricc distance)

"""

from __future__ import annotations
import numpy as np
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
import astropy.coordinates as coord
