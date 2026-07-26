"""
geometry.py

3D coordinate and grid management for digrt.

v0.1: heliocentric Galactic (l, b, d) <-> Cartesian (x, y, z) transforms,
plus a simple regular sightline sampler. Galactocentric conversion is
included since density models (density.py) are most naturally written
in galactocentric cylindrical coordinates (R, z).

Conventions
-----------
- l, b in degrees (Galactic longitude / latitude)
- d in kpc (heliocentric distance)
- Heliocentric Cartesian: x toward Galactic center, y toward l=90, z toward NGP
- Galactocentric Cartesian: origin at Galactic center, Sun at (-R_sun, 0, z_sun)
  (astropy's convention, we defer to astropy.coordinates.Galactocentric for this)
"""

from __future__ import annotations
import numpy as np
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
import astropy.coordinates as coord


# Sun's position parameters (can be overridden by advanced users)
R_SUN_KPC = 8.122
Z_SUN_KPC = 0.0208


def lbd_to_helio_cartesian(l_deg, b_deg, d_kpc):
    """
    Convert heliocentric Galactic (l, b, d) to heliocentric Cartesian (x, y, z), kpc.

    Parameters
    ----------
    l_deg, b_deg : float or array
        Galactic longitude / latitude in degrees.
    d_kpc : float or array
        Heliocentric distance in kpc.

    Returns
    -------
    x, y, z : ndarray (kpc)
    """
    l = np.radians(np.asarray(l_deg, dtype=float))
    b = np.radians(np.asarray(b_deg, dtype=float))
    d = np.asarray(d_kpc, dtype=float)

    x = d * np.cos(b) * np.cos(l)
    y = d * np.cos(b) * np.sin(l)
    z = d * np.sin(b)
    return x, y, z


def helio_to_galactocentric(x, y, z, r_sun=R_SUN_KPC, z_sun=Z_SUN_KPC):
    """
    Convert heliocentric Cartesian (kpc) to galactocentric cylindrical (R, phi, z), kpc/rad.

    Uses astropy's Galactocentric frame under the hood for consistency with
    literature conventions (so this plugs cleanly into dustmaps / Gaia work later).
    """
    gal = coord.Galactic(
        u=np.asarray(x) * u.kpc,
        v=np.asarray(y) * u.kpc,
        w=np.asarray(z) * u.kpc,
        representation_type="cartesian",
    )
    gc = gal.transform_to(Galactocentric(galcen_distance=r_sun * u.kpc, z_sun=z_sun * u.kpc))
    X = gc.x.to_value(u.kpc)
    Y = gc.y.to_value(u.kpc)
    Z = gc.z.to_value(u.kpc)
    R = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    return R, phi, Z


class Sightline:
    """
    A single line of sight from the Sun toward Galactic (l, b), sampled at
    a set of distances.

    Parameters
    ----------
    l, b : float
        Galactic longitude / latitude, degrees.
    d_min, d_max : float
        Distance range to sample, kpc.
    n_steps : int
        Number of samples along the sightline.
    spacing : {"linear", "log"}
        Sample spacing in distance.
    """

    def __init__(self, l, b, d_min=0.01, d_max=15.0, n_steps=300, spacing="linear"):
        self.l = l
        self.b = b
        if spacing == "linear":
            self.d = np.linspace(d_min, d_max, n_steps)
        elif spacing == "log":
            self.d = np.logspace(np.log10(d_min), np.log10(d_max), n_steps)
        else:
            raise ValueError("spacing must be 'linear' or 'log'")
        self.dl = np.gradient(self.d)  # path-length element per sample, kpc

        self.x, self.y, self.z = lbd_to_helio_cartesian(l, b, self.d)
        self.R, self.phi, self.Z = helio_to_galactocentric(self.x, self.y, self.z)

    def __len__(self):
        return len(self.d)

    def __repr__(self):
        return f"Sightline(l={self.l}, b={self.b}, d=[{self.d.min():.2f},{self.d.max():.2f}] kpc, n={len(self.d)})"
