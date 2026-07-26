"""
density.py

Synthetic, parametric 3D density fields for gas and dust — placeholders for
v0.1 so the rest of the pipeline (LOS integration, extinction, emissivity)
can be built and visually validated before wiring in real 3D dust maps
(Bayestar19 / DECaPS2 / Edenhofer2023 via the `dustmaps` package) in v0.3.

All models are functions of galactocentric cylindrical (R, z) in kpc and
return a dimensionless local density (arbitrary normalization — the
LOS integrator turns integrated density into a physical column via a
calibration constant supplied by the user / extinction model).

Includes simple structure (a foreground "dust clump") to demonstrate the
geometric effects the package is meant to study.
"""

from __future__ import annotations
import numpy as np

class ExponentialDiskDensity:
    """
    Double-exponential disk: n(R, z) = n0 * exp(-R/h_R) * exp(-|z|/h_z)

    Typical defaults are loosely tuned to the local ISM dust layer
    (h_R ~ 3 kpc, h_z ~ 0.1-0.15 kpc) or DIG (h_z ~ 0.5-1 kpc) — override
    per use case.
    """

    def __init__(self, n0=1.0, h_R=3.0, h_z=0.15):
        self.n0 = n0
        self.h_R = h_R
        self.h_z = h_z

    def __call__(self, R, z):
        R = np.asarray(R, dtype=float)
        z = np.asarray(z, dtype=float)
        return self.n0 * np.exp(-R / self.h_R) * np.exp(-np.abs(z) / self.h_z)


class GaussianClump:
    """
    A localized 3D Gaussian density enhancement, in galactocentric (R, phi, z)
    but specified by its heliocentric (x, y, z) center for convenience — used
    to inject foreground dust clumps / gas clouds to test geometric effects.
    """

    def __init__(self, center_xyz_helio, amplitude=5.0, sigma=0.3):
        self.center = np.asarray(center_xyz_helio, dtype=float)
        self.amplitude = amplitude
        self.sigma = sigma

    def __call__(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r2 = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 + (z - self.center[2]) ** 2
        return self.amplitude * np.exp(-0.5 * r2 / self.sigma ** 2)


class CompositeDensity:
    """Sum of a smooth disk (evaluated in R,z) plus optional clumps (evaluated in x,y,z)."""

    def __init__(self, disk: ExponentialDiskDensity, clumps=None):
        self.disk = disk
        self.clumps = clumps or []

    def evaluate(self, sightline):
        """Evaluate total density along a geometry.Sightline object."""
        n = self.disk(sightline.R, sightline.Z)
        for clump in self.clumps:
            n = n + clump(sightline.x, sightline.y, sightline.z)
        return n
