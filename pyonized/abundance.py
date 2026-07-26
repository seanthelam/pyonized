"""
abundance.py

Ionic abundance models: n_ion / n_H+ ratios needed to turn pyneb's
per-ion emissivity coefficients into physical volume emissivities.

v0.2 approximation: abundance ratios are set per-ion as either a constant
or a simple radial (galactocentric R) linear gradient in log space, which
is the standard first-order description of Galactic O/H, N/H, S/H
gradients (e.g. ~ -0.04 dex/kpc for O/H). This does NOT yet solve
photoionization equilibrium for the ionization fraction (i.e. it doesn't
know that low-ionization-parameter DIG boosts [NII]/Ha and [SII]/Ha
relative to HII regions) -- that physics is exactly what makes DIG
diagnostics interesting, and is deferred to a proper ionization-parameter
-dependent grid (planned v0.3/v0.4, informed by e.g. Haffner-style
DIG models). For now, treat these as user-tunable knobs: advanced users
can hand-set a harder or softer ionization state via `ionization_boost`.

All ratios are ionic abundances RELATIVE TO H+ (i.e. n_X+i / n_H+), which
is exactly what multiplies pyneb's getEmissivity() coefficient alongside
n_e to get a physical volume emissivity.
"""

from __future__ import annotations
import numpy as np

from .ionization import IonizationParameterModel, ION_U_POWER


class RadialAbundanceGradient:
    """
    log10(n_ion/n_H+) = log10_ratio_solar + slope_dex_per_kpc * (R - R_0)

    Parameters
    ----------
    log10_ratio_solar : float
        log10(n_ion/n_H+) at R = R_0 (solar circle by default).
    slope_dex_per_kpc : float
        Radial gradient, dex/kpc (negative = more enriched toward center).
    R_0 : float
        Reference galactocentric radius, kpc (default 8.122, solar).
    """

    def __init__(self, log10_ratio_solar, slope_dex_per_kpc=-0.04, R_0=8.122):
        self.log10_ratio_solar = log10_ratio_solar
        self.slope = slope_dex_per_kpc
        self.R_0 = R_0

    def __call__(self, R):
        R = np.asarray(R, dtype=float)
        log_ratio = self.log10_ratio_solar + self.slope * (R - self.R_0)
        return 10.0 ** log_ratio


# Reasonable literature-typical DIG/HII ionic abundance ratios at the solar
# circle (log10 n_ion/n_H+), roughly consistent with commonly used DIG/HII
# BPT-diagnostic values. These are starting points, not measurements --
# advanced users should override per their science case.
# Reasonable literature-typical DIG/HII ionic abundance ratios at the solar
# circle (log10 n_ion/n_H+), calibrated against pyneb emissivity coefficients
# (T_e=1e4 K, n_e=100 cm^-3) to reproduce approximate star-forming-locus line
# ratios ([NII]/Ha ~ 0.4, [OIII]/Hb ~ 1.0, [SII]6717/Ha ~ 0.15) at R = R_sun.
# These are starting points, not measurements -- advanced users should
# override per their science case.
DEFAULT_IONIC_ABUNDANCE = {
    "O3": RadialAbundanceGradient(log10_ratio_solar=-4.45, slope_dex_per_kpc=-0.04),
    "N2": RadialAbundanceGradient(log10_ratio_solar=-4.62, slope_dex_per_kpc=-0.05),
    "S2": RadialAbundanceGradient(log10_ratio_solar=-5.73, slope_dex_per_kpc=-0.05),
}


class IonizationState:
    """
    Bundles ionic abundance models (metallicity gradient) with a spatially
    varying ionization parameter field (see ionization.py) that scales the
    low-ionization line ratios up in diffuse, off-plane gas -- this is what
    makes DIG-like [NII]/Ha, [SII]/Ha enhancement fall out of the actual 3D
    geometry rather than being a manual scalar fudge.

    An optional extra `ionization_boost` scalar remains available as a
    manual override on top of the U-dependent scaling, for advanced users
    who want to hand-tune beyond the built-in geometric model (default 1.0,
    no-op).
    """

    def __init__(self, ionic_abundance=None, ionization_model=None, ionization_boost=1.0):
        self.ionic_abundance = dict(ionic_abundance or DEFAULT_IONIC_ABUNDANCE)
        if ionization_model is None:
            ionization_model = IonizationParameterModel()
        self.ionization_model = ionization_model
        self.ionization_boost = ionization_boost

    def ratio(self, ion_key, R, z=None, n_e=None):
        """
        Ionic abundance ratio n_ion/n_H+ at galactocentric radius R
        (metallicity gradient), optionally modulated by the local
        ionization parameter U(z, n_e) if z and n_e are supplied.
        """
        base = self.ionic_abundance[ion_key](R)

        if self.ionization_model is not None and z is not None and n_e is not None:
            U = self.ionization_model.U(z, n_e)
            p = ION_U_POWER.get(ion_key, 0.0)
            base = base * (U / self.ionization_model.U_ref) ** p

        return base * self.ionization_boost
