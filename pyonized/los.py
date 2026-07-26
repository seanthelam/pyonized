"""
los.py

Line-of-sight integration engine: combines a gas density field (via an emissivity model)
and a dust density field (via an extinction curve) to produce intrinsic and dust-attenuated
emission along an arbitrary sightline.

PLAN RIGHT NOW:
This is the computational core the rest of the package builds on: BPT
geometric-bias mapping (v0.4) is just running this twice (line-ratio
numerator/denominator) under different dust geometries and comparing;
all-sky maps (v0.5) are just looping this over a HEALPix grid of (l,b).

Units note (v0.1): densities are in whatever units the density model
returns; `dust_column_per_kpc` converts integrated dust density into the
"reddening" units the ExtinctionCurve was calibrated against (mag per
unit reddening -> here we treat 1 unit of column as producing
`dust_column_per_kpc` magnitudes of A(lambda_ref) per kpc at unit density).
This is a deliberately simple, swappable calibration -- v0.3 replaces it
with a real dustmap query in physical E(B-V) or equivalent units.
"""

from __future__ import annotations
import numpy as np

class LOSResult:
  """Container for the outputs of a line-of-sight integration."""

    def __init__(self, sightline, wavelength_nm, j_local, reddening_cum,
                 transmission, dI_intrinsic, dI_attenuated):
        self.sightline = sightline
        self.wavelength_nm = wavelength_nm
        self.j_local = j_local                  # local emissivity along LOS
        self.reddening_cum = reddening_cum       # cumulative foreground reddening
        self.transmission = transmission         # transmission fraction at each step
        self.dI_intrinsic = dI_intrinsic         # per-step intrinsic contribution
        self.dI_attenuated = dI_attenuated       # per-step attenuated contribution

    @property
    def I_intrinsic(self):
        """Total intrinsic (dust-free) integrated intensity along the LOS."""
        return np.sum(self.dI_intrinsic)

    @property
    def I_attenuated(self):
        """Total dust-attenuated integrated intensity along the LOS."""
        return np.sum(self.dI_attenuated)

    @property
    def cumulative_intrinsic(self):
        return np.cumsum(self.dI_intrinsic)

    @property
    def cumulative_attenuated(self):
        return np.cumsum(self.dI_attenuated)

    @property
    def total_extinction_mag(self):
        """Total foreground extinction (mag) at the far end of the sightline."""
        return self.reddening_cum[-1] * self._A_per_unit_reddening

    def __repr__(self):
        return (f"LOSResult(l={self.sightline.l}, b={self.sightline.b}, "
                f"lambda={self.wavelength_nm}nm, "
                f"I_intrinsic={self.I_intrinsic:.4g}, I_attenuated={self.I_attenuated:.4g}, "
                f"attenuation_frac={1 - self.I_attenuated/self.I_intrinsic:.3f})")


def integrate_sightline(sightline, gas_density, dust_density, extinction_curve,
                         emissivity_func, wavelength_nm,
                         n_e_scale=1.0, dust_column_per_kpc=0.05, T_e=1.0e4):
    """
    Integrate intrinsic and dust-attenuated line emission along a sightline.

    Parameters
    ----------
    sightline : geometry.Sightline
    gas_density : density.CompositeDensity (or any object with .evaluate(sightline))
        Governs n_e(l) = n_e_scale * gas_density.evaluate(sightline).
    dust_density : density.CompositeDensity
        Governs the dust column build-up along the LOS.
    extinction_curve : extinction.ExtinctionCurve
    emissivity_func : callable(n_e, T_e=...) -> volume emissivity
        e.g. emissivity.halpha_emissivity
    wavelength_nm : float
        Rest wavelength of the line, nm.
    n_e_scale : float
        Converts dimensionless gas density field to physical n_e in cm^-3.
    dust_column_per_kpc : float
        Reddening-curve "column" produced per kpc at unit dust density
        (calibration knob for the synthetic dust model; see module docstring).
    T_e : float
        Electron temperature, K.

    Returns
    -------
    LOSResult
    """
    n_e = n_e_scale * gas_density.evaluate(sightline)
    j_local = emissivity_func(n_e, T_e=T_e)  # erg/s/cm^3, per LOS sample

    dust_local = dust_density.evaluate(sightline)  # dimensionless density
    dl_kpc = sightline.dl
    # cumulative reddening-curve "column" from the observer out to each sample
    reddening_cum = np.cumsum(dust_local * dl_kpc) * dust_column_per_kpc

    transmission = extinction_curve.transmission(wavelength_nm, reddening_cum)

    # kpc -> cm for a physically dimensioned path integral (erg/s/cm^3 * cm = erg/s/cm^2)
    KPC_TO_CM = 3.0857e21
    dl_cm = dl_kpc * KPC_TO_CM

    dI_intrinsic = j_local * dl_cm
    dI_attenuated = dI_intrinsic * transmission

    result = LOSResult(sightline, wavelength_nm, j_local, reddening_cum,
                        transmission, dI_intrinsic, dI_attenuated)
    result._A_per_unit_reddening = extinction_curve.A(wavelength_nm)
    return result


def integrate_sightline_multiline(sightline, gas_density, dust_density, extinction_curve,
                                   line_names, ionization_state,
                                   n_e_scale=1.0, dust_column_per_kpc=0.05, T_e=1.0e4):
    """
    v0.2: integrate several pyneb-backed lines along the same sightline in
    one call, sharing the gas/dust geometry and T_e -- this is what makes
    line-ratio (BPT) work possible, since intrinsic and attenuated ratios
    need every line evaluated on the *same* physical LOS.

    Parameters
    ----------
    line_names : list of str
        Keys into emissivity_pyneb.LINES, e.g. ["Halpha","Hbeta","OIII5007","NII6584","SII6717","SII6731"].
    ionization_state : abundance.IonizationState
        Supplies ionic abundance ratios (and the ionization_boost knob) for CELs.
    (other parameters as in integrate_sightline)

    Returns
    -------
    dict[str, LOSResult]
    """
    from . import emissivity_pyneb as epn

    n_e = n_e_scale * gas_density.evaluate(sightline)
    dust_local = dust_density.evaluate(sightline)
    dl_kpc = sightline.dl
    reddening_cum = np.cumsum(dust_local * dl_kpc) * dust_column_per_kpc

    KPC_TO_CM = 3.0857e21
    dl_cm = dl_kpc * KPC_TO_CM

    results = {}
    for name in line_names:
        j_local = epn.line_emissivity(name, n_e, T_e, R=sightline.R, z=sightline.Z,
                                       ionization_state=ionization_state)
        wl = epn.wavelength_nm(name)
        transmission = extinction_curve.transmission(wl, reddening_cum)

        dI_intrinsic = j_local * dl_cm
        dI_attenuated = dI_intrinsic * transmission

        result = LOSResult(sightline, wl, j_local, reddening_cum,
                            transmission, dI_intrinsic, dI_attenuated)
        result._A_per_unit_reddening = extinction_curve.A(wl)
        results[name] = result

    return results


def line_ratio(results, num_line, den_line, attenuated=True):
    """Convenience: integrated flux ratio num_line/den_line from a results dict."""
    num = results[num_line].I_attenuated if attenuated else results[num_line].I_intrinsic
    den = results[den_line].I_attenuated if attenuated else results[den_line].I_intrinsic
    return num / den
