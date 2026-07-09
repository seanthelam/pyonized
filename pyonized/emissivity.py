"""
emissivity.py

Voxel-level line emissivity. This will, for now, implement a simplified Case B
recombination emissivity for Halpha (Osterbrock and Ferland),
parameterized by electron density and temperature, as a stand-in until I add
pyneb for multi-line, abundance-dependent emissivity
"""

from __future__ import annotations
import numpy as np
import pyneb as pn

# Case B effective recombination coefficient for Halpha at T=1e4 K, cm^3/s
ALPHA_HA_1E4 = 1.17e-13


def halpha_emissivity(n_e, T_e=1.0e4, n_p=None):
    """
    Simplified Case B Halpha volume emissivity.

    Parameters
    ----------
    n_e : array
        Electron density, cm^-3.
    T_e : array or float
        Electron temperature, K. Default 1e4 K.
    n_p : array or None
        Proton (ionized H) density, cm^-3. Defaults to n_e (fully ionized,
        pure hydrogen approximation).

    Returns
    -------
    j : ndarray
        Volume emissivity, erg / s / cm^3.
    """
    n_e = np.asarray(n_e, dtype=float)
    T4 = np.asarray(T_e, dtype=float) / 1.0e4
    if n_p is None:
        n_p = n_e
    alpha_eff = ALPHA_HA_1E4 * T4 ** (-0.94)
    # ~3e-12 erg is the Halpha photon energy (hc/lambda, lambda=656.3nm)
    E_Ha = 3.03e-12
    return alpha_eff * n_e * n_p * E_Ha


# Registry so later versions (pyneb-backed) can be swapped in without
# changing calling code in los.py
EMISSIVITY_MODELS = {
    "Halpha": halpha_emissivity,
}

"""
pyneb implementation

v0.2: full atomic-physics line emissivity via pyneb, covering the standard
optical BPT diagnostic set: Halpha, Hbeta (recombination), [OIII]5007,
[NII]6584, [SII]6717, [SII]6731 (collisionally excited lines, CELs).

Recombination lines:
    j_line = alpha_eff(T_e, n_e) * n_e * n_p        [erg/s/cm^3]

Collisionally excited lines:
    j_line = coeff(T_e, n_e) * n_e * n_ion            [erg/s/cm^3]
    n_ion = ionic_abundance_ratio(R) * n_p             (see abundance.py)

Atom/RecAtom objects are created once per process (pyneb reads atomic data
from disk) and cached at module level, since they're expensive to
instantiate repeatedly inside a LOS loop over many sightlines.
"""

# --- cached atomic data objects ---
_H1 = pn.RecAtom("H", 1)
_O3 = pn.Atom("O", 3)
_N2 = pn.Atom("N", 2)
_S2 = pn.Atom("S", 2)

# line registry: name -> (kind, atom, selector_kwarg, selector_value, rest wavelength nm, ion_key)
LINES = {
    "Halpha": dict(kind="rec", atom=_H1, sel={"label": "3_2"}, wavelength_nm=656.28, ion_key=None),
    "Hbeta":  dict(kind="rec", atom=_H1, sel={"label": "4_2"}, wavelength_nm=486.13, ion_key=None),
    "OIII5007": dict(kind="cel", atom=_O3, sel={"wave": 5007}, wavelength_nm=500.68, ion_key="O3"),
    "NII6584":  dict(kind="cel", atom=_N2, sel={"wave": 6584}, wavelength_nm=658.35, ion_key="N2"),
    "SII6717":  dict(kind="cel", atom=_S2, sel={"wave": 6717}, wavelength_nm=671.6,  ion_key="S2"),
    "SII6731":  dict(kind="cel", atom=_S2, sel={"wave": 6731}, wavelength_nm=673.08, ion_key="S2"),
}


def line_emissivity(line_name, n_e, T_e, R=None, z=None, ionization_state=None, n_p=None):
    """
    Physical volume emissivity for a named line, along an array of LOS samples.

    Parameters
    ----------
    line_name : str
        One of digrt.emissivity_pyneb.LINES keys.
    n_e : array
        Electron density, cm^-3, per LOS sample.
    T_e : array or float
        Electron temperature, K, per LOS sample (or scalar).
    R : array or None
        Galactocentric radius, kpc, per LOS sample -- required for CELs
        (used to evaluate the ionic abundance metallicity gradient).
    z : array or None
        Height above the Galactic plane, kpc, per LOS sample -- used (along
        with n_e) to evaluate the local ionization parameter for CELs.
        If omitted, the ionization-parameter modulation is skipped and only
        the metallicity gradient is applied.
    ionization_state : abundance.IonizationState or None
        Required for CELs; ignored for recombination lines.
    n_p : array or None
        Proton density, cm^-3. Defaults to n_e (fully ionized H approx).

    Returns
    -------
    j : ndarray, erg / s / cm^3
    """
    info = LINES[line_name]
    n_e = np.asarray(n_e, dtype=float)
    T_e = np.broadcast_to(np.asarray(T_e, dtype=float), n_e.shape).astype(float)
    if n_p is None:
        n_p = n_e

    # avoid pyneb warnings/errors on exactly-zero density
    n_e_safe = np.clip(n_e, 1e-6, None)

    coeff = info["atom"].getEmissivity(tem=T_e, den=n_e_safe, product=False, **info["sel"])
    coeff = np.asarray(coeff, dtype=float)

    if info["kind"] == "rec":
        j = coeff * n_e * n_p
    else:
        if ionization_state is None or R is None:
            raise ValueError(f"{line_name} is a collisionally excited line and requires "
                              f"`R` and `ionization_state` (see abundance.IonizationState).")
        ion_ratio = ionization_state.ratio(info["ion_key"], R, z=z, n_e=n_e_safe)
        n_ion = ion_ratio * n_p
        j = coeff * n_e * n_ion

    # zero out emission where there's effectively no ionized gas
    j = np.where(n_e > 1e-6, j, 0.0)
    return j


def wavelength_nm(line_name):
    return LINES[line_name]["wavelength_nm"]

