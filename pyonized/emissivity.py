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
