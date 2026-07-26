"""
ionization.py

A physically motivated (but deliberately simplified) local ionization
parameter field, replacing the v0.2 placeholder scalar `ionization_boost`
knob with something tied to actual 3D gas geometry.

    U = Phi_ion / (n_H c)      (dimensionless ionization parameter)

We do not yet track individual ionizing sources -- that's the natural
future extension (an OB-star population feeding a real Phi_ion(x,y,z)
from stellar population synthesis / Starburst99-type outputs). Instead we
use the standard reduced picture used in DIG modeling: ionizing photon
flux from the thin OB-star layer near the midplane dilutes geometrically
with distance from the plane (~ 1/r_eff^2, with r_eff = sqrt(z^2 + r0^2)
regularizing the midplane singularity), and U is evaluated using the
ACTUAL local electron density from the density model -- so U falls
self-consistently with both height |z| and with locally underdense gas,
which is exactly why this model produces DIG-like low-ionization-line
enhancement at low density / large |z| without being told to.

This is a qualitative/directional model, not a quantitative photoionization
grid (MAPPINGS/CLOUDY). It gets the SIGN and rough functional form of the
effect right (elevated [NII]/Ha, [SII]/Ha in diffuse, off-plane gas -- the
best-established empirical DIG signature; see Haffner et al. 2009,
Rev. Mod. Phys. for a review), but the absolute normalization and the more
degenerate [OIII]/Hb behavior (dilution lowers it, spectral hardening from
preferential absorption of soft photons can raise it -- real DIG shows
both) should not be trusted quantitatively. A literature photoionization
grid is the natural v0.6+ upgrade once the package needs that.

KNOWN LIMITATION (found during v0.2 verification, not yet fixed): because
U = flux/(n_e c) uses the *same* exponential disk density field for both
the photon-dilution geometry and the denominator n_e, the exponential
density falloff with height eventually outpaces the 1/z^2 geometric
dilution at large |z| (roughly z > 1-1.5 kpc for typical h_z ~ 0.5 kpc
disks), causing U -- and therefore the low-ionization line enhancement --
to turn over and partially reverse at very large heights instead of
continuing to rise. The trend is correctly monotonic (and quantitatively
reasonable) over the height range that covers the great majority of real
WHAM/LVM DIG detections (|z| below ~1-1.5 kpc); it should not be trusted
in the extraplanar/thick-disk regime without decoupling the photon-source
distribution from the local gas density more carefully.
"""

from __future__ import annotations
import numpy as np


class IonizationParameterModel:
    """
    Parameters
    ----------
    U_ref : float
        Reference ionization parameter at the calibration point
        (z=0-ish "inside an HII region" scale, n_e=n_ref). Typical
        Galactic HII regions have log U ~ -3 to -2.5; default log U = -3.2.
    n_ref_cm3 : float
        Reference electron density (cm^-3) at which U_ref applies. Default
        (0.15 cm^-3) is set to match the near-plane density regime of the
        smooth single-phase disk models in density.py -- NOT a compact HII
        region density. This is a real limitation worth flagging: this
        version doesn't yet distinguish discrete, dense HII-region clumps
        from the smooth diffuse background (a genuine two-phase medium is
        planned for a later version); if you swap in a density field with
        a different characteristic near-plane density, you should re-tune
        n_ref_cm3 (and/or U_ref) to match, or this ionization-parameter
        scaling will silently give unphysical results, as it briefly did
        during v0.2 development/verification.
    r0_kpc : float
        Regularization scale for the OB-star layer thickness / minimum
        source-to-gas distance, kpc. Prevents a 1/z^2 singularity at the
        midplane and sets how quickly U falls off with height.
    """

    def __init__(self, U_ref=10 ** -3.2, n_ref_cm3=0.15, r0_kpc=0.05):
        self.U_ref = U_ref
        self.n_ref = n_ref_cm3
        self.r0 = r0_kpc

    def U(self, z_kpc, n_e_cm3):
        """Local ionization parameter given height above plane and n_e."""
        z = np.abs(np.asarray(z_kpc, dtype=float))
        n_e = np.clip(np.asarray(n_e_cm3, dtype=float), 1e-4, None)
        r_eff = np.sqrt(z ** 2 + self.r0 ** 2)
        dilution = (self.r0 / r_eff) ** 2
        return self.U_ref * (self.n_ref / n_e) * dilution


# Power-law index p in ratio(U) = ratio_ref * (U/U_ref)^p, per ion.
# Sign/species choice follows the well-established qualitative DIG trend:
# low-ionization lines ([NII], [SII]) strengthen as U drops; [OIII] is left
# only weakly coupled given the real degeneracy noted above.
ION_U_POWER = {
    "O3": 0.15,
    "N2": -0.40,
    "S2": -0.50,
}

