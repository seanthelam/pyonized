"""
visualize.py

Plotting helpers. v0.1: sightline diagnostic panel (density/emissivity/
extinction/flux vs. distance) and a raw extinction-curve plot. Later
versions add all-sky HEALPix mollweide maps and BPT diagram overlays.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_extinction_curve(ext_curve, ax=None, label=None):
    ax = ax or plt.gca()
    wl = np.logspace(np.log10(ext_curve.wavelength_nm.min()),
                      np.log10(ext_curve.wavelength_nm.max()), 300)
    ax.plot(wl, ext_curve.A(wl), lw=2, label=label)
    ax.scatter(ext_curve.wavelength_nm, ext_curve.extinction, s=14, color="k", zorder=5,
                label="tabulated points" if label is None else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"$A_\lambda$ (curve units)")
    ax.set_title("Extinction curve")
    ax.legend(frameon=False)
    return ax


def plot_los_diagnostics(result, gas_density, dust_density, sightline, figsize=(11, 8)):
    """
    Four-panel diagnostic: gas/dust density vs distance, local emissivity,
    cumulative foreground extinction (mag), and cumulative intrinsic vs.
    attenuated intensity.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    d = sightline.d

    n_e_field = gas_density.evaluate(sightline)
    dust_field = dust_density.evaluate(sightline)

    ax = axes[0, 0]
    ax.plot(d, n_e_field, color="tab:blue", label="gas density (n$_e$, arb.)")
    ax2 = ax.twinx()
    ax2.plot(d, dust_field, color="tab:orange", label="dust density (arb.)")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel("gas density", color="tab:blue")
    ax2.set_ylabel("dust density", color="tab:orange")
    ax.set_title(f"Density along LOS (l={sightline.l}, b={sightline.b})")

    ax = axes[0, 1]
    ax.plot(d, result.j_local, color="tab:green")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel(r"local emissivity $j$ [erg s$^{-1}$ cm$^{-3}$]")
    ax.set_yscale("log")
    ax.set_title(f"Line emissivity ({result.wavelength_nm:.1f} nm)")

    ax = axes[1, 0]
    A_lambda = result._A_per_unit_reddening if hasattr(result, "_A_per_unit_reddening") else 1.0
    ax.plot(d, result.reddening_cum * A_lambda, color="tab:red")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel(r"cumulative foreground $A_\lambda$ [mag]")
    ax.set_title("Foreground extinction build-up")

    ax = axes[1, 1]
    ax.plot(d, result.cumulative_intrinsic / result.cumulative_intrinsic[-1],
            color="tab:blue", label="intrinsic (dust-free)")
    ax.plot(d, result.cumulative_attenuated / result.cumulative_intrinsic[-1],
            color="tab:red", label="dust-attenuated")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel("cumulative intensity (norm. to total intrinsic)")
    ax.set_title(f"Attenuation: {100*(1 - result.I_attenuated/result.I_intrinsic):.1f}% of flux lost")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig, axes


def plot_multiline_los(results, sightline, figsize=(11, 7)):
    """
    v0.2: plot local emissivity and cumulative intrinsic/attenuated intensity
    for several lines on the same sightline, for visual sanity-checking.
    """
    d = sightline.d
    lines = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    for name in lines:
        r = results[name]
        ax.plot(d, r.j_local, label=f"{name} ({r.wavelength_nm:.0f} nm)")
    ax.set_yscale("log")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel(r"local emissivity $j$ [erg s$^{-1}$ cm$^{-3}$]")
    ax.set_title("Line emissivities along LOS")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    x = np.arange(len(lines))
    intrinsic = [results[n].I_intrinsic for n in lines]
    attenuated = [results[n].I_attenuated for n in lines]
    width = 0.35
    ax.bar(x - width / 2, intrinsic, width, label="intrinsic", color="tab:blue")
    ax.bar(x + width / 2, attenuated, width, label="attenuated", color="tab:red")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(lines, rotation=30, ha="right")
    ax.set_ylabel(r"integrated intensity [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_title("Total intrinsic vs. attenuated flux per line")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig, axes


def plot_bpt_shift(results_list, labels=None, figsize=(6.5, 6)):
    """
    v0.2 preview of the BPT geometric-bias plot (full version in v0.4):
    plots log([NII]/Halpha) vs log([OIII]/Hbeta) for both intrinsic and
    dust-attenuated fluxes, for one or more sightlines/geometries, with an
    arrow connecting each pair to show how dust moves the point.
    """
    from . import los as _los
    fig, ax = plt.subplots(figsize=figsize)
    labels = labels or [f"sightline {i}" for i in range(len(results_list))]

    for i, (results, label) in enumerate(zip(results_list, labels)):
        x_int = np.log10(_los.line_ratio(results, "NII6584", "Halpha", attenuated=False))
        y_int = np.log10(_los.line_ratio(results, "OIII5007", "Hbeta", attenuated=False))
        x_att = np.log10(_los.line_ratio(results, "NII6584", "Halpha", attenuated=True))
        y_att = np.log10(_los.line_ratio(results, "OIII5007", "Hbeta", attenuated=True))

        ax.scatter(x_int, y_int, color="tab:blue", marker="o", s=60, zorder=5)
        ax.scatter(x_att, y_att, color="tab:red", marker="s", s=60, zorder=5)
        ax.annotate("", xy=(x_att, y_att), xytext=(x_int, y_int),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
        ax.annotate(f"{i+1}: {label}", xy=(x_att, y_att), fontsize=8,
                    xytext=(8, -10 - 14 * i), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))

    # Kewley (2001) extreme starburst line, for reference framing only
    x_line = np.linspace(-1.5, 0.3, 200)
    y_line = 0.61 / (x_line - 0.47) + 1.19
    ax.plot(x_line, y_line, "k--", lw=1, label="Kewley+01 max starburst")

    ax.scatter([], [], color="tab:blue", marker="o", label="intrinsic")
    ax.scatter([], [], color="tab:red", marker="s", label="dust-attenuated")

    all_x = [np.log10(_los.line_ratio(r, "NII6584", "Halpha", attenuated=a))
             for r in results_list for a in (True, False)]
    all_y = [np.log10(_los.line_ratio(r, "OIII5007", "Hbeta", attenuated=a))
             for r in results_list for a in (True, False)]
    pad_x = max(0.15, 0.3 * (max(all_x) - min(all_x)))
    pad_y = max(0.15, 0.3 * (max(all_y) - min(all_y)))
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ax.set_xlabel(r"log([NII]6584 / H$\alpha$)")
    ax.set_ylabel(r"log([OIII]5007 / H$\beta$)")
    ax.set_title("Dust-driven BPT displacement (preview)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig, ax
