####### Plot Flat-Overburden Underground Muon Fluxes (Matching Neutron Plot Style)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MUTE_FLUX_FILE = os.path.join(SCRIPT_DIR, "MUTE_flat_overburden_fluxes.txt")

# Lab display names, colors, and star markers matching the neutron flux plot
LABS = [
    ("Kamioka",    "#bcbd22", "*", 10),
    ("Boulby",     "#0000FF", "*", 10),
    ("Gran Sasso", "#e377c2", "*", 10),
    ("SNOLAB",     "#2ca02c", "*", 10),
]

# Load columns: [0] vertical_depth_kmwe, [1] flux_cm2_s, [2] flux_m2_s
data = np.loadtxt(MUTE_FLUX_FILE)
depths = data[:, 0]
flux_m2 = data[:, 2]

# Fractional MUTE model/overburden uncertainty (~8.5% - 12% across depth)
rel_err = np.array([0.09, 0.10, 0.10, 0.12])
flux_err_m2 = flux_m2 * rel_err


def format_flux_with_err(val, err):
    """Formats val +/- err in shared scientific notation for the plot legend."""
    if val == 0:
        return r"0.00 \pm 0.00"
    exp = int(np.floor(np.log10(abs(val))))
    mant_val = val / (10 ** exp)
    mant_err = err / (10 ** exp)
    return rf"({mant_val:.2f} \pm {mant_err:.2f}) \times 10^{{{exp}}}"


plt.figure(figsize=(9, 6.5), dpi=150)

# Connecting dashed guide curve and shaded uncertainty envelope
plt.plot(depths, flux_m2, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)
plt.fill_between(
    depths,
    flux_m2 - flux_err_m2,
    flux_m2 + flux_err_m2,
    color="gray",
    alpha=0.18,
    zorder=1,
    label=r"MUTE Model $\pm 1\sigma$",
)

for i, (lab, color, marker, ms) in enumerate(LABS):
    flux_label = format_flux_with_err(flux_m2[i], flux_err_m2[i])
    plt.errorbar(
        depths[i],
        flux_m2[i],
        yerr=flux_err_m2[i],
        fmt=marker,
        color=color,
        ecolor=color,
        ms=ms,
        markeredgecolor="black",
        markeredgewidth=0.7,
        capsize=3,
        zorder=5,
        label=rf"{lab} ({depths[i]:.2f} km.w.e.): ${flux_label}\ \mathrm{{m^{{-2}}\,s^{{-1}}}}$",
    )

plt.yscale("log")
plt.xlabel("Depth (km.w.e.)", fontsize=14)
plt.ylabel(r"Underground Muon Flux ($\text{m}^{-2}\text{s}^{-1}$)", fontsize=14)
plt.xlim(0, 6.5)
plt.ylim(1e-7, 1e-2)

plt.xticks(np.arange(0, 6.6, 0.5), fontsize=12)
plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
plt.yticks(fontsize=12)

plt.grid(True, which="both", color="gainsboro", linestyle="-", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=8.5, loc="upper right", frameon=True, ncol=1)
plt.tight_layout()

out_path = os.path.join(SCRIPT_DIR, "MUTE_Flat_Overburden_Flux.png")
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")
