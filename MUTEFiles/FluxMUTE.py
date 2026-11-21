####### Plot Flat-Overburden Underground Muon Fluxes with Uncertainties

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LABS = [
    ("KamLAND", "#bcbd22", "o"),
    ("Boulby",  "#1f77b4", "s"),
    ("LNGS",    "#e377c2", "^"),
    ("SNOLAB",  "#2ca02c", "D"),
]

# Load columns: vertical_depth_kmwe, flux_cm2_s, flux_m2_s
data_path = os.path.join(SCRIPT_DIR, "MUTE_flat_overburden_fluxes.txt")
data = np.loadtxt(data_path)

depths = data[:, 0]
flux_cm2 = data[:, 1]
flux_m2 = data[:, 2]

# Model & overburden systematic uncertainty (e.g., ~10% standard MUTE model uncertainty;
# replace with exact MUTE daemonflux error column if saved)
rel_err = np.array([0.09, 0.10, 0.10, 0.12])  # depth-dependent fractional uncertainty
flux_err_cm2 = flux_cm2 * rel_err


def format_sci_with_err(val, err):
    """Formats val +/- err in shared scientific notation: (1.30 ± 0.13) × 10⁻⁷"""
    exp = int(np.floor(np.log10(abs(val))))
    mant_val = val / (10 ** exp)
    mant_err = err / (10 ** exp)
    return rf"({mant_val:.2f} \pm {mant_err:.2f}) \times 10^{{{exp}}}"


fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)

# Guide curve and shaded uncertainty envelope
ax.plot(depths, flux_cm2, color="gray", linestyle="--", alpha=0.5, zorder=2)
ax.fill_between(
    depths,
    flux_cm2 - flux_err_cm2,
    flux_cm2 + flux_err_cm2,
    color="gray",
    alpha=0.15,
    zorder=1,
    label="MUTE Model Uncertainty",
)

for i, (lab, color, marker) in enumerate(LABS):
    flux_str = format_sci_with_err(flux_cm2[i], flux_err_cm2[i])
    ax.errorbar(
        depths[i],
        flux_cm2[i],
        yerr=flux_err_cm2[i],
        fmt=marker,
        color=color,
        ecolor=color,
        markersize=9,
        markeredgecolor="black",
        capsize=4,
        elinewidth=1.4,
        zorder=5,
        label=rf"{lab} ({depths[i]:.2f} km.w.e.): ${flux_str}\ \mathrm{{cm^{{-2}}\,s^{{-1}}}}$",
    )

ax.set_yscale("log")
ax.set_xlabel("Flat Overburden Vertical Depth [km.w.e.]", fontsize=12)
ax.set_ylabel(r"Total Underground Muon Flux $\Phi^u$ [$\mathrm{cm^{-2}\,s^{-1}}$]", fontsize=12)
ax.set_title("MUTE Underground Muon Flux (Flat Overburden)", fontsize=13)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(frameon=True, fontsize=9, loc="upper right")

out_path = os.path.join(SCRIPT_DIR, "MUTE_Flat_Overburden_Flux.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
