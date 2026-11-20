

####### Flux for MUTE data before GEANT4

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. LAB CONFIGURATION (Flat Overburden Vertical Depths in km.w.e.)
# ==============================================================================
# Note: SNOLAB depth (5.89 - 6.00 km.w.e.), KamLAND (~2.05 km.w.e.),
# LNGS (~3.10 - 3.40 km.w.e.), Boulby (~2.80 - 2.84 km.w.e.)
LABS_CONFIG = {
    "KamLAND": {
        "file": "MUTE_KamLAND_underground_intensities_dd.txt",
        "depth_kmwe": 2.05,
        "color": "#bcbd22",
        "marker": "o",
    },
    "Boulby": {
        "file": "MUTE_Boulby_underground_intensities_dd.txt",
        "depth_kmwe": 2.84,
        "color": "#1f77b4",
        "marker": "s",
    },
    "LNGS": {
        "file": "MUTE_LNGS_underground_intensities_dd.txt",
        "depth_kmwe": 3.10,
        "color": "#e377c2",
        "marker": "^",
    },
    "SNOLAB": {
        "file": "MUTE_SNOLAB_underground_intensities_dd.txt",
        "depth_kmwe": 5.89,
        "color": "#2ca02c",
        "marker": "D",
    },
}


def get_mute_grids(n_depths, n_angles):
    """
    Retrieves MUTE's default slant depth and zenith angle grids if MUTE is
    installed; otherwise falls back to MUTE's standard default grid spacing.
    """
    try:
        import mute.constants as mtc
        depths = np.array(mtc.slant_depths, dtype=float)
        angles_deg = np.array(mtc.angles, dtype=float)
        if len(depths) == n_depths and len(angles_deg) == n_angles:
            return depths, angles_deg
    except ImportError:
        pass

    # Standard MUTE v1/v3 default grids:
    # Slant depths: 0.5 to 14.0 km.w.e. in steps of 0.5 (28 points)
    # Zenith angles: 0 to 85 deg in steps of 5 deg (18 points) or 0 to 90 (19/91 points)
    depths = np.linspace(0.5, 0.5 * n_depths, n_depths)
    angles_deg = np.linspace(0.0, 85.0 if n_angles == 18 else 90.0, n_angles)
    return depths, angles_deg


def calc_flat_overburden_flux(intensity_matrix, vertical_depth_kmwe):
    """
    Calculates total integrated underground muon flux Phi_u [cm^-2 s^-1]
    for a flat overburden of vertical depth h [km.w.e.].
    """
    n_rows, n_cols = intensity_matrix.shape

    # Case A: Standard flat-overburden intensity table I(X, theta) -> shape (N_depths, N_angles)
    if n_rows != n_cols and n_rows >= 20:
        depths, angles_deg = get_mute_grids(n_rows, n_cols)
        angles_rad = np.deg2rad(angles_deg)

        # Build 2D interpolator over (slant_depth, zenith_angle_deg)
        interp = RegularGridInterpolator(
            (depths, angles_deg),
            intensity_matrix,
            bounds_error=False,
            fill_value=0.0,
        )

        # Fine zenith grid up to max angle in table (avoid cos(90 deg) = 0)
        theta_max = min(angles_deg[-1], 89.0)
        theta_deg_fine = np.linspace(0.0, theta_max, 500)
        theta_rad_fine = np.deg2rad(theta_deg_fine)

        # Slant depth for flat overburden: X(theta) = h / cos(theta)
        slant_depths = vertical_depth_kmwe / np.cos(theta_rad_fine)

        # Mask angles whose slant depth exceeds the maximum tabulated depth
        valid = slant_depths <= depths[-1]
        pts = np.column_stack([slant_depths[valid], theta_deg_fine[valid]])
        I_theta = interp(pts)

        # Integrate Phi_u = 2 * pi * integral( I(h/cos(theta), theta) * sin(theta) d_theta )
        integrand = 2.0 * np.pi * I_theta * np.sin(theta_rad_fine[valid])
        flux_cm2 = np.trapezoid(integrand, theta_rad_fine[valid]) if hasattr(np, "trapezoid") else np.trapz(integrand, theta_rad_fine[valid])
        return flux_cm2

    # Case B: Angular intensity map I(theta, phi) -> shape (N_angles, N_azimuths)
    else:
        theta_deg = np.linspace(0.0, 90.0, n_rows)
        phi_deg = np.linspace(0.0, 360.0, n_cols)
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)

        # Integrate over azimuth phi first, then zenith theta with sin(theta)
        trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        I_theta = trapz_fn(intensity_matrix, phi_rad, axis=1)
        flux_cm2 = trapz_fn(I_theta * np.sin(theta_rad), theta_rad)
        return flux_cm2


# ==============================================================================
# 2. COMPUTE FLUXES FOR ALL 4 LABS
# ==============================================================================
results = {}
print(f"{'Lab':<10} | {'Depth (km.w.e.)':<15} | {'Shape':<10} | {'Flux [cm^-2 s^-1]':<18} | {'Flux [m^-2 s^-1]':<18}")
print("-" * 80)

for lab, cfg in LABS_CONFIG.items():
    if not os.path.exists(cfg["file"]):
        print(f"Warning: {cfg['file']} not found. Skipping {lab}.")
        continue

    data = np.loadtxt(cfg["file"])
    flux_cm2 = calc_flat_overburden_flux(data, cfg["depth_kmwe"])
    flux_m2 = flux_cm2 * 1e4  # Convert cm^-2 s^-1 -> m^-2 s^-1

    results[lab] = {
        "depth": cfg["depth_kmwe"],
        "flux_cm2": flux_cm2,
        "flux_m2": flux_m2,
        "color": cfg["color"],
        "marker": cfg["marker"],
    }

    print(f"{lab:<10} | {cfg['depth_kmwe']:<15.2f} | {str(data.shape):<10} | {flux_cm2:<18.4e} | {flux_m2:<18.4e}")


# ==============================================================================
# 3. PLOT TOTAL UNDERGROUND MUON FLUX VS. DEPTH
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

for lab, res in results.items():
    ax.scatter(
        res["depth"],
        res["flux_cm2"],
        color=res["color"],
        marker=res["marker"],
        s=90,
        edgecolor="black",
        zorder=5,
        label=f"{lab} ({res['depth']:.2f} km.w.e.): {res['flux_cm2']:.2e} $\\mathrm{{cm^{{-2}}\\,s^{{-1}}}}$",
    )

ax.set_yscale("log")
ax.set_xlabel("Flat Overburden Vertical Depth [km.w.e.]", fontsize=12)
ax.set_ylabel(r"Total Underground Muon Flux $\Phi^u$ [$\mathrm{cm^{-2}\,s^{-1}}$]", fontsize=12)
ax.set_title("MUTE Underground Muon Flux (Flat Overburden)", fontsize=13)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(frameon=True, fontsize=9.5)

plt.tight_layout()
plt.savefig("MUTE_Flat_Overburden_Flux.png")
print("\nSaved plot to MUTE_Flat_Overburden_Flux.png")

