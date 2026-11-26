import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/home/taylor33/scratch"
SAVE_DIR = "/home/taylor33/code/plots"
MUTE_FLUX_FILE = "/home/taylor33/code/MUTEFiles/MUTE_flat_overburden_fluxes.txt"

# Load MUTE flat-overburden primary muon fluxes: [0] depth_kmwe, [1] flux_cm2_s, [2] flux_m2_s
mute_data = np.loadtxt(MUTE_FLUX_FILE)

SITES = [
    {"label": "Kamioka", "csv": "0_KamLAND.csv", "depth": mute_data[0, 0], "phi_mu": mute_data[0, 2], "color": "#FFBF00"},
    {"label": "Boulby",  "csv": "0_Boulby.csv",  "depth": mute_data[1, 0], "phi_mu": mute_data[1, 2], "color": "#3F1FE0"},
    {"label": "LNGS",    "csv": "0_LNGS.csv",    "depth": mute_data[2, 0], "phi_mu": mute_data[2, 2], "color": "#D02F9D"},
    {"label": "SNOLAB",  "csv": "0_SNOLAB.csv",  "depth": mute_data[3, 0], "phi_mu": mute_data[3, 2], "color": "#3BC472"},
]

AREA_CAVERN = 66.2  # Rock-cavern boundary surface area (m^2)
R_GEN = 8.4         # Primary generation hemisphere radius (m)

# Full Table III systematic uncertainty quadrature (~18.3%)
SYS_COMPONENTS = [0.085, 0.050, 0.100, 0.100, 0.060, 0.005]
SYS_ERR_FRACTION = np.sqrt(sum(v**2 for v in SYS_COMPONENTS))


def compute_neutron_flux(csv_path, phi_mu):
    """Computes muon-induced neutron flux (m^-2 s^-1) and total uncertainty from Geant4 CSV."""
    df = pd.read_csv(csv_path)
    e_mu_mean = (
        df["MuonInitialEnergy (GeV)"].mean()
        if "MuonInitialEnergy (GeV)" in df.columns
        else df["MuonInitialEnergy"].mean()
    )

    if "secondaryName" in df.columns:
        df["secondaryName"] = df["secondaryName"].astype(str).str.strip()
        n_count = len(df[df["secondaryName"] == "neutron"])
    else:
        n_count = len(df)

    n_mu = df["EventID"].max() + 1 if ("EventID" in df.columns and len(df) > 0) else len(df)
    m_sim = n_count / n_mu if n_mu > 0 else 0.0

    # Eq. 5 multiplicity bias correction
    f_corr = 1.0 / (1.0 - 0.314 * (e_mu_mean ** 0.128) + 1.68e6 * (e_mu_mean ** -5.793))
    n_corr = f_corr * m_sim * n_mu

    a_gen = 2.0 * np.pi * (R_GEN ** 2)
    t_live = n_mu / (phi_mu * a_gen)

    flux = n_corr / (t_live * AREA_CAVERN)
    stat_err = (f_corr * np.sqrt(n_count)) / (t_live * AREA_CAVERN) if n_count > 0 else 0.0
    total_err = np.sqrt(stat_err**2 + (flux * SYS_ERR_FRACTION)**2)
    return flux, total_err


# 1. Extract simulated muon-induced neutron fluxes and uncertainties from 0_*.csv
my_fluxes, my_errs = [], []
for s in SITES:
    flux, err = compute_neutron_flux(os.path.join(BASE_DIR, s["csv"]), s["phi_mu"])
    my_fluxes.append(flux)
    my_errs.append(err)

my_fluxes = np.array(my_fluxes)
my_errs = np.array(my_errs)

# 2. Reference paper neutron fluxes (m^-2 s^-1) from Keblbeck et al. Fig. 9
paper = np.array([
    [2.0423452768729633, 1.2826498305280598e-4],  # Kamioka
    [2.845819761129206,  4.5000000000000000e-5],  # Boulby* (experimental literature conversion)
    [3.095548317046687,  3.0180000000000000e-5],  # LNGS
    [6.00542888165038,   6.8000000000000000e-7],  # SNOLAB
])

# Site-specific relative uncertainties on the reference points:
# Kamioka, LNGS, SNOLAB = Table III simulation systematics (~18.3%)
# Boulby = converted experimental literature yield (SciBath/BUST, ~40%)
PAPER_REL_ERRS = np.array([
    SYS_ERR_FRACTION,  # Kamioka
    0.40,              # Boulby* (experimental conversion)
    SYS_ERR_FRACTION,  # LNGS
    SYS_ERR_FRACTION,  # SNOLAB
])

ratios = my_fluxes / paper[:, 1]
rel_err_mine = my_errs / my_fluxes
ratio_errs = ratios * np.sqrt(rel_err_mine**2 + PAPER_REL_ERRS**2)

print(f"{'Lab':<10} | {'My Phi_n (m^-2 s^-1)':>22} | {'Paper Val':>14} | {'Ratio +/- Err':>20}")
print("-" * 76)
for s, mf, me, pf, r, re in zip(SITES, my_fluxes, my_errs, paper[:, 1], ratios, ratio_errs):
    print(f"{s['label']:<10} | {mf:>12.4e} +/- {me:<6.1e} | {pf:>14.4e} | {r:>8.3f} +/- {re:<8.3f}")

# 3. Linear-scale Ratio Plot
labs = [s["label"] for s in SITES]
colors = [s["color"] for s in SITES]

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
bars1 = ax1.bar(
    labs, ratios, yerr=ratio_errs, color=colors, width=0.55,
    edgecolor="black", linewidth=0.8, zorder=3,
    error_kw=dict(ecolor="black", elinewidth=1.6, capsize=4, capthick=1.4, zorder=4)
)

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.3, zorder=2, label="Ratio = 1.0")
ax1.set_ylabel(r"Muon-Induced Neutron Flux Ratio ($\Phi_{n,\mathrm{Sim}} / \Phi_{n,\mathrm{Paper}}$)", fontsize=11)
ax1.set_title("Muon-Induced Neutron Flux Ratios by Underground Laboratory", fontsize=12.5, fontweight="bold", pad=12)

ax1.set_yscale("linear")
ax1.set_ylim(0, max(ratios + ratio_errs) * 1.15)
ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=1)
ax1.tick_params(axis="x", rotation=15)

for bar, val, err in zip(bars1, ratios, ratio_errs):
    ax1.text(
        bar.get_x() + bar.get_width() / 2, val + err + 0.03, f"{val:.3f}",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold"
    )

os.makedirs(SAVE_DIR, exist_ok=True)
out_path = os.path.join(SAVE_DIR, "neutron_flux_ratios.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
