import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/home/taylor33/scratch"
SAVE_DIR = "/home/taylor33/code/plots"
MUTE_FLUX_FILE = "/home/taylor33/code/MUTEFiles/MUTE_flat_overburden_fluxes.txt"

mute_data = np.loadtxt(MUTE_FLUX_FILE)

SITES = [
    {"label": "Kamioka", "csv": "0_KamLAND.csv", "depth": mute_data[0, 0], "phi_mu": mute_data[0, 2], "color": "#FFBF00"},
    {"label": "Boulby",  "csv": "0_Boulby.csv",  "depth": mute_data[1, 0], "phi_mu": mute_data[1, 2], "color": "#3F1FE0"},
    {"label": "LNGS",    "csv": "0_LNGS.csv",    "depth": mute_data[2, 0], "phi_mu": mute_data[2, 2], "color": "#D02F9D"},
    {"label": "SNOLAB",  "csv": "0_SNOLAB.csv",  "depth": mute_data[3, 0], "phi_mu": mute_data[3, 2], "color": "#3BC472"},
]

AREA_CAVERN = 66.2  # m^2
R_GEN = 8.4         # m

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


# 1. Extract muon-induced neutron fluxes directly from 0_*.csv
my_fluxes, my_errs = [], []
for s in SITES:
    flux, err = compute_neutron_flux(os.path.join(BASE_DIR, s["csv"]), s["phi_mu"])
    my_fluxes.append(flux)
    my_errs.append(err)

my_fluxes = np.array(my_fluxes)
my_errs = np.array(my_errs)

# 2. Keblbeck et al. Fig. 9 muon-induced neutron fluxes (m^-2 s^-1)
# (Digitized values from your previous plot corrected to m^-2 s^-1 neutron scale;
#  Boulby corresponds to the BUST/experimental point at ~2.84 km.w.e.)
paper_neutron = np.array([
    [2.0432, 1.3308e-4],   # Kamioka (Sim in Keblbeck et al.)
    [2.8453, 4.5000e-5],   # Boulby (BUST/exp reference in Keblbeck et al.)
    [3.0935, 3.0180e-5],   # LNGS (Sim in Keblbeck et al.)
    [6.0144, 6.5000e-7],   # SNOLAB (Sim in Keblbeck et al.)
])

ratios = my_fluxes / paper_neutron[:, 1]
ratio_errs = my_errs / paper_neutron[:, 1]

print(f"{'Lab':<10} | {'My Phi_n (m^-2 s^-1)':>22} | {'Paper Phi_n':>14} | {'Ratio +/- Err':>20}")
print("-" * 76)
for s, mf, me, pf, r, re in zip(SITES, my_fluxes, my_errs, paper_neutron[:, 1], ratios, ratio_errs):
    print(f"{s['label']:<10} | {mf:>12.4e} +/- {me:<6.1e} | {pf:>14.4e} | {r:>8.4f} +/- {re:<8.4f}")

# 3. Plot
labs = [s["label"] for s in SITES]
colors = [s["color"] for s in SITES]

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
bars1 = ax1.bar(
    labs, ratios, yerr=ratio_errs, color=colors, width=0.55,
    edgecolor="black", linewidth=0.8, zorder=3,
    error_kw=dict(ecolor="black", elinewidth=1.6, capsize=4, capthick=1.4, zorder=4)
)

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, zorder=2, label="Ratio = 1.0")
ax1.set_ylabel(r"Muon-Induced Neutron Flux Ratio ($\Phi_{n,\mathrm{Sim}} / \Phi_{n,\mathrm{Paper}}$)", fontsize=11)
ax1.set_title("Muon-Induced Neutron Flux Ratios by Underground Laboratory", fontsize=12.5, fontweight="bold", pad=12)
ax1.set_ylim(0, 1.8)
ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=1)
ax1.tick_params(axis="x", rotation=15)

for bar, val, err in zip(bars1, ratios, ratio_errs):
    ax1.text(
        bar.get_x() + bar.get_width() / 2, val + err + 0.04, f"{val:.3f}",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold"
    )

os.makedirs(SAVE_DIR, exist_ok=True)
out_path = os.path.join(SAVE_DIR, "neutron_flux_ratios.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
