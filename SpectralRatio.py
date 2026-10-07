import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_FULL = os.path.expanduser("~/scratch/Shielding_secondary.csv.gz")
PATH_MEAN = os.path.expanduser("~/scratch/Shielding_sim_MEAN.csv")

if not os.path.exists(PATH_FULL) and os.path.exists(PATH_FULL + ".gz"):
    PATH_FULL += ".gz"
if not os.path.exists(PATH_MEAN) and os.path.exists(PATH_MEAN + ".gz"):
    PATH_MEAN += ".gz"

N_MU_FULL = 1e6  
N_MU_MEAN = 1e6  

def extract_exit_energy(path):
    df = pd.read_csv(path)
    # Clean particle name string
    df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
    
    neutrons = df[df['secondaryName'] == 'neutron'].copy()
    
    energy_cols = [
        'ExitEnergy (MeV)', 'KineticEnergy (MeV)', 'Energy (MeV)', 
        'e_kin', 'E_kin', 'ekin', 'energy', 'Energy'
    ]
    for col in energy_cols:
        if col in neutrons.columns:
            energies = neutrons[col].dropna()
            return energies[energies > 0].values
            
    raise KeyError(f"Could not find energy column in {path}. Available columns: {list(df.columns)}")

# Extract secondary neutron kinetic energies from both runs
E_kin_full = extract_exit_energy(PATH_FULL)
E_kin_mean = extract_exit_energy(PATH_MEAN)

# Logarithmic binning across the energy range (10^-10 MeV to 10^5 MeV)
bins = np.logspace(-10, 5, 100)
bin_widths = np.diff(bins)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

counts_full, _ = np.histogram(E_kin_full, bins=bins)
counts_mean, _ = np.histogram(E_kin_mean, bins=bins)

dN_dE_full = counts_full / (bin_widths * N_MU_FULL)
dN_dE_mean = counts_mean / (bin_widths * N_MU_MEAN)

#Spectral Ratio: <E_mu> / E_FULL
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(dN_dE_full > 0, dN_dE_mean / dN_dE_full, np.nan)

print(f"Neutrons in FULL dataset: {len(E_kin_full)}")
print(f"Neutrons in MEAN dataset: {len(E_kin_mean)}")
print(f"Non-NaN ratio points: {np.count_nonzero(~np.isnan(ratio))}")

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 5.5), dpi=150)

plt.plot(bin_centers, ratio, color='#1f77b4', linewidth=1.5, label=r'$\langle E_{\mu} \rangle\ /\ E_{\text{FULL}}$')
plt.axhline(1.0, color='gray', linestyle='-', linewidth=1.0, label='Ratio = 1')

plt.xscale('log')
plt.xlabel('Energy (MeV)', fontsize=14)
plt.ylabel(r'$\langle E_{\mu} \rangle\ /\ E_{\text{FULL}}$', fontsize=16)
plt.xlim(1e-10, 1e5)
plt.ylim(-0.05, 2.7)
plt.xticks(fontsize=12)
plt.yticks(np.arange(0.0, 3.0, 0.5), fontsize=12)
plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=12, loc='upper right', frameon=True)
plt.tight_layout()

plt.savefig("plots/spectral_ratio_shielding.png", dpi=300)
plt.show()