import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHIELDING_PATH = "/scratch/taylor33/Shielding_secondary.csv"
if not os.path.exists(SHIELDING_PATH) and os.path.exists(SHIELDING_PATH + ".gz"):
    SHIELDING_PATH += ".gz"
df_shielding = pd.read_csv(SHIELDING_PATH)

df_shielding['secondaryName'] = df_shielding['secondaryName'].astype(str).str.strip()
neutrons_shielding = df_shielding[df_shielding['secondaryName'] == 'neutron'].copy()

def extract_primary_energy(df):
    mu_cols = ['PrimaryEnergy (GeV)', 'PrimaryEnergy', 'E_mu', 'E_primary', 'muon_energy']
    for col in mu_cols:
        if col in df.columns:
            e_mu = df[col].copy()
            if e_mu.mean() > 1000:  # Convert MeV to GeV if needed
                e_mu = e_mu / 1000.0
            return e_mu
    # Default fallback if primary energy is constant or missing
    return pd.Series(280.0, index=df.index)

def extract_exit_energy(df):
    energy_cols = [
        'ExitEnergy (MeV)', 'KineticEnergy (MeV)', 'Energy (MeV)', 
        'e_kin', 'E_kin', 'ekin', 'energy', 'Energy'
    ]
    for col in energy_cols:
        if col in df.columns:
            return df[col].copy()
    raise KeyError(f"Could not find an energy column in CSV. Available columns: {list(df.columns)}")

def mei_hime_correction_factor(E_mu_GeV):
    denominator = 1.0 - 0.314 * (E_mu_GeV ** 0.128) + 1.68e6 * (E_mu_GeV ** -5.793)
    return 1.0 / denominator

energy = extract_exit_energy(neutrons_shielding)
e_mu = extract_primary_energy(neutrons_shielding)

valid = energy.notna() & e_mu.notna() & (energy > 0)
energy = energy[valid]
e_mu = e_mu[valid]

# Event-by-event weights for full muon spectrum
weights_full = mei_hime_correction_factor(e_mu)

# Fixed weight using the average muon energy <E_mu>
mean_e_mu = e_mu.mean()
weight_mean_e_mu = mei_hime_correction_factor(mean_e_mu)
weights_mean = np.full_like(weights_full, weight_mean_e_mu)

#Compute Energy Spectrum Histograms & Ratio
# 100 logarithmic bins across the energy range (1e-10 MeV to 1e5 MeV)
bins = np.logspace(-10, 5, 100)

counts_full, bin_edges = np.histogram(energy, bins=bins, weights=weights_full)
counts_mean, _ = np.histogram(energy, bins=bins, weights=weights_mean)
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

# Ratio: <E_mu> spectrum / E_FULL spectrum
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(counts_full > 0, counts_mean / counts_full, np.nan)

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 5.5), dpi=150)

plt.plot(bin_centers, ratio, color='#1f77b4', linewidth=1.5)
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