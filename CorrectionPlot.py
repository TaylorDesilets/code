import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHIELDING_PATH = "/home/taylor33/scratch/Shielding_secondary.csv.gz"
if not os.path.exists(SHIELDING_PATH) and os.path.exists("/home/taylor33/scratch/Shielding_secondary.csv"):
    SHIELDING_PATH = "/home/taylor33/scratch/Shielding_secondary.csv"

print(f"Loading Shielding dataset from: {SHIELDING_PATH}")
df = pd.read_csv(SHIELDING_PATH)
df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
neutrons_df = df[df['secondaryName'] == 'neutron'].copy()

# Update PRIMARIES_PER_RUN to match the total primary muons thrown in Geant4 per run macro
PRIMARIES_PER_RUN = 1e6  

#liquid scintillator material properties
RHO_TARGET = 0.86  # Target density in g/cm^3
MEAN_TRACK_LENGTH_CM = 100.0  # Path length / thickness in cm
AREAL_DENSITY = RHO_TARGET * MEAN_TRACK_LENGTH_CM  # g/cm^2

# hardcoded Experimental Data [E_mu (GeV), Yield (n / mu / g cm^-2), Error]
E_mu_data  = np.array([16.5, 90.0, 128.0, 260.0, 340.0])
data_yield = np.array([1.60e-5, 1.18e-4, 2.10e-4, 2.80e-4, 3.40e-4])
data_err   = np.array([0.70e-5, 0.30e-4, 0.20e-4, 0.35e-4, 1.70e-4])

# 5. Extract Unique Primary Energies and Calculate Yields
# Find energy column dynamically
energy_col = None
for col in ['MuonInitialEnergy (GeV)']:
    if col in df.columns:
        energy_col = col
        break

if energy_col is None:
    raise KeyError(f"Could not find primary energy column in DataFrame. Available: {list(df.columns)}")

# Convert to GeV if values are in MeV
if df[energy_col].mean() > 1000:
    neutrons_df['E_mu_GeV'] = neutrons_df[energy_col] / 1000.0
else:
    neutrons_df['E_mu_GeV'] = neutrons_df[energy_col]

neutrons_df['E_mu_round'] = neutrons_df['E_mu_GeV'].round(1)

sim_E_mu, sim_yield, sim_err = [], [], []

for energy, group in neutrons_df.groupby('E_mu_round'):
    n_count = len(group)
    
    # Yield = N_neutrons / (N_mu * ArealDensity)
    yield_val = n_count / (PRIMARIES_PER_RUN * AREAL_DENSITY)
    yield_error = np.sqrt(n_count) / (PRIMARIES_PER_RUN * AREAL_DENSITY)
    
    sim_E_mu.append(energy)
    sim_yield.append(yield_val)
    sim_err.append(yield_error)

sim_E_mu = np.array(sim_E_mu)
sim_yield = np.array(sim_yield)
sim_err = np.array(sim_err)

def mei_hime_correction_factor(E_mu_GeV):
    """
    Computes the energy-dependent correction factor C(E_mu) from Eq. 5.
    E_mu_GeV: Mean primary muon energy in GeV
    """
    denom = 1.0 - 0.314 * (E_mu_GeV ** 0.128) + 1.68e6 * (E_mu_GeV ** -5.793)
    return 1.0 / denom

#energy-dependent factor for each simulated point
corr_factors = mei_hime_correction_factor(sim_E_mu)

#correction to yield and include the 8.5% systematic uncertainty mentioned in the text
sim_corr_yield = sim_yield * corr_factors

# Combine statistical error with the conservative 8.5% systematic error in quadrature
sigma_corr = 0.085
sim_corr_err = sim_corr_yield * np.sqrt((sim_err / sim_yield)**2 + sigma_corr**2)

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 6), dpi=150)

# Data (Blue)
plt.errorbar(
    E_mu_data, data_yield, yerr=data_err,
    fmt='o', color='#1f77b4', ecolor='#1f77b4',
    ms=5, capsize=4, capthick=1, label='Data'
)

# Uncorrected Simulation (Green)
if len(sim_E_mu) > 0:
    plt.errorbar(
        sim_E_mu, sim_yield, yerr=sim_err,
        fmt='o', color='darkgreen', ecolor='darkgreen',
        ms=5, capsize=4, capthick=1, label='Simulation'
    )

    # Corrected Simulation (Orange/Brown)
    plt.errorbar(
        sim_E_mu, sim_corr_yield, yerr=sim_corr_err,
        fmt='^', color='#b84500', ecolor='#b84500',
        ms=6, capsize=4, capthick=1, label='Simulation Corrected'
    )

plt.yscale('log')
plt.xlabel(r'$E_{\mu}$ (GeV)', fontsize=16)
plt.ylabel(r'Neutron Yield $\left(\frac{\text{n}}{\mu \cdot \text{g}\cdot\text{cm}^{-2}}\right)$', fontsize=16)
plt.xlim(-5, 360)
plt.ylim(7e-6, 5e-4)
plt.xticks(np.arange(0, 400, 50), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=13, loc='upper left', frameon=True)
plt.tight_layout()

plt.savefig("plots/neutron_yield_vs_Emu_shielding.png", dpi=300)
plt.show()