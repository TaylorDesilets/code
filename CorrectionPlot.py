import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File path for Shielding dataset
SHIELDING_PATH = "/scratch/taylor33/Shielding_secondary.csv"
if not os.path.exists(SHIELDING_PATH) and os.path.exists(SHIELDING_PATH + ".gz"):
    SHIELDING_PATH += ".gz"

print(f"Loading Shielding data from: {SHIELDING_PATH}")
df_shielding = pd.read_csv(SHIELDING_PATH)
df_shielding['secondaryName'] = df_shielding['secondaryName'].astype(str).str.strip()

neutrons_shielding = df_shielding[df_shielding['secondaryName'] == 'neutron'].copy()

#Primary Energy Extraction & Binning
def extract_primary_energy(df):
    """Finds and extracts primary muon energy series in GeV."""
    mu_cols = [
        'MuonInitialEnergy (GeV)', 'MuonInitialEnergy',
        'PrimaryEnergy (GeV)', 'PrimaryEnergy', 
        'E_mu', 'E_primary', 'muon_energy'
    ]
    for col in mu_cols:
        if col in df.columns:
            e_mu = df[col].copy()
            if e_mu.mean() > 1000:  # Convert MeV to GeV if needed
                e_mu = e_mu / 1000.0
            return e_mu
    raise KeyError(f"Could not find primary energy column in DataFrame. Available columns: {list(df.columns)}")

neutrons_shielding['E_mu_GeV'] = extract_primary_energy(neutrons_shielding)

# Experimental Data Points: [E_mu (GeV), Yield (n / ug cm^-2), Error]
E_mu_data  = np.array([13.5, 90.0, 128.0, 260.0, 340.0])
data_yield = np.array([1.60e-5, 1.18e-4, 2.10e-4, 2.80e-4, 3.40e-4])
data_err   = np.array([0.70e-5, 0.30e-4, 0.20e-4, 0.35e-4, 1.70e-4])

#energy bins centered around the experimental E_mu points
bin_edges = [0.0, 50.0, 110.0, 180.0, 300.0, 400.0]
neutrons_shielding['E_bin'] = pd.cut(neutrons_shielding['E_mu_GeV'], bins=bin_edges)

N_PRIMARIES_PER_BIN = 1e6  

TARGET_YIELD_SCALE = 1e-4  

def mei_hime_correction_factor(E_mu_GeV):
    denominator = 1.0 - 0.314 * (E_mu_GeV ** 0.128) + 1.68e6 * (E_mu_GeV ** -5.793)
    return 1.0 / denominator

#Calculation of Yields from Shielding Data
sim_E_mu = []
sim_yield = []
sim_err = []

for i in range(len(bin_edges) - 1):
    bin_mask = neutrons_shielding['E_bin'] == pd.Interval(bin_edges[i], bin_edges[i+1])
    bin_data = neutrons_shielding[bin_mask]
    
    if len(bin_data) > 0:
        mean_E = bin_data['E_mu_GeV'].mean()
        count = len(bin_data)
        
        # Calculate yield and statistical uncertainty (sqrt(N))
        yield_val = (count / N_PRIMARIES_PER_BIN) * TARGET_YIELD_SCALE
        yield_error = (np.sqrt(count) / N_PRIMARIES_PER_BIN) * TARGET_YIELD_SCALE
    else:
        mean_E = E_mu_data[i]
        yield_val = 0.0
        yield_error = 0.0
        
    sim_E_mu.append(mean_E)
    sim_yield.append(yield_val)
    sim_err.append(yield_error)

sim_E_mu = np.array(sim_E_mu)
sim_yield = np.array(sim_yield)
sim_err = np.array(sim_err)

# Apply Mei & Hime Correction 
corr_factors = mei_hime_correction_factor(sim_E_mu)
sim_corr_yield = sim_yield * corr_factors
sim_corr_err   = sim_err * corr_factors

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 6), dpi=150)

# Data (Blue circles)
plt.errorbar(
    E_mu_data, data_yield, yerr=data_err,
    fmt='o', color='#1f77b4', ecolor='#1f77b4',
    mew=1, mec='#1f77b4', ms=5, capsize=4, capthick=1,
    label='Data'
)

# Uncorrected Shielding Simulation (Green circles)
plt.errorbar(
    sim_E_mu, sim_yield, yerr=sim_err,
    fmt='o', color='darkgreen', ecolor='darkgreen',
    mew=1, mec='darkgreen', ms=5, capsize=4, capthick=1,
    label='Simulation'
)

# Corrected Shielding Simulation (Orange/Brown triangles)
plt.errorbar(
    sim_E_mu, sim_corr_yield, yerr=sim_corr_err,
    fmt='^', color='#b84500', ecolor='#b84500',
    mew=1, mec='#b84500', ms=6, capsize=4, capthick=1,
    label='Simulation Corrected'
)

# Formatting
plt.yscale('log')
plt.xlabel(r'E_mu (GeV)', fontsize=16)
plt.ylabel(r'Neutron Yield', fontsize=16)
plt.xlim(-5, 360)
plt.ylim(7e-6, 5e-4)
plt.xticks(np.arange(0, 400, 50), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=13, loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("plots/neutron_yield_vs_Emu_shielding.png", dpi=300)
plt.show()