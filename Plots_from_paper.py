import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths in /scratch/taylor33/
FTFP_PATH = "/scratch/taylor33/ftfp_secondary.csv"
SHIELDING_PATH = "/scratch/taylor33/Shielding_secondary.csv"

# Check if files exist, fallback to .csv.gz if uncompressed doesn't exist
if not os.path.exists(SHIELDING_PATH) and os.path.exists(SHIELDING_PATH + ".gz"):
    SHIELDING_PATH += ".gz"

print(f"Loading FTFP data from: {FTFP_PATH}")
df_ftfp = pd.read_csv(FTFP_PATH)

print(f"Loading Shielding data from: {SHIELDING_PATH}")
df_shielding = pd.read_csv(SHIELDING_PATH)

# Clean string columns
for df in [df_ftfp, df_shielding]:
    df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
    df['creationProcess'] = df['creationProcess'].astype(str).str.strip()

# Target process categories for Plot 1
process_categories = [
    'hadronCapture',
    'Kaon-process',
    'Pion-process',
    'MuMinusCapture',
    'MuonNuclear',
    'neutronInelastic',
    'photonNuclear',
    'protonInelastic',
    'Other'
]

def categorize_process(proc):
    if pd.isna(proc):
        return 'Other'
    
    proc_str = str(proc).strip()
    if proc_str in process_categories:
        return proc_str

    if proc_str in ['nCapture', 'hCaptureAtRest', 'hadronCapture'] or 'capture' in proc_str.lower():
        return 'hadronCapture'
    if any(k in proc_str.lower() for k in ['kaon', 'k0', 'k+', 'k-']):
        return 'Kaon-process'
    if any(p in proc_str.lower() for p in ['pion', 'pi+', 'pi-', 'pi0']):
        return 'Pion-process'
    if proc_str in ['muMinusCaptureAtRest', 'MuMinusCaptureAtRest']:
        return 'MuMinusCapture'
    if proc_str in ['muonNuclear', 'muonInelastic', 'muMinusInelastic', 'muPlusInelastic']:
        return 'MuonNuclear'
    if proc_str in ['neutronInelastic', 'nInelastic']:
        return 'neutronInelastic'
    if proc_str in ['photonNuclear', 'gammaNuclear', 'gInelastic']:
        return 'photonNuclear'
    if proc_str in ['protonInelastic', 'pInelastic']:
        return 'protonInelastic'

    return 'Other'

def get_relative_contributions(df):
    neutrons_df = df[df['secondaryName'] == 'neutron'].copy()
    neutrons_df['process_group'] = neutrons_df['creationProcess'].apply(categorize_process)
    counts = neutrons_df['process_group'].value_counts()
    total = len(neutrons_df) if len(neutrons_df) > 0 else 1
    return [counts.get(cat, 0) / total for cat in process_categories]

# Compute relative contributions for both datasets
contrib_ftfp = get_relative_contributions(df_ftfp)
contrib_shielding = get_relative_contributions(df_shielding)

# -------------------------------------------------------------------
# PLOT 1: Process Relative Contribution Comparison (Side-by-Side Bars)
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))

x = np.arange(len(process_categories))
width = 0.38

plt.bar(x - width/2, contrib_ftfp, width, label='FTFP_BERT_HP', color='#2ca02c', edgecolor='black', linewidth=0.8)
plt.bar(x + width/2, contrib_shielding, width, label='Shielding_HP', color='#ff7f0e', edgecolor='black', linewidth=0.8)

plt.ylabel('Relative Contribution', fontsize=14)
plt.xticks(x, process_categories, rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 0.72)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig("plots/relative_contribution_comparison.png", dpi=300)
plt.close()



# -------------------------------------------------------------------
# PLOT 2: Neutron Exit Energy Spectrum (MeV)
# -------------------------------------------------------------------
def extract_exit_energy(df):
    neutrons_df = df[df['secondaryName'] == 'neutron']
    
    # Check for common energy column names in Geant4 CSV output
    energy_cols = [
        'ExitEnergy (MeV)', 'KineticEnergy (MeV)', 'Energy (MeV)', 
        'e_kin', 'E_kin', 'ekin', 'energy', 'Energy'
    ]
    for col in energy_cols:
        if col in neutrons_df.columns:
            return neutrons_df[col].dropna()
            
    raise KeyError(
        f"Could not find an energy column in CSV. Available columns: {list(neutrons_df.columns)}"
    )

energy_ftfp = extract_exit_energy(df_ftfp)
energy_shielding = extract_exit_energy(df_shielding)

# Logarithmic binning from 1e-10 MeV to 1e5 MeV matching the target plot
bins_energy = np.logspace(-10, 5, 500)

plt.figure(figsize=(9, 5.5))

counts_ftfp, bin_edges = np.histogram(energy_ftfp, bins=bins_energy)
counts_shielding, _ = np.histogram(energy_shielding, bins=bins_energy)

# Geometric midpoints for logarithmic x-axis bin alignment
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

plt.step(
    bin_centers, counts_ftfp, where='mid', 
    color='#1f77b4', linestyle='-', linewidth=1.2, label='FTFP_BERT_HP'
)
plt.step(
    bin_centers, counts_shielding, where='mid', 
    color='#ff7f0e', linestyle='--', linewidth=1.2, label='Shielding_HP'
)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Neutron Exit Energy (MeV)', fontsize=14)
plt.ylabel('Count (a.u.)', fontsize=14)

plt.xlim(1e-10, 1e5)
plt.ylim(0.6, 1.2e4)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()

plt.savefig("plots/neutron_exit_energy_comparison.png", dpi=300)
plt.close()


# -------------------------------------------------------------------
# PLOT 3: Neutron Production Distance From Lab (m)
# -------------------------------------------------------------------
def extract_distance(df):
    neutrons_df = df[df['secondaryName'] == 'neutron']
    if {'CreationX (m)', 'CreationY (m)', 'CreationZ (m)'}.issubset(neutrons_df.columns):
        return np.sqrt(neutrons_df['CreationX (m)']**2 + neutrons_df['CreationY (m)']**2 + neutrons_df['CreationZ (m)']**2)
    elif 'ProductionDistance (m)' in neutrons_df.columns:
        return neutrons_df['ProductionDistance (m)'].dropna()
    else:
        x = neutrons_df.get('x', neutrons_df.get('CreationX', 0))
        y = neutrons_df.get('y', neutrons_df.get('CreationY', 0))
        z = neutrons_df.get('z', neutrons_df.get('CreationZ', 0))
        return np.sqrt(x**2 + y**2 + z**2)

dist_ftfp = extract_distance(df_ftfp)
dist_shielding = extract_distance(df_shielding)

bins_distance = np.linspace(0, 6.5, 200)

plt.figure(figsize=(9, 5.5))

counts_ftfp, bin_edges = np.histogram(dist_ftfp, bins=bins_distance)
counts_shielding, _ = np.histogram(dist_shielding, bins=bins_distance)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.step(bin_centers, counts_ftfp, where='mid', color='#2ca02c', linestyle='-', linewidth=1.5, label='FTFP_BERT_HP')
plt.step(bin_centers, counts_shielding, where='mid', color='#ff7f0e', linestyle='--', linewidth=1.5, label='Shielding_HP')

plt.yscale('log')
plt.xlabel('Neutron Production Distance From Lab (m)', fontsize=14)
plt.ylabel('Count (a.u.)', fontsize=14)
plt.xlim(-0.3, 6.8)
plt.ylim(0.6, 3e4)
plt.xticks(np.arange(0.0, 6.5, 0.5), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()

plt.savefig("plots/neutron_production_distance_comparison.png", dpi=300)
plt.close()
