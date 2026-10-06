import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths in /scratch/taylor33/
FTFP_PATH = "/scratch/taylor33/ftfp_secondary.csv"
SHIELDING_PATH = "/scratch/taylor33/Shielding_secondary.csv"

# -------------------------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------------------------
# Number of primary muons simulated in each run (adjust if different)
N_PRIMARIES_FTFP = 400e6
N_PRIMARIES_SHIELDING = 400e6

# Default primary muon energy in GeV (if not specified in CSV)
DEFAULT_E_MU_GEV = 280.0 

# Check if files exist, fallback to .csv.gz if uncompressed doesn't exist
if not os.path.exists(FTFP_PATH) and os.path.exists(FTFP_PATH + ".gz"):
    FTFP_PATH += ".gz"
if not os.path.exists(SHIELDING_PATH) and os.path.exists(SHIELDING_PATH + ".gz"):
    SHIELDING_PATH += ".gz"

print(f"Loading FTFP data from: {FTFP_PATH}")
df_ftfp = pd.read_csv(FTFP_PATH)

print(f"Loading Shielding data from: {SHIELDING_PATH}")
df_shielding = pd.read_csv(SHIELDING_PATH)

# Clean string columns and normalize case
for df in [df_ftfp, df_shielding]:
    df['secondaryName'] = df['secondaryName'].astype(str).str.strip().str.lower()
    df['creationProcess'] = df['creationProcess'].astype(str).str.strip()

# -------------------------------------------------------------------
# MEI & HIME BIAS CORRECTION FUNCTION (Eq. 5)
# -------------------------------------------------------------------
def mei_hime_correction_factor(E_mu_GeV):
    denominator = 1.0 - 0.314 * (E_mu_GeV ** 0.128) + 1.68e6 * (E_mu_GeV ** -5.793)
    return 1.0 / denominator

def get_weights(df_neutrons, n_primaries_simulated=400e6):
    """Computes event weights combining Mei & Hime correction and primary normalization."""
    mu_cols = ['PrimaryEnergy (GeV)', 'PrimaryEnergy', 'E_mu', 'E_primary', 'muon_energy']
    e_mu = None
    
    for col in mu_cols:
        if col in df_neutrons.columns:
            e_mu = df_neutrons[col]
            if e_mu.mean() > 1000:  # Convert MeV to GeV if needed
                e_mu = e_mu / 1000.0
            break
            
    # Always ensure e_mu is a pandas Series so weights support index operations
    if e_mu is None:
        e_mu = pd.Series(DEFAULT_E_MU_GEV, index=df_neutrons.index)

    mh_weights = mei_hime_correction_factor(e_mu)
    norm_scale = 400e6 / n_primaries_simulated
    return mh_weights * norm_scale

# -------------------------------------------------------------------
# PROCESS CATEGORIZATION
# -------------------------------------------------------------------
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

def get_relative_contributions(df, n_primaries):
    neutrons_df = df[df['secondaryName'] == 'neutron'].copy()
    neutrons_df['weight'] = get_weights(neutrons_df, n_primaries)
    neutrons_df['process_group'] = neutrons_df['creationProcess'].apply(categorize_process)
    
    weighted_counts = neutrons_df.groupby('process_group')['weight'].sum()
    total_weight = neutrons_df['weight'].sum() if len(neutrons_df) > 0 else 1.0
    return [weighted_counts.get(cat, 0.0) / total_weight for cat in process_categories]

# Compute relative contributions
contrib_ftfp = get_relative_contributions(df_ftfp, N_PRIMARIES_FTFP)
contrib_shielding = get_relative_contributions(df_shielding, N_PRIMARIES_SHIELDING)

os.makedirs("plots", exist_ok=True)

# -------------------------------------------------------------------
# PLOT 1: Process Relative Contribution Comparison
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
plt.savefig("plots/reweighted_normalized_relative_contribution_comparison.png", dpi=300)
plt.close()

# -------------------------------------------------------------------
# PLOT 2: Neutron Exit Energy Spectrum (MeV)
# -------------------------------------------------------------------
def extract_exit_energy_and_weights(df, n_primaries):
    neutrons_df = df[df['secondaryName'] == 'neutron'].copy()
    energy_cols = [
        'ExitEnergy (MeV)', 'KineticEnergy (MeV)', 'Energy (MeV)', 
        'e_kin', 'E_kin', 'ekin', 'energy', 'Energy'
    ]
    for col in energy_cols:
        if col in neutrons_df.columns:
            weights = get_weights(neutrons_df, n_primaries)
            valid_mask = neutrons_df[col].notna()
            return neutrons_df.loc[valid_mask, col], weights.loc[valid_mask]
            
    raise KeyError(f"Could not find energy column in CSV. Available columns: {list(neutrons_df.columns)}")

energy_ftfp, w_ftfp = extract_exit_energy_and_weights(df_ftfp, N_PRIMARIES_FTFP)
energy_shielding, w_shielding = extract_exit_energy_and_weights(df_shielding, N_PRIMARIES_SHIELDING)

bins_energy = np.logspace(-10, 5, 500)

plt.figure(figsize=(9, 5.5))

counts_ftfp, bin_edges = np.histogram(energy_ftfp, bins=bins_energy, weights=w_ftfp)
counts_shielding, _ = np.histogram(energy_shielding, bins=bins_energy, weights=w_shielding)
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

plt.step(bin_centers, counts_ftfp, where='mid', color='#1f77b4', linestyle='-', linewidth=1.2, label='FTFP_BERT_HP')
plt.step(bin_centers, counts_shielding, where='mid', color='#ff7f0e', linestyle='--', linewidth=1.2, label='Shielding_HP')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Reweighted, normalized Neutron Exit Energy (MeV)', fontsize=14)
plt.ylabel('Count (a.u.)', fontsize=14)
plt.xlim(1e-10, 1e5)
plt.ylim(0.6, 1.2e4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()

plt.savefig("plots/reweighted_normalized_neutron_exit_energy_comparison.png", dpi=300)
plt.close()

# -------------------------------------------------------------------
# PLOT 3: Neutron Production Distance From Lab (m)
# -------------------------------------------------------------------
def extract_distance_and_weights(df, n_primaries):
    neutrons_df = df[df['secondaryName'] == 'neutron'].copy()
    weights = get_weights(neutrons_df, n_primaries)
    
    if {'CreationX (m)', 'CreationY (m)', 'CreationZ (m)'}.issubset(neutrons_df.columns):
        dist = np.sqrt(neutrons_df['CreationX (m)']**2 + neutrons_df['CreationY (m)']**2 + neutrons_df['CreationZ (m)']**2)
    elif 'ProductionDistance (m)' in neutrons_df.columns:
        dist = neutrons_df['ProductionDistance (m)']
    else:
        x = neutrons_df.get('x', neutrons_df.get('CreationX', 0))
        y = neutrons_df.get('y', neutrons_df.get('CreationY', 0))
        z = neutrons_df.get('z', neutrons_df.get('CreationZ', 0))
        dist = np.sqrt(x**2 + y**2 + z**2)
        
    valid_mask = dist.notna()
    return dist.loc[valid_mask], weights.loc[valid_mask]

dist_ftfp, w_dist_ftfp = extract_distance_and_weights(df_ftfp, N_PRIMARIES_FTFP)
dist_shielding, w_dist_shielding = extract_distance_and_weights(df_shielding, N_PRIMARIES_SHIELDING)

bins_distance = np.linspace(0, 6.5, 200)

plt.figure(figsize=(9, 5.5))

counts_ftfp, bin_edges = np.histogram(dist_ftfp, bins=bins_distance, weights=w_dist_ftfp)
counts_shielding, _ = np.histogram(dist_shielding, bins=bins_distance, weights=w_dist_shielding)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.step(bin_centers, counts_ftfp, where='mid', color='#2ca02c', linestyle='-', linewidth=1.5, label='FTFP_BERT_HP')
plt.step(bin_centers, counts_shielding, where='mid', color='#ff7f0e', linestyle='--', linewidth=1.5, label='Shielding_HP')

plt.yscale('log')
plt.xlabel('Reweighted, normalized Neutron Production Distance From Lab (m)', fontsize=14)
plt.ylabel('Count (a.u.)', fontsize=14)
plt.xlim(-0.3, 6.8)
plt.ylim(0.6, 3e4)
plt.xticks(np.arange(0.0, 6.5, 0.5), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()

plt.savefig("plots/reweighted_normalized_neutron_production_distance_comparison.png", dpi=300)
plt.close()
