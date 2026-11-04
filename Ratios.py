import os
import numpy as np
import pandas as pd

BASE_DIR = "/home/taylor33/scratch/"

SITES_CONFIG = {
    'CURIE': {'path': os.path.join(BASE_DIR, 'Shielding_secondary.csv.gz'), 'depth': 0.40, 'Phi_mu': 1.15e-1},
    'Soudan': {'path': os.path.join(BASE_DIR, 'escaped_secondary_soudan.csv'), 'depth': 1.95, 'Phi_mu': 2.00e-3},
    'Kamioka': {'path': os.path.join(BASE_DIR, 'escaped_secondary_kamland.csv'), 'depth': 2.05, 'Phi_mu': 1.50e-3},
    'Gran Sasso': {'path': os.path.join(BASE_DIR, 'escaped_secondary_lngs.csv'), 'depth': 3.10, 'Phi_mu': 2.50e-4},
    'SNOLAB': {'path': os.path.join(BASE_DIR, 'escaped_secondary_snolab.csv'), 'depth': 6.00, 'Phi_mu': 3.00e-6}
}

SYS_ERR_FRACTION = 0.085  

def get_multiplicity_correction(E_mu):
    return 1.0 / (1.0 - 0.314 * (E_mu**0.128) + 1.68e6 * (E_mu**-5.793))

print(f"{'Site':<15} | {'Depth':<6} | {'500k / Full Ratio':<18} | {'1M / Full Ratio':<18} | {'2M / Full Ratio':<18}")
print("-" * 85)

for site, cfg in SITES_CONFIG.items():
    path = cfg['path']
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)
    
    # Calculate initial energy mean & multiplicity correction
    E_mu_col = 'MuonInitialEnergy (GeV)' if 'MuonInitialEnergy (GeV)' in df.columns else 'MuonInitialEnergy'
    E_mu_mean = df[E_mu_col].mean()
    m_corr = get_multiplicity_correction(E_mu_mean)

    if 'secondaryName' in df.columns:
        df['secondaryName'] = df['secondaryName'].astype(str).str.strip()

    # Total simulated primary muons in full file
    N_mu_full = df['EventID'].max() + 1 if 'EventID' in df.columns else len(df)
    n_count_full = len(df[df['secondaryName'] == 'neutron']) if 'secondaryName' in df.columns else len(df)
    
    flux_full = (n_count_full * m_corr / N_mu_full) * cfg['Phi_mu']
    err_full = np.sqrt(((np.sqrt(n_count_full * m_corr) / N_mu_full) * cfg['Phi_mu'])**2 + (flux_full * SYS_ERR_FRACTION)**2)

    # Subset limits based on target EventIDs (500k, 1M, 2M)
    subsets = {'500k': 500_000, '1M': 1_000_000, '2M': 2_000_000}
    ratios = {}
    err_ratios = {}

    for label, target_N in subsets.items():
        if target_N >= N_mu_full:
            ratios[label], err_ratios[label] = 1.0, 0.0
            continue

        # Filter by EventID to get the true actual subset
        if 'EventID' in df.columns:
            df_sub = df[df['EventID'] < target_N]
            N_mu_sub = target_N
        else:
            frac = target_N / N_mu_full
            df_sub = df.iloc[:int(len(df) * frac)]
            N_mu_sub = target_N

        n_count_sub = len(df_sub[df_sub['secondaryName'] == 'neutron']) if 'secondaryName' in df.columns else len(df_sub)
        
        flux_sub = (n_count_sub * m_corr / N_mu_sub) * cfg['Phi_mu']
        stat_err_sub = (np.sqrt(n_count_sub * m_corr) / N_mu_sub) * cfg['Phi_mu'] if n_count_sub > 0 else 0
        err_sub = np.sqrt(stat_err_sub**2 + (flux_sub * SYS_ERR_FRACTION)**2)

        ratio = flux_sub / flux_full if flux_full > 0 else 0
        err_ratio = ratio * (err_sub / flux_sub) if flux_sub > 0 else 0

        ratios[label] = ratio
        err_ratios[label] = err_ratio

    print(
        f"{site:<15} | {cfg['depth']:<6.2f} | "
        f"{ratios['500k']:.4f} +/- {err_ratios['500k']:.4f} | "
        f"{ratios['1M']:.4f} +/- {err_ratios['1M']:.4f} | "
        f"{ratios['2M']:.4f} +/- {err_ratios['2M']:.4f}"
    )