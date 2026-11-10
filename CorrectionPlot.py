import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = "/home/taylor33/scratch/"
SYS_ERR_FRACTION = 0.085  

SITES_CONFIG = {
    'CURIE': {
        'path': os.path.join(BASE_DIR, 'Shielding_secondary.csv.gz'),
        'depth': 0.40,
        'N_mu': 1e6
    }
}
TARGET_RHO_L = 100.0  

def get_multiplicity_correction(E_mu):
    """Eq5: Geant4 empirical multiplicity bias correction."""
    return 1.0 / (1.0 - 0.314 * (E_mu ** 0.128) + 1.68e6 * (E_mu ** -5.793))

def get_site_metrics(file_path):
    """Extracts mean muon energy E_mu and neutron count from CSV."""
    path = file_path
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None, None
    
    df = pd.read_csv(path)
    
    if 'MuonInitialEnergy (GeV)' in df.columns:
        E_mu_mean = df['MuonInitialEnergy (GeV)'].mean()
    elif 'MuonInitialEnergy' in df.columns:
        E_mu_mean = df['MuonInitialEnergy'].mean()
    else:
        raise KeyError("Could not find 'MuonInitialEnergy' column in CSV.")
    
    if 'secondaryName' in df.columns:
        df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
        n_count = len(df[df['secondaryName'] == 'neutron'])
    else:
        n_count = len(df)
        
    return n_count, E_mu_mean


site_cfg = SITES_CONFIG['CURIE']
n_count, E_mu = get_site_metrics(site_cfg['path'])

if n_count is not None and E_mu is not None:
    N_mu = site_cfg['N_mu']
    
    y_sim = n_count / (N_mu * TARGET_RHO_L)
    stat_err_sim = (np.sqrt(n_count) / (N_mu * TARGET_RHO_L)) if n_count > 0 else 0
    
    m_corr = get_multiplicity_correction(E_mu)
    y_corr = y_sim * m_corr
    stat_err_corr = stat_err_sim * m_corr
    tot_err_corr = np.sqrt(stat_err_corr**2 + (y_corr * SYS_ERR_FRACTION)**2)
    print(f"  Mean Muon Energy (E_mu): {E_mu:.2f} GeV")
    print(f"  Neutron Count:           {n_count}")
    print(f"  Uncorrected Yield:      {y_sim:.3e}")
    print(f"  Multiplicity Factor:    {m_corr:.3f}")
    print(f"  Corrected Yield:        {y_corr:.3e}")
    plt.figure(figsize=(7, 6), dpi=150)

    plt.errorbar(
        [E_mu], [y_sim], yerr=[stat_err_sim],
        fmt='o', color='darkgreen', ecolor='darkgreen',
        capsize=4, elinewidth=1.2, label='Simulation', zorder=3
    )

    plt.errorbar(
        [E_mu], [y_corr], yerr=[tot_err_corr],
        fmt='^', color='#a04000', ecolor='#d35400',
        capsize=4, elinewidth=1.2, label='Simulation Corrected', zorder=4
    )

    plt.yscale('log')
    plt.xlim(0, max(150.0, E_mu * 1.5)) 
    plt.ylim(1e-6, 1e-3)

    plt.xlabel(r'$E_\mu$ (GeV)', fontsize=14)
    plt.ylabel(r'Neutron Yield $\left(\frac{n}{\mu \, \text{g cm}^{-2}}\right)$', fontsize=14)

    plt.grid(True, which='both', linestyle=':', color='gainsboro', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)

    plt.legend(fontsize=12, loc='upper left', frameon=True, edgecolor='gainsboro')
    plt.tight_layout()
    plt.savefig('simulation_yield.png', dpi=300)
    plt.show()