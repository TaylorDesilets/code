import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/home/taylor33/scratch/"
PLOT_SAVE_PATH = "/home/taylor33/code/plots/ratio_convergence_square.png"

SITES_CONFIG = {
    'CURIE': {'path': os.path.join(BASE_DIR, 'Shielding_secondary.csv.gz'), 'depth': 0.40, 'Phi_mu': 1.15e-1, 'color': '#ff7f0e', 'marker': 'o'},
    'Soudan': {'path': os.path.join(BASE_DIR, 'escaped_secondary_soudan.csv'), 'depth': 1.95, 'Phi_mu': 2.00e-3, 'color': '#d62728', 'marker': 's'},
    'Kamioka': {'path': os.path.join(BASE_DIR, 'escaped_secondary_kamland.csv'), 'depth': 2.05, 'Phi_mu': 1.50e-3, 'color': '#bcbd22', 'marker': '^'},
    'Gran Sasso': {'path': os.path.join(BASE_DIR, 'escaped_secondary_lngs.csv'), 'depth': 3.10, 'Phi_mu': 2.50e-4, 'color': '#e377c2', 'marker': 'D'},
    'SNOLAB': {'path': os.path.join(BASE_DIR, 'escaped_secondary_snolab.csv'), 'depth': 6.00, 'Phi_mu': 3.00e-6, 'color': '#2ca02c', 'marker': '*'}
}

def get_multiplicity_correction(E_mu):
    return 1.0 / (1.0 - 0.314 * (E_mu**0.128) + 1.68e6 * (E_mu**-5.793))

site_results = {}

for site, cfg in SITES_CONFIG.items():
    path = cfg['path']
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)
    
    E_mu_col = 'MuonInitialEnergy (GeV)' if 'MuonInitialEnergy (GeV)' in df.columns else 'MuonInitialEnergy'
    E_mu_mean = df[E_mu_col].mean()
    m_corr = get_multiplicity_correction(E_mu_mean)

    if 'secondaryName' in df.columns:
        df['secondaryName'] = df['secondaryName'].astype(str).str.strip()

    N_mu_full = df['EventID'].max() + 1 if 'EventID' in df.columns else len(df)
    n_count_full = len(df[df['secondaryName'] == 'neutron']) if 'secondaryName' in df.columns else len(df)
    
    flux_full = (n_count_full * m_corr / N_mu_full) * cfg['Phi_mu']

    subsets = {'500k': 500_000, '1M': 1_000_000, '2M': 2_000_000}
    ratios = {}
    err_ratios = {}

    for label, target_N in subsets.items():
        if target_N >= N_mu_full:
            ratios[label], err_ratios[label] = 1.0, 0.0
            continue

        if 'EventID' in df.columns:
            df_sub = df[df['EventID'] < target_N]
            N_mu_sub = target_N
        else:
            frac = target_N / N_mu_full
            df_sub = df.iloc[:int(len(df) * frac)]
            N_mu_sub = target_N

        n_count_sub = len(df_sub[df_sub['secondaryName'] == 'neutron']) if 'secondaryName' in df.columns else len(df_sub)
        flux_sub = (n_count_sub * m_corr / N_mu_sub) * cfg['Phi_mu']
        
        ratio = flux_sub / flux_full if flux_full > 0 else 0
        
        # PURE STATISTICAL UNCERTAINTY PROPAGATION ON RATIO:
        # Relative stat error on subset = 1 / sqrt(n_count_sub)
        rel_stat_err_sub = 1.0 / np.sqrt(n_count_sub) if n_count_sub > 0 else 0
        err_ratio = ratio * rel_stat_err_sub

        ratios[label] = ratio
        err_ratios[label] = err_ratio

    site_results[site] = {
        'depth': cfg['depth'],
        'color': cfg['color'],
        'marker': cfg['marker'],
        'ratios': ratios,
        'err_ratios': err_ratios
    }

# ==============================================================================
# SQUARE PLOT GENERATION
# ==============================================================================
plt.figure(figsize=(6, 6), dpi=150)  # Square aspect ratio (6x6 inches)

offsets = {'500k': -0.06, '1M': 0.00, '2M': 0.06}
alpha_vals = {'500k': 0.45, '1M': 0.70, '2M': 0.95}

# Draw reference unity line
plt.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='Reference (Full = 1.0)', zorder=1)

# Plot Site points
for site, res in site_results.items():
    depth = res['depth']
    for label in ['500k', '1M', '2M']:
        x_pos = depth + offsets[label]
        plt.errorbar(
            x_pos, res['ratios'][label], yerr=res['err_ratios'][label], 
            fmt=res['marker'], color=res['color'], alpha=alpha_vals[label],
            capsize=3, capthick=1, ms=6, zorder=3
        )

plt.xlabel('Overburden Depth (km.w.e.)', fontsize=11)
plt.ylabel('Subset / Full Flux Ratio', fontsize=11)
plt.title('Simulation Subset Convergence Towards 1.0', fontsize=12, pad=12)

plt.xlim(-0.2, 6.4)
plt.ylim(0.85, 1.15)
plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

# Build custom legend entries
site_handles = [
    plt.Line2D([0], [0], marker=cfg['marker'], color='w', markerfacecolor=cfg['color'], 
               markersize=8, label=site) 
    for site, cfg in SITES_CONFIG.items()
]
subset_handles = [
    plt.Line2D([0], [0], marker='o', color='gray', alpha=alpha_vals[lbl], 
               linestyle='None', label=f'{lbl} Subset') 
    for lbl in ['500k', '1M', '2M']
]

plt.legend(handles=site_handles + subset_handles, loc='lower right', fontsize=8, frameon=True, ncol=2)

plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
plt.savefig(PLOT_SAVE_PATH, dpi=300)
plt.show()
