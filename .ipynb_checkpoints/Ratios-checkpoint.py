import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/home/taylor33/scratch/"
PLOT_SAVE_PATH = "/home/taylor33/code/plots/flux_convergence_plot.png"

SITES_CONFIG = {
    'CURIE': {'path': os.path.join(BASE_DIR, 'Shielding_secondary.csv.gz'), 'depth': 0.40, 'Phi_mu': 1.15e-1, 'color': '#ff7f0e', 'marker': 'o'},
    'Soudan': {'path': os.path.join(BASE_DIR, 'escaped_secondary_soudan.csv'), 'depth': 1.95, 'Phi_mu': 2.00e-3, 'color': '#d62728', 'marker': 's'},
    'Kamioka': {'path': os.path.join(BASE_DIR, 'escaped_secondary_kamland.csv'), 'depth': 2.05, 'Phi_mu': 1.50e-3, 'color': '#bcbd22', 'marker': '^'},
    'Gran Sasso': {'path': os.path.join(BASE_DIR, 'escaped_secondary_lngs.csv'), 'depth': 3.10, 'Phi_mu': 2.50e-4, 'color': '#e377c2', 'marker': 'D'},
    'SNOLAB': {'path': os.path.join(BASE_DIR, 'escaped_secondary_snolab.csv'), 'depth': 6.00, 'Phi_mu': 3.00e-6, 'color': '#2ca02c', 'marker': '*'}
}

SYS_ERR_FRACTION = 0.085  

def get_multiplicity_correction(E_mu):
    return 1.0 / (1.0 - 0.314 * (E_mu**0.128) + 1.68e6 * (E_mu**-5.793))

# Data structures for plotting and console reporting
site_results = {}

print("=" * 110)
print(f"{'Site':<12} | {'Depth':<6} | {'Full Flux (m^-2 s^-1)':<24} | {'500k Flux':<20} | {'1M Flux':<20} | {'2M Flux':<20}")
print("=" * 110)

for site, cfg in SITES_CONFIG.items():
    path = cfg['path']
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        print(f"Warning: File not found for {site} at {path}. Skipping.")
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
    
    # Absolute flux and updated total uncertainty calculation
    flux_full = (n_count_full * m_corr / N_mu_full) * cfg['Phi_mu']
    stat_err_full = (np.sqrt(n_count_full * m_corr) / N_mu_full) * cfg['Phi_mu'] if n_count_full > 0 else 0
    err_full = np.sqrt(stat_err_full**2 + (flux_full * SYS_ERR_FRACTION)**2)

    subsets = {'500k': 500_000, '1M': 1_000_000, '2M': 2_000_000}
    fluxes = {}
    err_fluxes = {}
    ratios = {}
    err_ratios = {}

    for label, target_N in subsets.items():
        if target_N >= N_mu_full:
            fluxes[label] = flux_full
            err_fluxes[label] = err_full
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
        stat_err_sub = (np.sqrt(n_count_sub * m_corr) / N_mu_sub) * cfg['Phi_mu'] if n_count_sub > 0 else 0
        err_sub = np.sqrt(stat_err_sub**2 + (flux_sub * SYS_ERR_FRACTION)**2)

        ratio = flux_sub / flux_full if flux_full > 0 else 0
        err_ratio = ratio * (err_sub / flux_sub) if flux_sub > 0 else 0

        fluxes[label] = flux_sub
        err_fluxes[label] = err_sub
        ratios[label] = ratio
        err_ratios[label] = err_ratio

    site_results[site] = {
        'depth': cfg['depth'],
        'color': cfg['color'],
        'marker': cfg['marker'],
        'flux_full': flux_full,
        'err_full': err_full,
        'fluxes': fluxes,
        'err_fluxes': err_fluxes,
        'ratios': ratios,
        'err_ratios': err_ratios
    }

    print(
        f"{site:<12} | {cfg['depth']:<6.2f} | "
        f"{flux_full:.4e} +/- {err_full:.4e} | "
        f"{fluxes['500k']:.4e} | "
        f"{fluxes['1M']:.4e} | "
        f"{fluxes['2M']:.4e}"
    )

print("-" * 110)

# ==============================================================================
# PLOTTING CONVERGENCE TOWARDS 1
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

offsets = {'500k': -0.05, '1M': 0.00, '2M': 0.05}
alpha_vals = {'500k': 0.5, '1M': 0.75, '2M': 1.0}

for site, res in site_results.items():
    depth = res['depth']
    
    # Top Panel: Absolute Secondary Neutron Fluxes
    ax1.errorbar(depth, res['flux_full'], yerr=res['err_full'], fmt=res['marker'], 
                 color=res['color'], label=site, ms=8, capsize=4)
    
    # Bottom Panel: Subset / Full Flux Ratios
    for label in ['500k', '1M', '2M']:
        x_pos = depth + offsets[label]
        ax2.errorbar(x_pos, res['ratios'][label], yerr=res['err_ratios'][label], 
                     fmt=res['marker'], color=res['color'], alpha=alpha_vals[label],
                     capsize=3, ms=6)

# Formatting Top Subplot
ax1.set_yscale('log')
ax1.set_ylabel(r'Secondary Neutron Flux $\Phi_n$ ($\mathrm{m}^{-2}\mathrm{s}^{-1}$)', fontsize=11)
ax1.set_title('Secondary Neutron Flux and Convergence Across Subsets', fontsize=13)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(title='Sites', loc='upper right')

# Formatting Bottom Subplot (Ratio / Convergence)
ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Reference (Full Dataset = 1.0)')
ax2.set_xlabel('Overburden Depth (km.w.e.)', fontsize=11)
ax2.set_ylabel('Subset / Full Ratio', fontsize=11)
ax2.set_ylim(0.8, 1.2)
ax2.grid(True, linestyle='--', alpha=0.5)

# Custom legend markers for subset labels on the ratio plot
for label in ['500k', '1M', '2M']:
    ax2.plot([], [], marker='o', color='gray', alpha=alpha_vals[label], label=f'{label} Subset')
ax2.legend(loc='lower right', fontsize=9)

plt.tight_layout()

# Save plot if directory exists
os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
plt.savefig(PLOT_SAVE_PATH, dpi=300)
print(f"\nPlot successfully saved to: {PLOT_SAVE_PATH}")
plt.show()