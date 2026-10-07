import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION & INPUT DICTIONARY
# Configure your site simulation file paths, depth (km.w.e.), primary muon count (N_mu),
# and site muon flux (Phi_mu in m^-2 s^-1) for normalization.
# ==============================================================================
SITES_CONFIG = {
    'CURIE': {
        'path': '/scratch/taylor33/CURIE_secondary.csv',
        'depth': 0.40,             # km.w.e.
        'N_mu': 1e6,               # Primary muons thrown in simulation
        'Phi_mu': 1.15e-1,         # Primary muon flux at site (m^-2 s^-1)
        'color': '#ff7f0e',        # Orange
        'marker': '*',
        'ms': 10
    },
    'Soudan': {
        'path': '/scratch/taylor33/Soudan_secondary.csv',
        'depth': 1.95,
        'N_mu': 1e6,
        'Phi_mu': 2.00e-3,
        'color': '#d62728',        # Red
        'marker': '*',
        'ms': 10
    },
    'Kamioka': {
        'path': '/scratch/taylor33/Kamioka_secondary.csv',
        'depth': 2.05,
        'N_mu': 1e6,
        'Phi_mu': 1.50e-3,
        'color': '#bcbd22',        # Yellow-green
        'marker': '*',
        'ms': 10
    },
    'Boulby': {
        'path': '/scratch/taylor33/Boulby_secondary.csv',
        'depth': 2.85,
        'N_mu': 1e6,
        'Phi_mu': 4.00e-4,
        'color': '#17becf',        # Cyan
        'marker': 'o',
        'ms': 7
    },
    'Gran Sasso': {
        'path': '/scratch/taylor33/GranSasso_secondary.csv',
        'depth': 3.10,
        'N_mu': 1e6,
        'Phi_mu': 2.50e-4,
        'color': '#e377c2',        # Pink
        'marker': '*',
        'ms': 10
    },
    'SNOLAB': {
        'path': '/scratch/taylor33/SNOLAB_secondary.csv',
        'depth': 6.00,
        'N_mu': 1e6,
        'Phi_mu': 3.00e-6,
        'color': '#2ca02c',        # Green
        'marker': '*',
        'ms': 10
    }
}

# Systematic error percentage on simulated neutron flux (e.g., 8.5%)
SYS_ERR_FRACTION = 0.085  

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_neutron_count(file_path):
    """
    Reads CSV (or .csv.gz), filters for secondary neutrons, and returns total count.
    """
    if not os.path.exists(file_path) and os.path.exists(file_path + ".gz"):
        file_path += ".gz"
        
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}. Skipping site.")
        return None

    df = pd.read_csv(file_path)
    
    # Clean particle name column
    if 'secondaryName' in df.columns:
        df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
        neutrons = df[df['secondaryName'] == 'neutron']
    else:
        # Fallback if filtered in GEANT4 output directly
        neutrons = df

    return len(neutrons)

def mei_hime_flux(X):
    """
    Mei & Hime analytical model parametrization for Rock-Cavern Neutron Flux vs Depth X (km.w.e.).
    Phi_n(X) = A * (X_0 / X)^n * exp(-X / X_0)
    """
    A = 1.3e-3
    X0 = 0.85
    n = 1.1
    return A * ((X0 / X) ** n) * np.exp(-X / X0)

# ==============================================================================
# 3. PROCESS SIMULATION DATA
# ==============================================================================
sim_depths = []
sim_fluxes = []
sim_errors = []
site_labels = []
colors = []
markers = []
markersizes = []

print("Processing Simulation Files...")
for site, cfg in SITES_CONFIG.items():
    n_count = get_neutron_count(cfg['path'])
    
    if n_count is None:
        continue
        
    # Calculate Neutron Flux Phi_n = (N_neutrons / N_muons) * Phi_mu
    flux = (n_count / cfg['N_mu']) * cfg['Phi_mu']
    
    # Statistical error (Poisson sqrt(N))
    stat_err = (np.sqrt(n_count) / cfg['N_mu']) * cfg['Phi_mu'] if n_count > 0 else 0
    
    # Combined error (Statistical + Systematic in quadrature)
    total_err = np.sqrt(stat_err**2 + (flux * SYS_ERR_FRACTION)**2)
    
    sim_depths.append(cfg['depth'])
    sim_fluxes.append(flux)
    sim_errors.append(total_err)
    site_labels.append(f"{site} (This Work)")
    colors.append(cfg['color'])
    markers.append(cfg['marker'])
    markersizes.append(cfg['ms'])
    
    print(f" -> {site:12s} | Depth: {cfg['depth']} km.w.e. | Neutrons: {n_count:6d} | Flux: {flux:.3e} m^-2 s^-1")

# ==============================================================================
# 4. GENERATE PLOT
# ==============================================================================
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6.5), dpi=150)

# --- Analytical Model Curve & Uncertainty Band ---
X_mesh = np.linspace(0.05, 6.4, 300)
flux_mh = mei_hime_flux(X_mesh)

# Continuous line (Depth >= 0.8 km.w.e.)
mask_solid = X_mesh >= 0.8
plt.plot(X_mesh[mask_solid], flux_mh[mask_solid], 'k-', linewidth=1.5, label='M&H Model')

# Dashed line (Extrapolated shallow depths < 0.8 km.w.e.)
mask_dash = X_mesh <= 0.8
plt.plot(X_mesh[mask_dash], flux_mh[mask_dash], 'k--', linewidth=1.5, label='M&H Extrapolated')

# +/- 1 sigma uncertainty band
plt.fill_between(X_mesh, flux_mh * 0.5, flux_mh * 2.0, color='gray', alpha=0.18, label=r'M&H Model $\pm 1\sigma$')

# --- Plot Simulated Site Points ---
for depth, flux, err, label, col, mk, ms in zip(sim_depths, sim_fluxes, sim_errors, site_labels, colors, markers, markersizes):
    plt.errorbar(
        depth, flux, yerr=err,
        fmt=mk, color=col, ecolor=col,
        ms=ms, markeredgecolor='black', markeredgewidth=0.7,
        capsize=3, capthick=1, label=label
    )

# --- Axis Formatting & Labels ---
plt.yscale('log')
plt.xlabel('Depth (km.w.e.)', fontsize=14)
plt.ylabel(r'Rock-Cavern Flux ($\text{m}^{-2}\text{s}^{-1}$)', fontsize=14)
plt.xlim(-0.2, 6.5)
plt.ylim(5e-8, 1e1)

plt.xticks(np.arange(0, 7, 1), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.5)

# Two-column legend positioning
plt.legend(fontsize=10, loc='upper right', frameon=True, ncol=2)
plt.tight_layout()

# Save output
output_path = "plots/rock_cavern_neutron_flux_vs_depth.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nPlot successfully saved to: {output_path}")