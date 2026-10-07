import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 1. CONFIGURATION & INPUT DICTIONARY
# ==============================================================================
BASE_DIR = "/home/taylor33/scratch/"

SITES_CONFIG = {
    'CURIE': {
        'path': os.path.join(BASE_DIR, 'CURIE_Sim.csv'),
        'depth': 0.40,               # km.w.e.
        'N_mu': 1e6,                 # Primary muons
        'Phi_mu': 1.15e-1,           # Primary muon flux (m^-2 s^-1)
        'color': '#ff7f0e',          # Orange
        'marker': '*',
        'ms': 10
    },
    'Soudan': {
        'path': os.path.join(BASE_DIR, 'Soudan_Sim.csv'),
        'depth': 1.95,
        'N_mu': 1e6,
        'Phi_mu': 2.00e-3,
        'color': '#d62728',          # Red
        'marker': '*',
        'ms': 10
    },
    'Kamioka': {
        'path': os.path.join(BASE_DIR, 'Kamland_Sim.csv'), # KamLAND/Kamioka location
        'depth': 2.05,
        'N_mu': 1e6,
        'Phi_mu': 1.50e-3,
        'color': '#bcbd22',          # Yellow-green
        'marker': '*',
        'ms': 10
    },
    'Gran Sasso': {
        'path': os.path.join(BASE_DIR, 'LNGS_Sim.csv'),   # LNGS = Gran Sasso
        'depth': 3.10,
        'N_mu': 1e6,
        'Phi_mu': 2.50e-4,
        'color': '#e377c2',          # Pink
        'marker': '*',
        'ms': 10
    },
    'SNOLAB': {
        'path': os.path.join(BASE_DIR, 'SNOLAB_Sim.csv'),
        'depth': 6.00,
        'N_mu': 1e6,
        'Phi_mu': 3.00e-6,
        'color': '#2ca02c',          # Green
        'marker': '*',
        'ms': 10
    }
}

# Systematic error percentage on simulated neutron flux (e.g., 8.5%)
SYS_ERR_FRACTION = 0.085  

# 2. HELPER FUNCTIONS
# ==============================================================================
def get_neutron_count(file_name):
    path = os.path.join(SCRATCH_DIR, file_name)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}. Skipping.")
        return None
    df = pd.read_csv(path)
    if 'secondaryName' in df.columns:
        df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
        return len(df[df['secondaryName'] == 'neutron'])
    return len(df)

def mei_hime_flux(X):
    """Standard Mei & Hime (2006) parametrization"""
    A, X0, n = 1.3e-3, 0.85, 1.1
    return A * ((X0 / X) ** n) * np.exp(-X / X0)

def param_model(X, A, X0, n):
    """Equation 6 parameterization for fitting"""
    return A * ((X0 / X) ** n) * np.exp(-X / X0)

# 3. PROCESS SIMULATION DATA
# ==============================================================================
sim_data = {}
for site, cfg in SITES_CONFIG.items():
    n_count = get_neutron_count(cfg['file'])
    if n_count is None:
        continue
    flux = (n_count / cfg['N_mu']) * cfg['Phi_mu']
    stat_err = (np.sqrt(n_count) / cfg['N_mu']) * cfg['Phi_mu'] if n_count > 0 else 0
    total_err = np.sqrt(stat_err**2 + (flux * SYS_ERR_FRACTION)**2)
    
    sim_data[site] = {
        'depth': cfg['depth'],
        'flux': flux,
        'err': total_err,
        'color': cfg['color'],
        'marker': cfg['marker'],
        'ms': cfg['ms']
    }

# 4. PLOTTING FUNCTION
# ==============================================================================
def plot_simulation_flux(include_eqn6_fit=True, output_file="simulation_neutron_flux.png"):
    plt.figure(figsize=(9, 6.5), dpi=150)
    X_mesh = np.linspace(0.05, 6.4, 300)

    # Base M&H Model
    flux_mh = mei_hime_flux(X_mesh)
    plt.plot(X_mesh[X_mesh >= 0.8], flux_mh[X_mesh >= 0.8], 'k-', linewidth=1.5, label='M&H Model')
    plt.plot(X_mesh[X_mesh <= 0.8], flux_mh[X_mesh <= 0.8], 'k--', linewidth=1.5, label='M&H Extrapolated')
    plt.fill_between(X_mesh, flux_mh * 0.5, flux_mh * 2.0, color='gray', alpha=0.18, label=r'M&H Model $\pm 1\sigma$')

    # Fit to simulation data
    if include_eqn6_fit and len(sim_data) > 0:
        x_fit = [d['depth'] for d in sim_data.values()]
        y_fit = [d['flux'] for d in sim_data.values()]
        popt, _ = curve_fit(param_model, x_fit, y_fit, p0=[1.3e-3, 0.85, 1.1])
        flux_fit = param_model(X_mesh, *popt)
        
        plt.plot(X_mesh, flux_fit, 'r--', linewidth=1.2, label=r'Sim Fit: $\Phi_n = A \cdot (\frac{X_0}{X})^n \cdot e^{-X/X_0}$')
        plt.fill_between(X_mesh, flux_fit * 0.6, flux_fit * 1.6, color='red', alpha=0.12, label=r'Sim Fit $\pm 1\sigma$')

    # Plot simulation data points only
    for label, item in sim_data.items():
        plt.errorbar(item['depth'], item['flux'], yerr=item['err'], fmt=item['marker'], 
                     color=item['color'], ecolor=item['color'], ms=item['ms'], 
                     markeredgecolor='black', markeredgewidth=0.7, capsize=3, label=f"{label} (Sim)")

    plt.yscale('log')
    plt.xlabel('Depth (km.w.e.)', fontsize=14)
    plt.ylabel(r'Rock-Cavern Flux ($\text{m}^{-2}\text{s}^{-1}$)', fontsize=14)
    plt.xlim(-0.2, 6.5)
    plt.ylim(1e-7, 1e1)
    plt.xticks(np.arange(0, 7, 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=9, loc='upper right', frameon=True, ncol=2)
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", output_file)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

# Run the single plot
plot_simulation_flux(include_eqn6_fit=True, output_file="Simulation_Neutron_Flux.png")