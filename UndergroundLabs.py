import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

BASE_DIR = "/home/taylor33/scratch/"
save_path = "/home/taylor33/code/plots/"

SITES_CONFIG = {
    'CURIE': {
        'path': os.path.join(BASE_DIR, 'escaped_secondary_hideout.csv'),
        'depth': 0.40,               # km.w.e.
        'area': 66.2,                # area of rock cavern not including floor in m^2
        'R_gen': 8.4,                # Primary generation hemisphere radius in meters
        'Phi_mu': 1.15e-1,           # Primary muon flux (m^-2 s^-1)
        'color': '#ff7f0e',          # Orange
        'marker': '*',
        'ms': 10
    },

    'Soudan': {
        'path': os.path.join(BASE_DIR, 'escaped_secondary_soudan.csv'),
        'depth': 1.95,
        'area': 66.2,                # soudan 2 cavern where CDMS 2 is 3620
        'R_gen': 8.4,                
        'Phi_mu': 2.00e-3,
        'color': '#d62728',          # Red
        'marker': '*',
        'ms': 10
    },

    'Kamioka': {
        'path': os.path.join(BASE_DIR, 'escaped_secondary_kamland.csv'),
        'depth': 2.05,
        'area': 66.2,               # Where Hyper Kamiokande is (newly excavated) 2091
        'R_gen': 8.4,                
        'Phi_mu': 1.50e-3,
        'color': '#bcbd22',          # Yellow-green
        'marker': '*',
        'ms': 10
    },

    'Gran Sasso': {
        'path': os.path.join(BASE_DIR, 'escaped_secondary_lngs.csv'),
        'depth': 3.10,
        'area': 66.2,                # Hall C (Where NEWS is) 5376
        'R_gen': 8.4,                
        'Phi_mu': 2.50e-4,
        'color': '#e377c2',          # Pink
        'marker': '*',
        'ms': 10
    },

    'SNOLAB': {
        'path': os.path.join(BASE_DIR, 'escaped_secondary_snolab.csv'),
        'depth': 6.00,
        'area': 66.2,                 # cube hall 276
        'R_gen': 8.4,             
        'Phi_mu': 3.00e-6,
        'color': '#2ca02c',          # Green
        'marker': '*',
        'ms': 10
    }
}

SYS_ERR_FRACTION = 0.085  

def get_site_metrics(file_path):
    path = file_path
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}. Skipping.")
        return None, None, None

    df = pd.read_csv(path)

    # Primary muon energy
    if 'MuonInitialEnergy (GeV)' in df.columns:
        E_mu_mean = df['MuonInitialEnergy (GeV)'].mean()
    elif 'MuonInitialEnergy' in df.columns:
        E_mu_mean = df['MuonInitialEnergy'].mean()
    else:
        raise KeyError("Could not find 'MuonInitialEnergy' column in CSV.")

    # Secondary neutron count
    if 'secondaryName' in df.columns:
        df['secondaryName'] = df['secondaryName'].astype(str).str.strip()
        n_count = len(df[df['secondaryName'] == 'neutron'])
    else:
        n_count = len(df)

    # Calculate N_mu from EventID (0-indexed EventIDs mean total muons = max_id + 1)
    if 'EventID' in df.columns and len(df) > 0:
        n_mu_simulated = df['EventID'].max() + 1
    else:
        n_mu_simulated = len(df)

    return n_count, E_mu_mean, n_mu_simulated

def get_multiplicity_correction(E_mu):
    """Eq5: Geant4 empirical multiplicity bias correction."""
    return 1.0 / (1.0 - 0.314 * (E_mu**0.128) + 1.68e6 * (E_mu**-5.793))

def mei_hime_flux(X):
    """Mei & Hime (2006) parametrization"""
    A, X0, n = 2.0e-3, 0.85, 1.1  
    return A * ((X0 / X) ** n) * np.exp(-X / X0)

def param_model(X, A, X0, n):
    """Eq6: parameterization for fitting"""
    return A * ((X0 / X) ** n) * np.exp(-X / X0)

def sim_livetime(n_mu_simulated, phi_mu, r_gen):
    """simulation livetime in seconds from primary generation surface."""
    a_gen = 2.0 * np.pi * (r_gen ** 2)
    return n_mu_simulated / (phi_mu * a_gen)

print("=" * 72)
print(f"{'Site':<15} | {'Filename':<30} | {'Simulated Primary Muons (N_mu)':>22}")
print("-" * 72)

sim_data = {}
for site, cfg in SITES_CONFIG.items():
    n_count, E_mu_calculated, N_mu_simulated = get_site_metrics(cfg['path'])
    if n_count is None:
        continue

    fname = os.path.basename(cfg['path'])
    print(f"{site:<15} | {fname:<30} | {N_mu_simulated:>22,}")

    m_corr = get_multiplicity_correction(E_mu_calculated)
    corr_n_count = n_count * m_corr

    t_live = sim_livetime(N_mu_simulated, cfg['Phi_mu'], cfg['R_gen'])
    flux = corr_n_count / (t_live * cfg['area']) #  Phi_RC = N_s / (T_live * A_cavern)

    stat_err = np.sqrt(corr_n_count) / (t_live * cfg['area']) if corr_n_count > 0 else 0
    total_err = np.sqrt(stat_err**2 + (flux * SYS_ERR_FRACTION)**2)

    sim_data[site] = {
        'depth': cfg['depth'],
        'flux': flux,
        'err': total_err,
        'color': cfg['color'],
        'marker': cfg['marker'],
        'ms': cfg['ms']
    }

print("=" * 72)

def plot_simulation_flux(include_eqn6_fit=True, output_filename="Simulation_Neutron_Flux.png", target_dir=save_path):
    plt.figure(figsize=(9, 6.5), dpi=150)
    X_mesh = np.linspace(0.05, 6.4, 300)

    # Mei & Hime Model
    flux_mh = mei_hime_flux(X_mesh)
    plt.plot(X_mesh[X_mesh >= 0.8], flux_mh[X_mesh >= 0.8], 'k-', linewidth=1.5, label='M&H Model')
    plt.plot(X_mesh[X_mesh <= 0.8], flux_mh[X_mesh <= 0.8], 'k--', linewidth=1.5, label='M&H Extrapolated')
    plt.fill_between(X_mesh, flux_mh * 0.5, flux_mh * 2.0, color='gray', alpha=0.18, label=r'M&H Model $\pm 1\sigma$')

    if include_eqn6_fit and len(sim_data) >= 3:
        x_fit = np.array([d['depth'] for d in sim_data.values()])
        y_fit = np.array([d['flux'] for d in sim_data.values()])
        y_err = np.array([d['err'] for d in sim_data.values()])

        def log_param_model(X, A, X0, n):
            return np.log10(param_model(X, A, X0, n))
        
        y_fit_log = np.log10(y_fit)
        y_err_log = y_err / (y_fit * np.log(10))

        popt, _ = curve_fit(
            log_param_model, x_fit, y_fit_log,
            sigma=y_err_log,
            absolute_sigma=True,
            p0=[0.003, 1.015, 1.562],
            bounds=([1e-5, 0.1, 0.1], [0.1, 10.0, 5.0]),
            maxfev=50000
        )

        flux_fit = param_model(X_mesh, *popt)
        plt.plot(X_mesh, flux_fit, 'r--', linewidth=1.2, label=r'Sim Fit: $\Phi_n = A \cdot (\frac{X_0}{X})^n \cdot e^{-X/X_0}$')
        plt.fill_between(X_mesh, flux_fit * 0.6, flux_fit * 1.6, color='red', alpha=0.12, label=r'Sim Fit $\pm 1\sigma$')

    for label, item in sim_data.items():
        plt.errorbar(
            item['depth'], item['flux'], yerr=item['err'], fmt=item['marker'],
            color=item['color'], ecolor=item['color'], ms=item['ms'],
            markeredgecolor='black', markeredgewidth=0.7, capsize=3, label=f"{label} (Sim)"
        )

    plt.yscale('log')
    plt.xlabel('Depth (km.w.e.)', fontsize=14)
    plt.ylabel(r'Rock-Cavern Flux ($\text{m}^{-2}\text{s}^{-1}$)', fontsize=14)
    plt.xlim(0, 6.5)
    plt.ylim(1e-8, 1e1)

    plt.xticks(np.arange(0, 6.6, 0.5), fontsize=12)
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    plt.yticks(fontsize=12)
    
    plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=9, loc='upper right', frameon=True, ncol=2)
    plt.tight_layout()

    os.makedirs(target_dir, exist_ok=True)
    full_output_path = os.path.join(target_dir, output_filename)
    plt.savefig(full_output_path, dpi=300)
    print(f"Plot saved to {full_output_path}")
    plt.show()

plot_simulation_flux(include_eqn6_fit=True, output_filename="Simulation_Neutron_Flux.png")