import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

BASE_DIR = "/home/taylor33/scratch"
save_path = "/home/taylor33/code/plots/"
MUTE_FLUX_FILE = "/home/taylor33/code/MUTEFiles/MUTE_flat_overburden_fluxes.txt"

# Load columns from MUTE output: [0] vertical_depth_kmwe, [1] flux_cm2_s, [2] flux_m2_s
# Row order: 0=KamLAND, 1=Boulby, 2=LNGS, 3=SNOLAB
mute_data = np.loadtxt(MUTE_FLUX_FILE)

SITES_CONFIG = {
    'Kamioka': {
        'path': os.path.join(BASE_DIR, '0_KamLAND.csv'),
        'depth': mute_data[0, 0],
        'area': 66.2,               # Rock-cavern boundary surface area (m^2)
        'R_gen': 8.4,               # Primary generation hemisphere radius (m)
        'Phi_mu': mute_data[0, 2],
        'color': '#bcbd22',         # Yellow-green
        'marker': '*',
        'ms': 10
    },

    'Boulby': {
        'path': os.path.join(BASE_DIR, '0_Boulby.csv'),
        'depth': mute_data[1, 0],
        'area': 66.2,
        'R_gen': 8.4,
        'Phi_mu': mute_data[1, 2],
        'color': '#0000FF',         # Blue
        'marker': '*',
        'ms': 10
    },

    'Gran Sasso': {
        'path': os.path.join(BASE_DIR, '0_LNGS.csv'),
        'depth': mute_data[2, 0],
        'area': 66.2,
        'R_gen': 8.4,
        'Phi_mu': mute_data[2, 2],
        'color': '#e377c2',         # Pink
        'marker': '*',
        'ms': 10
    },

    'SNOLAB': {
        'path': os.path.join(BASE_DIR, '0_SNOLAB.csv'),
        'depth': mute_data[3, 0],
        'area': 66.2,
        'R_gen': 8.4,
        'Phi_mu': mute_data[3, 2],
        'color': '#2ca02c',         # Green
        'marker': '*',
        'ms': 10
    }
}

# Full systematic error budget from Keblbeck et al. Table III (combined in quadrature)
SYS_COMPONENTS = {
    'sigma_corr': 0.085,       # Multiplicity bias correction (+/- 8.5%)
    'sigma_rho_E': 0.050,      # Rock density + muon energy spectrum (+/- 5.0%)
    'sigma_SH': 0.100,         # Sternheimer / standard-rock MUTE intensity (+/- 10.0%)
    'sigma_Tlive': 0.100,      # Livetime normalization (+/- 10.0%)
    'sigma_A': 0.250,          # Cavern wall surface area (+/- 25%)
    'sigma_geant4': 0.005,     # Geant4 physics model (+/- 0.5%)
}
SYS_ERR_FRACTION = np.sqrt(sum(v**2 for v in SYS_COMPONENTS.values()))  # sum of squared error


def get_site_metrics(file_path):
    path = file_path
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}. Skipping.")
        return None, None, None, None

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

    # Simulated neutron multiplicity
    M_sim = n_count / n_mu_simulated if n_mu_simulated > 0 else 0.0

    return n_count, E_mu_mean, n_mu_simulated, M_sim


def get_multiplicity_correction_factor(E_mu):
    """
    Eq5: Geant4 empirical multiplicity bias correction factor f_corr(E_mu)
    such that M_corr = f_corr * M_sim.
    """
    return 1.0 / (1.0 - 0.314 * (E_mu**0.128) + 1.68e6 * (E_mu**-5.793))


def mei_hime_flux(X):
    """
    Mei & Hime (2006) Eq. (13) Parametrization for total neutron flux in standard rock.
    P0 = 4.0e-7 cm^-2 s^-1 = 4.0e-3 m^-2 s^-1
    P1 = 0.86 km.w.e.
    """
    P0 = 4.0e-3  # m^-2 s^-1
    P1 = 0.86    # km.w.e.
    return P0 * (P1 / X) * np.exp(-X / P1)


def param_model(X, A, X0, n):
    """
    Eq6: parameterization for fitting
    A = 0.003 +- 0.001 m^-2 s^-1
    X0 = 1.015 +- 0.083 km.w.e
    n = 1.562 +- 0.192
    X = Depth (km.w.e.)
    """
    return A * ((X0 / X) ** n) * np.exp(-X / X0)


def sim_livetime(n_mu_simulated, phi_mu, r_gen):
    """Simulation livetime in seconds from primary generation hemisphere surface."""
    a_gen = 2.0 * np.pi * (r_gen ** 2)
    return n_mu_simulated / (phi_mu * a_gen)


def format_flux_with_err(val, err):
    """Formats val +/- err in shared scientific notation for the plot legend."""
    if val == 0:
        return r"0.00 \pm 0.00"
    exp = int(np.floor(np.log10(abs(val))))
    mant_val = val / (10 ** exp)
    mant_err = err / (10 ** exp)
    return rf"({mant_val:.2f} \pm {mant_err:.2f}) \times 10^{{{exp}}}"


print("=" * 112)
print(f"Total Fractional Systematic Uncertainty (Table III Quadrature): {SYS_ERR_FRACTION:.2%}")
print("-" * 112)
print(f"{'Site':<15} | {'Depth (km.w.e.)':<15} | {'N_neutrons':>10} | {'Stat Err':>10} | {'Flux +/- Total Err (m^-2 s^-1)':>32}")
print("-" * 112)

sim_data = {}
for site, cfg in SITES_CONFIG.items():
    n_count, E_mu_calculated, N_mu_simulated, M_sim = get_site_metrics(cfg['path'])
    if n_count is None:
        continue

    f_corr = get_multiplicity_correction_factor(E_mu_calculated)
    M_corr = f_corr * M_sim
    N_s = M_corr * N_mu_simulated  # Total corrected neutron count

    t_live = sim_livetime(N_mu_simulated, cfg['Phi_mu'], cfg['R_gen'])
    flux = N_s / (t_live * cfg['area'])  # Phi_RC = N_s / (T_live * A_cavern)

    # Propagate Poisson counting uncertainty through multiplicity correction + full systematic budget
    stat_err = (f_corr * np.sqrt(n_count)) / (t_live * cfg['area']) if n_count > 0 else 0.0
    sys_err = flux * SYS_ERR_FRACTION
    total_err = np.sqrt(stat_err**2 + sys_err**2)
    rel_stat = (stat_err / flux * 100.0) if flux > 0 else 0.0

    print(f"{site:<15} | {cfg['depth']:<15.2f} | {n_count:>10,} | {rel_stat:>9.2f}% | {flux:>12.4e} +/- {total_err:<12.4e}")

    sim_data[site] = {
        'depth': cfg['depth'],
        'flux': flux,
        'err': total_err,
        'color': cfg['color'],
        'marker': cfg['marker'],
        'ms': cfg['ms']
    }

print("=" * 112)


def plot_simulation_flux(include_eqn6_fit=True, output_filename="MUTE_Sims_RockCavern_0.png", target_dir=save_path):
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
        flux_label = format_flux_with_err(item['flux'], item['err'])
        plt.errorbar(
            item['depth'], item['flux'], yerr=item['err'], fmt=item['marker'],
            color=item['color'], ecolor=item['color'], ms=item['ms'],
            elinewidth=1.8, capthick=1.5, capsize=4, zorder=5,
            markeredgecolor='black', markeredgewidth=0.7, capsize=3,
            label=rf"{label} (Sim): ${flux_label}\ \mathrm{{m^{{-2}}\,s^{{-1}}}}$"
        )

    plt.yscale('log')
    plt.title(r'Muon-Induced Neutron Flux at Rock–Cavern Boundary for Each Lab', fontsize=13, pad=10)
    plt.xlabel('Depth (km.w.e.)', fontsize=14)
    plt.ylabel(r'Rock-Cavern Flux ($\text{m}^{-2}\text{s}^{-1}$)', fontsize=14)
    plt.xlim(0, 6.5)
    plt.ylim(1e-7, 1e1)

    plt.xticks(np.arange(0, 7, 1), fontsize=12)
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    plt.yticks(fontsize=12)

    plt.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=8.5, loc='upper right', frameon=True, ncol=1)
    plt.tight_layout()

    os.makedirs(target_dir, exist_ok=True)
    full_output_path = os.path.join(target_dir, output_filename)
    plt.savefig(full_output_path, dpi=300)
    print(f"Plot saved to {full_output_path}")


plot_simulation_flux(include_eqn6_fit=True, output_filename="MUTE_Sims_RockCavern_0.png")
