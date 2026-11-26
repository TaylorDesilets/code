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

SETUPS = {
    0: "Setup 0: Rock–Cavern Boundary (No NEWS-G)",
    1: "Setup 1: Copper Sphere Only",
    2: "Setup 2: Cu + Archaeological Lead",
    3: "Setup 3: Cu + Arch. Pb + Low-Activity Pb",
    4: "Setup 4: Full Detector (Cu + Pb + HDPE)",
}

SITES_META = {
    'Kamioka': {
        'file_tag': 'KamLAND',
        'depth': mute_data[0, 0],
        'area': 66.2,               # Rock-cavern boundary surface area (m^2)
        'R_gen': 8.4,               # Primary generation hemisphere radius (m)
        'Phi_mu': mute_data[0, 2],
        'color': '#bcbd22',         # Yellow-green
        'marker': '*',
        'ms': 10
    },
    'Boulby': {
        'file_tag': 'Boulby',
        'depth': mute_data[1, 0],
        'area': 66.2,
        'R_gen': 8.4,
        'Phi_mu': mute_data[1, 2],
        'color': '#0000FF',         # Blue
        'marker': '*',
        'ms': 10
    },
    'Gran Sasso': {
        'file_tag': 'LNGS',
        'depth': mute_data[2, 0],
        'area': 66.2,
        'R_gen': 8.4,
        'Phi_mu': mute_data[2, 2],
        'color': '#e377c2',         # Pink
        'marker': '*',
        'ms': 10
    },
    'SNOLAB': {
        'file_tag': 'SNOLAB',
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
SYS_ERR_FRACTION = np.sqrt(sum(v**2 for v in SYS_COMPONENTS.values()))


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

    M_sim = n_count / n_mu_simulated if n_mu_simulated > 0 else 0.0
    return n_count, E_mu_mean, n_mu_simulated, M_sim


def get_multiplicity_correction_factor(E_mu):
    return 1.0 / (1.0 - 0.314 * (E_mu**0.128) + 1.68e6 * (E_mu**-5.793))


def mei_hime_flux(X):
    P0 = 4.0e-3  # m^-2 s^-1
    P1 = 0.86    # km.w.e.
    return P0 * (P1 / X) * np.exp(-X / P1)


def param_model(X, A, X0, n):
    return A * ((X0 / X) ** n) * np.exp(-X / X0)


def sim_livetime(n_mu_simulated, phi_mu, r_gen):
    a_gen = 2.0 * np.pi * (r_gen ** 2)
    return n_mu_simulated / (phi_mu * a_gen)


def format_flux_with_err(val, err):
    if val == 0:
        return r"0.00 \pm 0.00"
    exp = int(np.floor(np.log10(abs(val))))
    mant_val = val / (10 ** exp)
    mant_err = err / (10 ** exp)
    return rf"({mant_val:.2f} \pm {mant_err:.2f}) \times 10^{{{exp}}}"


def process_setup(setup_idx):
    print("=" * 112)
    print(f"{SETUPS[setup_idx]} | Total Fractional Systematic Uncertainty: {SYS_ERR_FRACTION:.2%}")
    print("-" * 112)
    print(f"{'Site':<15} | {'Depth (km.w.e.)':<15} | {'N_neutrons':>10} | {'Stat Err':>10} | {'Flux +/- Total Err (m^-2 s^-1)':>32}")
    print("-" * 112)

    sim_data = {}
    for site, cfg in SITES_META.items():
        file_path = os.path.join(BASE_DIR, f"{setup_idx}_{cfg['file_tag']}.csv")
        n_count, E_mu_calculated, N_mu_simulated, M_sim = get_site_metrics(file_path)
        if n_count is None:
            continue

        f_corr = get_multiplicity_correction_factor(E_mu_calculated)
        M_corr = f_corr * M_sim
        N_s = M_corr * N_mu_simulated

        t_live = sim_livetime(N_mu_simulated, cfg['Phi_mu'], cfg['R_gen'])
        flux = N_s / (t_live * cfg['area'])

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

    print("=" * 112 + "\n")
    return sim_data


def plot_all_setups_grid(all_sim_data, include_eqn6_fit=True, output_filename="MUTE_Sims_AllSetups.png", target_dir=save_path):
    # 2x3 grid for 5 plots; the 6th slot will hold a shared legend or remain blank
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    X_mesh = np.linspace(0.05, 6.4, 300)
    flux_mh = mei_hime_flux(X_mesh)

    for setup_idx in range(5):
        ax = axes_flat[setup_idx]
        sim_data = all_sim_data.get(setup_idx, {})

        # Mei & Hime Reference Curves
        ax.plot(X_mesh[X_mesh >= 0.8], flux_mh[X_mesh >= 0.8], 'k-', linewidth=1.4, label='M&H Model')
        ax.plot(X_mesh[X_mesh <= 0.8], flux_mh[X_mesh <= 0.8], 'k--', linewidth=1.4, label='M&H Extrapolated')
        ax.fill_between(X_mesh, flux_mh * 0.5, flux_mh * 2.0, color='gray', alpha=0.18, label=r'M&H Model $\pm 1\sigma$')

        # Fit Eqn 6 only to positive flux points
        valid_pts = {k: v for k, v in sim_data.items() if v['flux'] > 0}
        if include_eqn6_fit and len(valid_pts) >= 3:
            x_fit = np.array([d['depth'] for d in valid_pts.values()])
            y_fit = np.array([d['flux'] for d in valid_pts.values()])
            y_err = np.array([d['err'] for d in valid_pts.values()])

            def log_param_model(X, A, X0, n):
                return np.log10(param_model(X, A, X0, n))

            y_fit_log = np.log10(y_fit)
            y_err_log = y_err / (y_fit * np.log(10))

            try:
                popt, _ = curve_fit(
                    log_param_model, x_fit, y_fit_log,
                    sigma=y_err_log,
                    absolute_sigma=True,
                    p0=[0.003, 1.015, 1.562],
                    bounds=([1e-8, 0.1, 0.1], [0.1, 10.0, 5.0]),
                    maxfev=50000
                )
                flux_fit = param_model(X_mesh, *popt)
                ax.plot(X_mesh, flux_fit, 'r--', linewidth=1.2, label=r'Sim Fit: $\Phi_n = A (\frac{X_0}{X})^n e^{-X/X_0}$')
                ax.fill_between(X_mesh, flux_fit * 0.6, flux_fit * 1.6, color='red', alpha=0.12, label=r'Sim Fit $\pm 1\sigma$')
            except RuntimeError:
                print(f"Warning: Curve fit did not converge for Setup {setup_idx}.")

        # Plot Site Points
        for label, item in sim_data.items():
            if item['flux'] <= 0:
                continue
            flux_label = format_flux_with_err(item['flux'], item['err'])
            ax.errorbar(
                item['depth'], item['flux'], yerr=item['err'], fmt=item['marker'],
                color=item['color'], ecolor=item['color'], ms=item['ms'],
                elinewidth=1.6, capthick=1.3, capsize=3.5, zorder=5,
                markeredgecolor='black', markeredgewidth=0.7,
                label=rf"{label}: ${flux_label}$"
            )

        ax.set_yscale('log')
        ax.set_title(SETUPS[setup_idx], fontsize=11, fontweight='bold', pad=8)
        ax.set_xlim(0, 6.5)
        ax.set_ylim(1e-9, 1e1)  # Extended lower bound to accommodate attenuated shielding setups

        # Ticks & Grid formatting
        ax.set_xticks(np.arange(0, 7, 1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.tick_params(axis='x', which='major', labelsize=11, length=5)
        ax.tick_params(axis='x', which='minor', length=3)

        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis='y', which='major', labelsize=11, length=5)
        ax.tick_params(axis='y', which='minor', length=2.5)

        ax.grid(True, which='both', color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=7.5, loc='upper right', frameon=True, ncol=1)

    # Turn off the empty 6th subplot (index 5)
    axes_flat[5].axis('off')

    # Shared axis labels and overall title
    fig.supxlabel('Depth (km.w.e.)', fontsize=14, y=0.02)
    fig.supylabel(r'Neutron Flux ($\mathrm{m}^{-2}\mathrm{s}^{-1}$)', fontsize=14, x=0.01)
    fig.suptitle('Muon-Induced Neutron Flux Across Detector Shielding Configurations', fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    os.makedirs(target_dir, exist_ok=True)
    full_output_path = os.path.join(target_dir, output_filename)
    plt.savefig(full_output_path, dpi=300)
    print(f"Combined plot saved to {full_output_path}")


if __name__ == "__main__":
    all_results = {}
    for idx in SETUPS:
        all_results[idx] = process_setup(idx)

    plot_all_setups_grid(all_results, include_eqn6_fit=True, output_filename="MUTE_Sims_AllSetups.png")
