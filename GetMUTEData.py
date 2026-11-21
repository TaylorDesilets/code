import numpy as np
import mute.constants as mtc
import mute.underground as mtu

# Flat-overburden vertical depths in km.w.e.
flat_depths = {
    "KamLAND": 2.05,
    "Boulby":  2.84,
    "LNGS":    3.10,
    "SNOLAB":  5.89,
}

mtc.set_output(False)
flux_summary = []

for lab_name, depth in flat_depths.items():
    print(f"--- Generating Flat-Overburden Data for {lab_name} ({depth} km.w.e.) ---")

    # 1. Load lab's precomputed rock composition & transfer tensor[cite: 6, 7]
    mtc.set_lab(lab_name)

    # 2. Override overburden mode to flat at the lab's vertical depth
    mtc.set_overburden("flat")
    mtc.set_vertical_depth(depth)

    # 3. Total integrated underground flux [cm^-2 s^-1]
    tot_flux_cm2 = float(np.squeeze(mtu.calc_u_tot_flux()))
    tot_flux_m2 = tot_flux_cm2 * 1e4
    print(f"  Total Flux: {tot_flux_cm2:.4e} cm^-2 s^-1 ({tot_flux_m2:.4e} m^-2 s^-1)")
    flux_summary.append((lab_name, depth, tot_flux_cm2, tot_flux_m2))

    # 4. Differential energy spectrum at this vertical depth -> normalized CDF (0 -> 1)
    e_spect = np.squeeze(mtu.calc_u_e_spect())
    energies = np.array(mtc.ENERGIES, dtype=float)
    bin_integrals = 0.5 * (e_spect[:-1] + e_spect[1:]) * np.diff(energies)
    energy_cdf = np.concatenate([[0.0], np.cumsum(bin_integrals)])
    energy_cdf /= energy_cdf[-1]

    np.savetxt(f"/home/taylor33/code/MUTEFiles/MUTE_{lab_name}_underground_energy_CDFs.txt", energy_cdf)

np.savetxt(
    "/home/taylor33/code/MUTEFiles/MUTE_flat_overburden_fluxes.txt",
    np.array([[d, f_cm2, f_m2] for _, d, f_cm2, f_m2 in flux_summary]),
    header="vertical_depth_kmwe flux_cm2_s flux_m2_s",
)
print("\nSaved MUTE_flat_overburden_fluxes.txt and lab-specific energy CDFs.")
