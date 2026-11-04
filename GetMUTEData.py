import os
import numpy as np
import mute.constants as mtc
import mute.underground as mtu

save_dir = '/home/taylor33/code/MUTEFiles'

built_in_labs = ["SNOLAB", "LNGS", "Boulby", "KamLAND"]

for lab_name in built_in_labs:
    print(f"--- Processing Built-in Lab: {lab_name} ---")
    mtc.set_lab(lab_name)
    mtu.calc_u_fluxes()
    angles = mtc.angles        # Zenith angle grid
    energies = mtc.ENERGIES    # Energy grid

    # (calc_u_intensities / calc_u_e_spect store computed matrices in mtu)
    intensities = mtu.calc_u_intensities()
    energy_cdfs = mtu.calc_u_e_spect()

    angular_file = os.path.join(save_dir, f"MUTE_{lab_name}_underground_intensities_dd.txt")
    cdf_file = os.path.join(save_dir, f"MUTE_{lab_name}_underground_energy_CDFs.txt")

    np.savetxt(angular_file, intensities)
    np.savetxt(cdf_file, energy_cdfs)
    print(f"Saved: {angular_file}")
    print(f"Saved: {cdf_file}")

