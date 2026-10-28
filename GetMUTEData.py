import numpy as np
import mute.constants as mtc
import mute.underground as mtu
# "lab_key": (medium, reference_density, mountain_name)
mountain_labs = {
    "Y2L": ("y2l_rock", 2.70, "Y2L"),
    "SuperK": ("rock", 2.70, "SuperK"),
    "KamLAND": ("rock", 2.70, "KamLAND"),
    "LNGS": ("rock", 2.72, "LNGS"),
    "LSM": ("frejus_rock", 2.73, "LSM"),
    "CJPL": ("rock", 2.80, "CJPL"),
}

for lab_name, (medium, density, mountain_file) in mountain_labs.items():
    print(f"--- Processing {lab_name} ---")

    mtc.set_medium(medium)
    mtc.set_reference_density(density)
    mtc.load_mountain(mountain_file)  # Loads {mountain_file.lower()}_mountain.txt

    mtu.calc_u_fluxes()

    angles = mtu.get_angles()      # Zenith angles in degrees
    energies = mtu.get_energies()  # Underground muon energies in GeV

    intensities = mtu.get_underground_intensities()
    energy_cdfs = mtu.get_underground_energy_cdfs()

    angular_file = f"Geant4_{lab_name}_underground_intensities_dd.txt"
    np.savetxt(angular_file, intensities)
    print(f"Saved: {angular_file}")

    cdf_file = f"Geant4_{lab_name}_underground_energy_CDFs.txt"
    np.savetxt(cdf_file, energy_cdfs)
    print(f"Saved: {cdf_file}")