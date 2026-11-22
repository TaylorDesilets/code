####### Generate Flat-Overburden Angular Intensities for Geant4 from MUTE v3

import os
import numpy as np
import mute.constants as mtc
import mute.underground as mtu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lab names and flat vertical depths (km.w.e.)
LABS = [
    ("KamLAND", 2.05),
    ("Boulby",  2.84),
    ("LNGS",    3.10),
    ("SNOLAB",  5.89),
]

# Configure global MUTE settings
mtc.set_verbose(2)
mtc.set_output(False)
mtc.set_overburden("flat")
mtc.set_medium("rock")
mtc.set_reference_density(2.65)

# Uniform azimuth grid (0 to 355 deg in 5 deg steps) so azimuth is sampled uniformly across [0, 360)
AZIMUTH_BINS = np.arange(0.0, 360.0, 5.0)

for lab, depth_kmwe in LABS:
    print(f"--- Generating Flat-Overburden Intensities for {lab} ({depth_kmwe} km.w.e.) ---")
    mtc.set_lab(lab)
    mtc.set_vertical_depth(depth_kmwe)

    # Calculate underground intensities across MUTE's zenith angle grid
    intensities = mtu.calc_u_intensities()
    u_int = np.squeeze(intensities)

    # MUTE v3's internal zenith angle grid in degrees
    angles_deg = mtc._ANGLES

    out_path = os.path.join(SCRIPT_DIR, f"MUTE_{lab}_flat_intensities.txt")
    with open(out_path, "w") as f:
        for zen, val in zip(angles_deg, u_int):
            if np.isfinite(val) and val > 0:
                # Weight by sin(theta) so Geant4's discrete CDF reproduces dN/dtheta ~ I(theta) * sin(theta)
                weight = val * np.sin(np.radians(zen))
                for az in AZIMUTH_BINS:
                    f.write(f"{zen:.4f} {az:.4f} {weight:.8e}\n")

    print(f"Saved {out_path}")

print("\nAll 4 flat-overburden intensity files generated successfully.")
