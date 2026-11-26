import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Paths & Configuration
# ==========================================
BASE_DIR = "/home/taylor33/scratch"
SAVE_DIR = "/home/taylor33/code/plots/neutron_spectra_layers"
os.makedirs(SAVE_DIR, exist_ok=True)

LAYERS = {
    0: {
        "short": "layer0_rock_cavern",
        "title": "Layer 0: Rock Cavern Boundary Only",
    },
    1: {
        "short": "layer1_cu_sphere",
        "title": "Layer 1: Copper Sphere Only",
    },
    2: {
        "short": "layer2_cu_ancient_pb",
        "title": "Layer 2: Copper Sphere + Ancient Lead",
    },
    3: {
        "short": "layer3_cu_ancient_lowbg_pb",
        "title": "Layer 3: Copper Sphere + Ancient Lead + Low-Background Lead",
    },
    4: {
        "short": "layer4_full_detector_hdpe",
        "title": "Layer 4: Full Detector with HDPE Covering",
    },
}

LABS = [
    {"name": "KamLAND", "color": "#FFBF00"},
    {"name": "Boulby",  "color": "#3F1FE0"},
    {"name": "LNGS",    "color": "#D02F9D"},
    {"name": "SNOLAB",  "color": "#3BC472"},
]

# ==========================================
# 2. Helper Functions
# ==========================================
def detect_energy_column(df):
    """Automatically finds the kinetic energy column in the Geant4 CSV."""
    candidates = [
        "KineticEnergy", "kineticEnergy", "KineticEnergy (MeV)",
        "Energy", "energy", "Energy (MeV)",
        "SecondaryEnergy", "secondaryEnergy", "Ekin", "ekin",
        "NeutronEnergy", "neutronEnergy"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # Fallback: look for any column containing 'energy' or 'ekin' (excluding muon primary energy)
    for col in df.columns:
        low = col.lower()
        if ("energy" in low or "ekin" in low) and "muon" not in low and "initial" not in low:
            return col
    raise ValueError(f"Could not identify neutron energy column. Available columns: {list(df.columns)}")


def load_neutron_energies(filepath):
    """Loads CSV, filters for neutrons if needed, and returns positive energy values."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Filter for neutrons if the file contains mixed secondary particles
    for p_col in ["secondaryName", "particleName", "ParticleName", "particle"]:
        if p_col in df.columns:
            df = df[df[p_col].astype(str).str.strip() == "neutron"]
            break

    energy_col = detect_energy_column(df)
    energies = pd.to_numeric(df[energy_col], errors="coerce").dropna()
    energies = energies[energies > 0].values  # Keep strictly positive energies for log binning
    return energies, energy_col


# ==========================================
# 3. Generate One Plot Per Layer (5 Total)
# ==========================================
# Logarithmic energy bins spanning thermal/epithermal up to GeV scale (in MeV)
# Adjust 1e-9 to 1e-3 if your simulation cuts off below 1 keV
BINS = np.logspace(-9, 4, 80)

for layer_idx, layer_info in LAYERS.items():
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    detected_col_label = "Energy (MeV)"

    for lab in LABS:
        lab_name = lab["name"]
        csv_path = os.path.join(BASE_DIR, f"{layer_idx}_{lab_name}.csv")

        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing file: {csv_path}")
            continue

        energies, energy_col = load_neutron_energies(csv_path)
        detected_col_label = energy_col
        n_total = len(energies)

        if n_total == 0:
            print(f"[WARNING] 0 neutrons found in {csv_path}")
            continue

        # Step histogram overlays cleanly without obscuring overlapping curves
        ax.hist(
            energies,
            bins=BINS,
            histtype="step",
            linewidth=1.8,
            color=lab["color"],
            label=f"{lab_name} (N = {n_total:,})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Neutron Kinetic Energy [{detected_col_label}]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Counts", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Neutron Energy Spectrum — {layer_info['title']}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
    ax.legend(title="Underground Laboratory", fontsize=10.5, title_fontsize=11, frameon=True, loc="upper right")

    plt.tight_layout()
    out_file = os.path.join(SAVE_DIR, f"{layer_info['short']}_neutron_spectrum.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")

print("\nDone! All 5 layer plots saved to:", SAVE_DIR)
