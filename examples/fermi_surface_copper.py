assert __name__ == "__main__", "Run as file, not import"
import sys
sys.path.append("./../src")

import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(precision=3, suppress=True)
from tight_binding_redweasel import *
from tight_binding_redweasel import qespresso_interface as qe
from tight_binding_redweasel import fermi_surface as fs

name = "cu"

material = qe.from_disk(name)
_, _, _, fermi_energy = material.read_bands_crystal()
print("Fermi-Energy:", fermi_energy)

tb = BandStructureModel.load(f"{name}_asym_4.json")

print("computing 2D Fermi Surfaces")
fs.plot_2D_fermi_surface(tb, fermi_energy, N=100, k_range=[-1, 1])
print("computing 3D Fermi Surface")
fs.plot_3D_fermi_surface(tb, fermi_energy, N=30, k_range=[-1, 1])
