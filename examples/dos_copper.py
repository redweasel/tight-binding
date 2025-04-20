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
k_smpl, ref_bands, S, fermi_energy = material.read_bands_crystal()
sym = Symmetry.point_group(S)
A_norm = material.A / (abs(material.A[0,0])*2) # this is the normalisation for fcc and bcc

tb = BandStructureModel.load(f"{name}_asym_4.json")

# show a comparison plot
interp2 = kpaths.interpolate(k_smpl, ref_bands, sym)
interp = lambda k: interp2(k @ A_norm)
kpaths.FCC_PATH.plot_comparison(tb, interp, label_bands="left")
plt.axhline(fermi_energy, color="k")
plt.show()

dos_model = dos.DensityOfStates(tb, A=qe.fcc(1.0), N=32)
electrons = round(dos_model.states_below(fermi_energy) * 2) / 2
print("found", electrons * 2, "valence electrons")
fermi_energy_dos = dos_model.fermi_energy(electrons)
print("computed Fermi-Energy: ", fermi_energy_dos)
print("qespresso Fermi-Energy:", fermi_energy)
e_smpl, states, density = dos_model.full_curve(N=20, T=0)
plt.plot(e_smpl, states)
plt.plot(e_smpl, density)
plt.axvline(fermi_energy_dos, color="k")
plt.show()
