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

read_electrons = False
material = qe.from_disk(name, prepare_pseudopotentials=read_electrons)
if read_electrons:
    electrons = material.read_valence_electrons() / 2
else:
    electrons = 5.5

tb = BandStructureModel.load(f"{name}_asym_4.json")

dos_model = dos.DensityOfStates(tb, A=qe.fcc(1.0), N=48)
print("DoS model prepared")
e_smpl, states, density = dos_model.full_curve(N=20, T=0)
fermi_energy_dos = dos_model.fermi_energy(electrons)
print(fermi_energy_dos)
plt.plot(e_smpl, states)
plt.plot(e_smpl, density)
plt.axvline(fermi_energy_dos, color="k")
plt.show()

kint = bulk.KIntegral(dos_model, electrons, T=[0.0, 300.0])
# compute the hall coefficient R_h
hall_coeff = kint.hall_coefficient_metal_cubic(material.A, spin_factor=2)
print("T=0K:", hall_coeff[0], "m³/C")
print("T=300K:", hall_coeff[1], "m³/C")

# TODO symmetrize
