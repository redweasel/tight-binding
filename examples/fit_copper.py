assert __name__ == "__main__", "Run as file, not import"
import sys
sys.path.append("./../src")

import numpy as np
np.set_printoptions(precision=3, suppress=True)
from tight_binding_redweasel import *
from tight_binding_redweasel.logger import OptimisationLogger
from tight_binding_redweasel import qespresso_interface as qe

name = "cu"
neighbors_src = [(0,0,0), (0.5,0.5,0), (1,0,0), (0.5, 1, 0.5), (1,1,0)]
band_count = 9 # the number of bands to fit
band_offset = 0 # the number of lower bands to skip while fitting
add_bands_below = 0 # the number of unfitted bands added to the model
add_bands_above = 0 # the number of unfitted bands added to the model
start_neighbors_count = 2 # non zero neighbors in the first fitting step

# read the data from Quantum Espresso
material = qe.from_disk(name)
k_smpl, ref_bands, S, fermi_energy = material.read_bands_crystal()
sym = Symmetry.point_group(S)

A_norm = material.A / (abs(material.A[0,0])*2) # this is the normalisation for fcc and bcc
k_smpl_fit, ref_bands_sym = sym.realize_symmetric_data(k_smpl, ref_bands)
k_smpl_fit = k_smpl_fit @ np.linalg.inv(A_norm)

assert band_offset >= 0 and band_count > 0 and band_count + band_offset <= ref_bands.shape[-1]
# reducing the weight on the top bands has increased the precision A LOT!
band_weights = [1]*(band_count-2) + [0.01]*2
ref_bands_fit = ref_bands_sym[:, band_offset:band_count + band_offset]
sym = Symmetry.cubic(True) # for the neighbors, use the real space symmetry

# the following fitting procedure is implemented in tight_binding_redweasel.tight_binding_asym.autofit_asym(...)
# however it is sometimes beneficial to work with the bare code to
# e.g. continue from some checkpoint or skip initial selection.
# For that, copy the code out of this function into your code and adjust it.
# The function autofit_asym still contains understandable high level code.
tb = tight_binding_asym.autofit_asym(
    name,
    neighbors_src,
    k_smpl_fit,
    ref_bands_fit,
    band_weights,
    sym,
    start_neighbors_count=start_neighbors_count,
    add_bands_below=add_bands_below,
    add_bands_above=add_bands_above,
    randomize=True
)
# show final error
tb.print_error(k_smpl_fit, ref_bands_fit, band_weights, add_bands_below)
