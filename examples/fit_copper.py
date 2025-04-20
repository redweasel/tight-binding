assert __name__ == "__main__", "Run as file, not import"
import sys
sys.path.append("./../src")

import numpy as np
from matplotlib import pyplot as plt
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

assert band_offset >= 0 and band_count > 0 and band_count + band_offset < ref_bands.shape[-1]
# reducing the weight on the top bands has increased the precision A LOT!
band_weights = [1]*(band_count-2) + [0.01]*2
ref_bands_fit = ref_bands_sym[:, band_offset:band_count + band_offset]
sym = Symmetry.cubic(True) # for the neighbors, use the real space symmetry

# the following fitting procedure is also implemented in tight_binding_redweasel.tight_binding_asym.autofit_asym(...)

assert add_bands_below >= 0 and add_bands_above >= 0

# the fitting requires (k_smpl_sym, ref_bands_fit, band_weights, neighbors)
# the best fitting protocol requires sorting k by distance to 0.
reorder = np.argsort(np.linalg.norm(k_smpl_fit, axis=-1))
k_smpl_fit = k_smpl_fit[reorder]
ref_bands_fit = ref_bands_fit[reorder]

neighbors_count = start_neighbors_count + 1
neighbors = sym.complete_neighbors(neighbors_src[:neighbors_count])

if True:
    best_tb_error = float("inf")
    best_tb = None # type: AsymTightBindingModel
    for _ in range(4):
        tb = AsymTightBindingModel.new(neighbors, add_bands_below + band_count + add_bands_above)
        tb.randomize(0.01)
        tb.normalize()
        tb.H.H_r[0] = np.diag(ref_bands_fit[0])
        # now fit in 10 steps extending from the gamma point (0-point)
        log = OptimisationLogger(update_line=False)
        for j in range(1, 11):
            n = (j * len(k_smpl_fit) + 9) // 10
            tb.optimize_cg(k_smpl_fit[:n], ref_bands_fit[:n], band_weights, add_bands_below, 1, max_cg_iterations=10, log=log)
        if log.last_loss() < best_tb_error:
            best_tb = tb
            best_tb_error = log.last_loss()
    assert best_tb is not None
    tb = best_tb
    # save checkpoint
    tb.save(f"{name}_asym_start.json")
    print(f"saved {name}_asym_start.json")
else:
    tb = AsymTightBindingModel.load(f"{name}_asym_start.json")

# now fit with increasing amount of neighbors
log = OptimisationLogger(update_line=False)
l, err = tb.error(k_smpl_fit, ref_bands_fit, band_weights, add_bands_below)
log.add_data(0, l, err)
while neighbors_count < len(neighbors_src):
    if start_neighbors_count + 1 != neighbors_count:
        new_neighbors = sym.complete_neighbors([neighbors_src[neighbors_count]])
        tb.H.add_neighbors(new_neighbors)
    print("fit with neighbors", neighbors_src[:neighbors_count])
    for _ in range(10):
        # small randomisation seems to help convergence...?!?
        tb.randomize(log.last_loss() * 0.1)
        tb.normalize()
        tb.optimize_cg(k_smpl_fit, ref_bands_fit, band_weights, add_bands_below, 100, convergence_threshold=1e-3, max_cg_iterations=5, log=log)
    tb.save(f"{name}_asym_{neighbors_count}.json")
    print(f"saved {name}_asym_{neighbors_count}.json")
    neighbors_count += 1

# show final error
tb.print_error(k_smpl_fit, ref_bands_fit, band_weights, add_bands_below)
