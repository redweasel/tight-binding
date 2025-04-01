import io
import re
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.testing as mpl_test
mpl_test.setup()

from tight_binding_redweasel import kpaths
from tight_binding_redweasel import symmetry

# in prior versions I had a test for the plot, but that
# is very unstable due to matplotlib changing its backends all the time.

def test_kpaths_interpolate():
    np.random.seed(13**5)
    k_smpl = np.array([(0, 0), (0, 0.5), (0, 1), (0.5, 0.5), (0.5, 1), (1, 1)])
    bands = np.random.random((len(k_smpl), 2))
    interp = kpaths.interpolate(k_smpl, bands, sym=symmetry.Symmetry.square(), periodic=False)
    for k, band in zip(k_smpl, bands):
        assert np.linalg.norm(interp(k) - band) < 1e-12, f"{interp(k)} != {band}"
        assert np.linalg.norm(interp(-k) - band) < 1e-12, f"{interp(-k)} != {band}"


def test_kpaths_interpolate():
    np.random.seed(13**5)
    k_smpl = np.array([(0, 0), (0, 0.5), (0.5, 0.5)])
    bands = np.random.random((len(k_smpl), 2))
    interp = kpaths.interpolate(k_smpl, bands, sym=symmetry.Symmetry.square(), periodic=True)
    for k, band in zip(k_smpl, bands):
        assert np.linalg.norm(interp(k) - band) < 1e-12, f"{interp(k)} != {band}"
        assert np.linalg.norm(interp(-k) - band) < 1e-12, f"{interp(-k)} != {band}"
        assert np.linalg.norm(interp(k+1) - band) < 1e-12, f"{interp(k+1)} != {band}"
