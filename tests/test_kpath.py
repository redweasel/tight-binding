import io
import re
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.testing as mpl_test
mpl_test.setup()

from tight_binding_redweasel import kpaths
from tight_binding_redweasel import symmetry

def test_kpaths_plot_svg():
    # testing matplotlib results is a pain...
    # svg doesn't work, because svg.hashsalt handling differs over multiple matplotlib versions.
    # png doesn't work because it includes the matplotlib version, even with 'Creator': None
    # pickle doesn't work because it includes the matplotlib version

    # the solution is to use svg, but remove some id data using regex
    read = True

    plt.figure(figsize=(5, 5), frameon=False)
    kpaths.SC_PATH.plot(lambda x: x, band_offset=1, label_bands="left", ylim=(-1.0, 1.0))
    s = io.StringIO()
    plt.savefig(s, format="svg", bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    svg = s.getvalue()
    svg = re.sub(r"url\(#\w+\)", r"url\(\)", svg)
    svg = re.sub(r'id="\w+"', r"", svg)
    if read:
        with open("tests/ref_kpath_left.svg", "r") as file:
            f = file.read()
            assert svg == f, f"generated svg ({len(svg)}) doesn't match saved svg ({len(f)})"
    else:
        with open("tests/ref_kpath_left.svg", "w") as file:
            file.write(svg)

    plt.figure(figsize=(5, 5), frameon=False)
    kpaths.FCC_PATH.plot(lambda x: x, band_offset=1, label_bands="right", ylim=(-1.0, 1.0))
    s = io.StringIO()
    plt.savefig(s, format="svg", bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    svg = s.getvalue()
    svg = re.sub(r"url\(#\w+\)", r"url\(\)", svg)
    svg = re.sub(r'id="\w+"', r"", svg)
    if read:
        with open("tests/ref_kpath_right.svg", "r") as file:
            f = file.read()
            assert svg == f, f"generated svg ({len(svg)}) doesn't match saved svg ({len(f)})"
    else:
        with open("tests/ref_kpath_right.svg", "w") as file:
            file.write(svg)


def test_kpaths_interpolate():
    np.random.seed(13**5)
    k_smpl = np.array([(0, 0), (0, 0.5), (0, 1), (0.5, 0.5), (0.5, 1), (1, 1)])
    bands = np.random.random((len(k_smpl), 2))
    interp = kpaths.interpolate(k_smpl, bands, sym=symmetry.Symmetry.square())
    for k, band in zip(k_smpl, bands):
        assert np.linalg.norm(interp(k) - band) < 1e-12, f"{interp(k)} != {band}"
        assert np.linalg.norm(interp(-k) - band) < 1e-12, f"{interp(-k)} != {band}"

test_kpaths_plot_svg()