import io
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.testing as mpl_test
mpl_test.setup()

from tight_binding_redweasel import kpaths
from tight_binding_redweasel import symmetry

# disabled this test, as it doesn't work over multiple matplotlib versions...
def _test_kpaths_plot_svg():
    # testing matplotlib results is a pain...
    # the svg.hashsalt doesn't seem consistent between versions.
    # so I have to either test png or figure out a way to remove the hashes from the output.

    plt.figure(figsize=(5, 5), frameon=False)
    kpaths.SC_PATH.plot(lambda x: x, band_offset=1, label_bands="left", ylim=(-1.0, 1.0))
    s = io.StringIO()
    plt.savefig(s, format="svg", bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    svg = s.getvalue()
    with open("tests/ref_kpath_left.svg", "r") as file:
        f = file.read()
        assert svg == f, f"generated svg ({len(svg)}) doesn't match saved svg ({len(f)})"
    #with open("tests/ref_kpath_left.svg", "w") as file:
    #    file.write(svg)

    plt.figure(figsize=(5, 5), frameon=False)
    kpaths.FCC_PATH.plot(lambda x: x, band_offset=1, label_bands="right", ylim=(-1.0, 1.0))
    s = io.StringIO()
    plt.savefig(s, format="svg", bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    svg = s.getvalue()
    with open("tests/ref_kpath_right.svg", "r") as file:
        f = file.read()
        assert svg == f, f"generated svg ({len(svg)}) doesn't match saved svg ({len(f)})"
    #with open("tests/ref_kpath_right.svg", "w") as file:
    #    file.write(svg)


def test_kpaths_plot_png():
    # testing matplotlib results is a pain...
    # the svg.hashsalt doesn't seem consistent between versions.
    # so I have to either test png or figure out a way to remove the hashes from the output.

    plt.figure(figsize=(5, 5), frameon=False)
    kpaths.SC_PATH.plot(lambda x: x, band_offset=1, label_bands="left", ylim=(-1.0, 1.0))
    s = io.BytesIO()
    plt.savefig(s, format="png", bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    png = s.getbuffer().tobytes()
    with open("tests/ref_kpath_left.png", "rb") as file:
        f = file.read()
        assert png == f
    #with open("tests/ref_kpath_left.png", "wb") as file:
    #    file.write(png)

    plt.figure(figsize=(5, 5), frameon=False)
    kpaths.FCC_PATH.plot(lambda x: x, band_offset=1, label_bands="right", ylim=(-1.0, 1.0))
    s = io.BytesIO()
    plt.savefig(s, format="png", bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    png = s.getbuffer().tobytes()
    with open("tests/ref_kpath_right.png", "rb") as file:
        f = file.read()
        assert png == f
    #with open("tests/ref_kpath_right.png", "wb") as file:
    #    file.write(png)


def test_kpaths_interpolate():
    np.random.seed(13**5)
    k_smpl = np.array([(0, 0), (0, 0.5), (0, 1), (0.5, 0.5), (0.5, 1), (1, 1)])
    bands = np.random.random((len(k_smpl), 2))
    interp = kpaths.interpolate(k_smpl, bands, sym=symmetry.Symmetry.square())
    for k, band in zip(k_smpl, bands):
        assert np.linalg.norm(interp(k) - band) < 1e-12, f"{interp(k)} != {band}"
        assert np.linalg.norm(interp(-k) - band) < 1e-12, f"{interp(-k)} != {band}"
