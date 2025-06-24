from collections.abc import Iterator, Callable, Sequence
import numpy as np
from . import symmetry as _sym

points = {}
points['G'] = (np.array([0, 0, 0]), 'Γ')
points['O'] = (np.array([0, 0, 0]), 'Γ')
# face centered cubic in tpiba_b units:
points['X2'] = (np.array([0, 1, 0]), 'X')  # Delta line
points['K'] = (np.array([3 / 4, 3 / 4, 0]), 'K')  # Sigma line
points['L'] = (np.array([1 / 2, 1 / 2, 1 / 2]), 'L')  # Lambda line
points['W'] = (np.array([1 / 2, 1, 0]), 'W')  # between X and K
points['U'] = (np.array([1 / 4, 1, 1 / 4]), 'U')  # between X and L
# simple cubic points:
points['X'] = (0.5 * np.array([0, 1, 0]), 'X')  # Delta line
points['M'] = (0.5 * np.array([1, 1, 0]), 'M')  # Sigma line
points['R'] = (0.5 * np.array([1, 1, 1]), 'R')  # Lambda line
# body centered points:
points['H'] = (np.array([1, 0, 0]), 'H')  # Delta line
points['H1'] = (np.array([0, 1, 0]), '$H_1$')
points['N'] = (np.array([1 / 2, 1 / 2, 0]), 'N')  # Sigma line
points['P'] = (np.array([1 / 2, 1 / 2, 1 / 2]), 'P')  # Lambda line
# 2D square symmetry points:
points['G2d'] = (0.5 * np.array([0, 0]), 'Γ')
points['X2d'] = (0.5 * np.array([0, 1]), 'X')  # Delta line
points['X12d'] = (0.5 * np.array([1, 0]), '$X_1$')
points['M2d'] = (0.5 * np.array([1, 1]), 'M')  # Sigma line
# 1D points
points['G1d'] = (0.5 * np.array([0]), 'Γ')
points['X1d'] = (0.5 * np.array([1]), 'X')


class KPath(Sequence):
    def __init__(self, start, name=None, points=points):
        """create new path in k-space.
        The dimension of the k-space is inferred from the start point.

        Args:
            start (str, array-like): Either a pointname from `kpaths.points.keys()` like "G" or an explicit point in k-space like (0, 0, 0). A full list of those types is also accepted and will be filled with the default point density.
            name (str, optional): The name of the point. This is only needed if the point is given as an array. Defaults to None.
            points (dict): High-symmetry-point name/position dictionary. Defaults to a dictionary with sc, fcc, bcc, square symmetries.
        """
        self.points = points  # no copy, be careful!
        self.indices = [0]  # symmetry point indices
        additional = []
        if isinstance(start, list) or isinstance(start, tuple):
            # accept a list of points instead of start
            additional = start[1:]
            start = start[0]
        if isinstance(start, str):
            self.path = [points[start][0].reshape(-1)]
            self.names = [name if name is not None else points[start][1]]
        else:
            self.path = [np.array(start).reshape(-1)]
            self.names = [name if name is not None else str(
                np.array(start).reshape(-1)).replace("  ", " ")]
        for point in additional:
            self.to(point)

    def to(self, point, name=None, N=32):
        """add a new (symmetry) waypoint to the end of the path.

        Args:
            point (str, array-like): Either a pointname from `kpath.points.keys()` or an explicit point in k-space.
            N (int, optional): The number of interpolated points from the last point in the path. Defaults to 32.
            name (str, optional): The name of the point. This is only needed if the point is given as an array. Defaults to None.

        Returns:
            Self: self
        """
        if isinstance(point, str):
            self.names.append(
                name if name is not None else self.points[point][1])
            point = self.points[point][0].reshape(1, -1)
        else:
            self.names.append(
                name if name is not None else "(" + " ".join(f"{x:.3f}" for x in point) + ")")
            point = np.array(point).reshape(1, -1)
        assert len(point[0]) == len(
            self.path[0]), f"All points in the path need to have the same dimension. Tried to add a {len(point[0])}d point to a {len(self.path[0])}d path."
        t = np.linspace(0, 1, N, endpoint=False).reshape(-1, 1) + 1 / N
        self.path.extend(self.path[-1] + t * (point - self.path[-1]))
        self.indices.append(len(self.path) - 1)
        return self

    def x(self):
        """generate a sequence of x-values for plotting the k-path without distortion.

        Returns:
            list: x-values for all k-values along the path.
        """
        x = [0.0]
        for i in range(1, len(self.path)):
            x.append(x[-1] + np.linalg.norm(self.path[i] - self.path[i - 1]))
        return x

    def sym_x(self):
        """get the x values for plotting which correspond to the used symmetry points

        Returns:
            list: the plot x-values of the used symmetry points.
        """
        # slow serial generation
        l = 0.0
        sym_x = []
        if 0 in self.indices:
            sym_x.append(0)
        for i in range(1, len(self.path)):
            l += np.linalg.norm(self.path[i] - self.path[i - 1])
            if i in self.indices:
                sym_x.append(l)
        return sym_x

    def sym_x_names(self):
        """Get the names of the (symmetry) points that got used to construct this k-path.

        Returns:
            list: a list of the names ready for display
        """
        return self.names

    def plot(self, func, *args, band_offset=0, label_bands=None, ylim=None, **kwargs):
        """This function creates a bandstructure plot using matplotlib.

        Args:
            func (function): a function that takes an array of k samples of shape (N, dim) and returns an array of band values in shape (N, band_count).
            band_offset (int, optional): offset for the color cycle. Defaults to 0.
            label_bands (str, optional): either "left", "right" or "", None. Specifies where to annotate bandindex numbers. Defaults to None.
            ylim (tuple, optional): y-axis limits for the plot. Defaults to None.

        Returns:
            None | (ax1, ax2): if label_bands is used, the two axes are returned. Otherwise None.
        """
        if label_bands not in {None, "", "left", "right"}:
            raise ValueError("label_bands must be left or right")
        from matplotlib import pyplot as plt
        ibands = func(np.array(self, dtype=float))
        x_smpl = self.x()
        sym_x_smpl = self.sym_x()
        ax1 = plt.gca()
        if label_bands == "left":
            ax2 = plt.gca().twinx()  # plot to twinx (now the gca)
        plt.gca().set_prop_cycle(None)
        for _ in range(band_offset):
            # "prop_cycle" API is not yet stabilised and changed recently.
            # Adding empty plot commands works as a workaround.
            plt.plot([], [])
        for i in range(len(ibands[0])):
            plt.plot(x_smpl, ibands[:, i], *args, **kwargs)
        # plot vertical lines for all symmetry positions, that are not at the boundary
        for sym_x in sym_x_smpl[1:-1]:
            plt.axvline(sym_x, color="k", linestyle="dashed")
        plt.xticks(sym_x_smpl, self.names)
        plt.xlim(x_smpl[0], x_smpl[-1])
        if label_bands == "left" or label_bands == "right":
            if ylim is None:
                ylim = np.min(ibands) - 0.1, np.max(ibands) + 0.1
            if label_bands == "left":
                # switch back to left
                plt.sca(ax1)
            else:
                ax2 = plt.gca().twinx()  # plot to twinx (now the gca)
            # compute the textsize in axis units
            # (the results are a little unprecise, because rounding
            # isn't quite the optimal solution.. just the simplest)
            textsize = 8 * (ylim[1] - ylim[0]) / \
                plt.gcf().get_figheight() * 0.15
            # rounded to display precision to avoid label overlap
            edge_bands = np.round(
                ibands[-1 if label_bands == "right" else 0] / textsize, 1) * textsize
            y_ticks, inv = np.unique(edge_bands, return_inverse=True)
            y_bands = [[] for i in range(len(y_ticks))]
            for i, j in enumerate(inv):
                y_bands[j].append(i)
            # detect sorting of bands. If bands are sorted format the labels as 10-12 instead of 10,11,12
            y_labels = [""] * len(y_ticks)
            for j, bands in enumerate(y_bands):
                # turn bands into ranges
                ranges = [[bands[0], bands[0]]]  # both sided inclusive ranges
                for i in bands[1:]:
                    if ranges[-1][1] + 1 == i:
                        ranges[-1][1] += 1
                    else:
                        ranges.append([i, i])
                # now format ranges properly
                y_labels[j] = ",".join([str(r[0]) if r[0] == r[1] else (
                    f"{r[0]},{r[1]}" if r[0] == r[1] - 1 else f"{r[0]}-{r[1]}") for r in ranges])
            plt.gca().set_yticks(y_ticks, labels=y_labels, fontsize=8)
            ax1.set_ylim(*ylim)
            ax2.set_ylim(*ylim)
            return ax1, ax2
        elif ylim is not None:
            plt.ylim(*ylim)

    def plot_comparison(self, func1, func2, label_bands=None, ylim=None):
        """Plot two functions along the path.
        This calls `self.plot` for both functions.
        The second function will be drawn with dashes instead of lines.
        The matching of band colors is done automatically.

        Args:
            func1 (Callable[[arraylike(N_k, dim)], arraylike(N_k, N_B)]): The first function to plot with lines (the fit).
            func2 (Callable[[arraylike(N_k, dim)], arraylike(N_k, N_B)]): The second function to plot with dashes (the reference).
            label_bands (str, optional): either "left", "right" or "", None. Specifies where to annotate bandindex numbers for the the reference. Defaults to None.
            ylim (tuple, optional): y-axis limits for the plot. Defaults to None.
        """
        from matplotlib import pyplot as plt
        # figure out band_offset by evaluating the model at some point
        point = np.array([self.path[0]], dtype=float)
        assert np.shape(point) == (1, self.dim())
        gamma_bands_mod = func1(point)[0]
        gamma_bands_ref = func2(point)[0]
        nm = len(gamma_bands_mod)
        nr = len(gamma_bands_ref)
        fits = []
        for i in range(1 - nr, nm):
            a = gamma_bands_mod[max(i, 0):]
            b = gamma_bands_ref[max(-i, 0):]
            m = min(len(a), len(b))
            err = np.linalg.norm(a[:m] - b[:m])
            fits.append(err)
        # find the configuration with minimal error
        band_offset = np.argmin(fits) + (1 - nr)
        self.plot(func1, band_offset=max(0, -band_offset), label_bands=None, ylim=ylim)
        return self.plot(func2, '--', band_offset=max(0, band_offset), label_bands=label_bands, ylim=plt.gca().get_ylim())

    def dim(self) -> int:
        """Get the dimension of the k-space of this path.

        Returns:
            int: k-space dimension
        """
        return len(self.path[0])

    def __iter__(self) -> Iterator:
        return self.path.__iter__()

    def __len__(self) -> int:
        return self.path.__len__()

    def __getitem__(self, i) -> int:
        return self.path.__getitem__(i)

    # representation for quantum espresso
    def __str__(self):
        k_points = f"K_POINTS {{tpiba_b}}\n{len(self.path)}\n"
        for x, y, z in self.path:
            k_points = k_points + f"{x * 2} {y * 2} {z * 2} 1\n"
        return k_points


def interpolate(k_smpl, bands, sym: _sym.Symmetry = None, method="cubic", periodic=True) -> Callable:
    """Given band structure data and symmetry, return an 1D/2D/3D interpolator for the bandstructure.
    This works only for data, which can be arranged in a rectilinear grid after symmetrization.

    Args:
        k_smpl (arraylike(N_k, dim(k))): k-points for interpolation
        bands (arraylike(N_k, N_B)): (band) data for the k-points
        sym (Symmetry, optional): symmetry for realize_symmetric_data on the data to complete the grid. Defaults to None.
        method (str, optional): see `scipy.interpolate.RegularGridInterpolator(method=...)`. Defaults to "cubic".
        periodic (bool, optional): if True, the data will be wrapped in a [-0.5, 0.5] unit cell and the interpolator will work for any k. Defaults to True.

    Returns:
        scipy.interpolate.RegularGridInterpolator: interpolator for the data
    """
    import scipy.interpolate as interp
    assert np.shape(k_smpl)[0] == np.shape(bands)[0], f"number of k_smpl and bands needs to match, but was k_smpl: ({np.shape(k_smpl)}), bands: ({np.shape(bands)})"
    dim = len(k_smpl[0])
    if sym is not None:
        assert sym.dim() == dim, f"dimensions of the symmetry and the k_smpl data don't match, symmetry: {sym.dim()}, k_smpl: {dim}"
        k_smpl_orig = k_smpl
        k_smpl, bands = sym.realize_symmetric_data(
            k_smpl, bands, unit_cell=periodic, average_bad_duplicates=True)
    n = round(len(k_smpl)**(1 / dim))
    assert n**dim == len(k_smpl), "could reconstruct full square/cubic volume"

    # sort (again) by x, y, z compatible with reshape to meshgrid
    bands = np.array(bands)
    k_smpl = np.array(k_smpl)
    for i in reversed(range(dim)):
        # the round here is annoying as it can break at wrong places
        # + np.pi makes it less likely, but it can still happen
        reorder = np.argsort(np.round(k_smpl[:, i] + np.pi, 4), kind='stable')
        k_smpl = k_smpl[reorder]
        bands = bands[reorder]

    used_k_smpl = np.moveaxis(k_smpl.reshape((n,) * dim + (dim,)), -1, 0)
    used_bands = bands.reshape((n,) * dim + (-1,))
    if periodic:
        # make it work for k outside of the original k_smpl range by
        # 1. extending the range of the data using periodic points
        #   -> this is done more than necessary to accomodate for larger interpolation kernels like the cubic one.
        # 2. wrapping the function argument of the returned function using % 1.0
        #   -> see at the bottom
        for i in range(dim):
            vec = np.zeros(dim)
            vec[i] = 1.0
            vec = vec.reshape((dim,) + (1,) * dim)
            used_k_smpl = np.concatenate([used_k_smpl.take([-1], axis=i + 1, mode="wrap") - vec,
                                          used_k_smpl, used_k_smpl.take([0, 1], axis=i + 1, mode="wrap") + vec], axis=i + 1)
            used_bands = np.concatenate([used_bands.take([-1], axis=i, mode="wrap"), used_bands,
                                         used_bands.take([0, 1], axis=i, mode="wrap")], axis=i)

    if dim == 1:
        interp_f = interp.RegularGridInterpolator(used_k_smpl, used_bands, method=method)
    elif dim == 2:
        interp_f = interp.RegularGridInterpolator((used_k_smpl[0][:, 0], used_k_smpl[1][0, :]),
                                                  used_bands,
                                                  method=method)
    elif dim == 3:
        interp_f = interp.RegularGridInterpolator((used_k_smpl[0][:, 0, 0], used_k_smpl[1][0, :, 0], used_k_smpl[2][0, 0, :]),
                                                  used_bands,
                                                  method=method)
    else:
        raise ValueError(f"interpolation not implemented for dimension {dim}")
    if periodic:
        return lambda k: interp_f((np.asanyarray(k) + 0.5) % 1.0 - 0.5)
    return interp_f

# given band structure data and symmetry, return an interpolator for the bandstructure.
# The interpolator will return NaN if the point was not in the data.
# So this function doesn't really interpolate.
# This is useful for plotting the data along a path


def interpolate_unstructured(k_smpl, bands, sym: _sym.Symmetry = None, max_error=1e-3) -> Callable:
    from scipy.spatial import KDTree
    dim = len(k_smpl[0])
    if sym is not None:
        assert sym.dim() == dim, "dimensions of the symmetry and the data don't match"
        k_smpl, bands = sym.realize_symmetric_data(k_smpl, bands)
    # add a NaN value as last entry
    bands = np.append(bands, [[np.nan] * len(bands[0])], axis=0)
    kdtree = KDTree(k_smpl)

    def interp(k):
        _dist, index = kdtree.query(k, distance_upper_bound=max_error)
        # missing neighbors are indicated by infinite distance and index outside of range
        return np.reshape(bands[index], (len(k), -1))
    return interp


# define some default paths
SC_PATH = KPath('G').to('X').to('M').to('G').to('R').to('X')
FCC_PATH = KPath('G').to('X2').to('W').to('L').to('G').to('K')
BCC_PATH = KPath('G').to('H').to('P').to('G').to('N').to('P')
DIAMOND_PATH = KPath('L').to('G').to('X2').to('U').to('G').to('K')
SC_PATH_2D = KPath('G2d').to('X2d').to('M2d').to('G2d')
PATH_1D = KPath('G1d').to('X1d', N=100)

# TODO implement https://www.nature.com/articles/s41524-020-00383-7
# to get all the paths.

# internal function to compute the more complicated
# symmetry points from the face centered points
# hsp = high symmetry point (abbreviated because it is internal and used often)


def _hsp(a, b=None, c=None):
    res = [np.linalg.norm(a)**2]
    mat = [a]
    if b is not None:
        res.append(np.linalg.norm(b)**2)
        mat.append(b)
    if c is not None:
        res.append(np.linalg.norm(c)**2)
        mat.append(c)
    x = np.linalg.lstsq(mat, res, rcond=None)[0]
    return x


def hexagonal_points(r: float, h: float) -> dict:
    """Generate the symmetry points for a hexagonal reciprocal lattice.

    Args:
        r (float): distance between reciprocal lattice points in the hexagon.
        h (float): height of the hexagon in reciprocal space

    Returns:
        dict: A dictionary with the points G, A, K, H, M, L for this particular hexagonal model.
    """
    points = {}
    points['G'] = (np.zeros(3), 'Γ')
    points['A'] = (np.array([0, 0, h / 2]), 'A')
    points['K'] = (np.array([r / 3**.5, 0, 0]), 'K')
    points['H'] = (np.array([r / 3**.5, 0, h / 2]), 'H')
    points['M'] = (np.array([0.5 * r, 0.5 / 3**.5 * r, 0]), 'M')
    points['L'] = (np.array([0.5 * r, 0.5 / 3**.5 * r, h / 2]), 'L')
    return points


def tetragonal_bc_points(c: float) -> dict:
    """Generate the symmetry points for the tetragonal centered lattice (a, a, c).
    It is body centered in real space and face centered in reciprocal space.
    The length a is assumed to be 1 in real space.

    Args:
        c (float): _description_

    Returns:
        dict: _description_
    """
    if c == 1:
        global points
        return points  # cubic, is contained in the global points dict
    points = {}
    points['G'] = (np.zeros(3), 'Γ')
    points['X'] = (np.array([0.5, 0.5, 0]), 'X')
    if c < 1:
        points['M'] = (np.array([1., 0, 0]), 'M')
        points['N'] = (np.array([0.5, 0, 0.5 / c]), 'N')
        points['Z1'] = (_hsp(points['N'][0], points['M'][0]), '$Z_1$')
        points['Z'] = (
            np.array([0, 0, 2 * points['N'][0][2] - points['Z1'][0][2]]), 'Z')
        points['P'] = (np.array([0.5, 0.5, points['N'][0][2]]), 'P')
    else:
        # there are two label types "SC" and "BI", both are implemented here
        points['M'] = (np.array([0, 0, 1 / c]), 'M')
        points['M2'] = (np.array([0, 0, -1 / c]), '$M_2$')
        points['Z'] = (np.array([0, 0, 1 / c]), 'Z')
        points['N'] = (np.array([0.5, 0, 0.5 / c]), 'N')
        points['N0'] = (np.array([0.5, 0, -0.5 / c]), '$N_0$')
        points['N1'] = (np.array([0, 0.5, 0.5 / c]), '$N_1$')
        points['Sig'] = (_hsp(points['N'][0], points['N0'][0]), 'Σ')
        points['S0'] = (points['Sig'][0], '$S_0$')
        points['S'] = (_hsp(points['M'][0], points['N'][0]), 'S')
        points['Sig1'] = (points['S'][0], '$\\Sigma_1$')
        points['S2'] = (points['S'][0] * np.array([1, 1, -1]), '$S_2$')
        points['R'] = (_hsp(points['N'][0], points['N0'][0], points['X'][0]), 'R')
        points['Y'] = (points['R'][0], 'Y')
        points['P'] = (_hsp(points['N'][0], points['N1'][0], points['X'][0]), 'P')
        points['Y1'] = (_hsp(points['N'][0], points['N1'][0], points['M'][0]), '$Y_1$')
        points['GG'] = (points['Y1'][0], 'G')
        points['G0'] = (points['Y1'][0] * np.array([1, 1, -1]), '$G_0$')
    return points


def trigonal_points(alpha: float, mirror_x=False) -> dict:
    """Generate the symmetry points for a trigonal/rhombohedral reciprocal lattice of a real trigonal with lattice constant 1.
    There are two possible shapes of the Brilouin zone for alpha < 90° and alpha >= 90°.

    Args:
        alpha (float): Angle (in radians) of the trigonal structure in real space
        mirror_x (bool, optional): Align the mirror symmetry to the x-axis if True or the y-axis if False.
            The matrix created with `qespresso_interface.rhombohedral` will lead to the y-alignment, hence the default.
            Quantum Espresso usually uses the other convention (x-axis mirror symmetry, True). Defaults to False.

    Returns:
        dict: A dictionary with the symmetry points for the correct trigonal model.
    """
    # The points are taken from the QE documentation
    # Good paths are F, G, L, X, G, Z, P1, F, Q, L
    # and P, G, Q, Q1, P1, F, G, Z ???
    # sin theta = sqrt(2/3) sqrt(1-cos alpha) = sqrt(4/3) sin(alpha/2)
    # theta = np.arcsin((4/3)**.5*np.sin(alpha/2)) # works, but results are inprecise
    # c, s = np.cos(theta), np.sin(theta)
    c, s = ((1 + 2 * np.cos(alpha)) / 3)**.5, (4 / 3)**.5 * np.sin(alpha / 2)
    h = 1 / c
    points = {}
    points['G'] = (np.zeros(3), 'Γ')
    if alpha < 90 / 180 * np.pi:
        # l2 = np.linalg.norm([1/3**.5/s/2, 1/6/s, -h/6])**2
        l2 = 1 / 9 / s**2 + 1 / 36 / c**2
        points['Z'] = (np.array([0, 0, h / 2]), 'Z')
        points['L'] = (np.array([1 / 3**.5 / s / 2, -1 / 3 / s / 2, h / 6]), 'L')
        points['L1'] = (np.array([1 / 3**.5 / s / 2, 1 / 3 / s / 2, -h / 6]), '$L_1$')
        points['F'] = (np.array([1 / 3**.5 / s / 2, 1 / 3 / s / 2, h / 3]), 'F')
        # complicated, but easy enough for direct computation
        points['X'] = (np.array([l2 / (1 / 3**.5 / s / 2), 0, 0]), 'X')
        points['P'] = (_hsp(points['Z'][0], points['L'][0]), 'P')
        points['P1'] = (_hsp(points['Z'][0], points['F'][0]), '$P_1$')
        points['P2'] = (_hsp(points['F'][0], points['L1'][0]), '$P_2$')
        points['Q'] = (_hsp(points['F'][0], points['L'][0]), 'Q')
        points['B'] = (_hsp(points['Z'][0], points['F']
                       [0], points['L'][0]), 'B')
        points['B1'] = (_hsp(points['L1'][0], points['F']
                        [0], points['L'][0]), '$B_1$')
    else:
        assert alpha < 120 / 180 * np.pi
        l2 = (1 / s / 3)**2 + (h / 6)**2
        points['X'] = (np.array([1 / 3**.5 / s, 0, 0]), 'X')
        points['F'] = (np.array([1 / 3**.5 / s / 2, 1 / s / 2, 0]), 'F')
        points['F1'] = (np.array([-1 / 3**.5 / s / 2, 1 / s / 2, 0]), '$F_1$')
        points['L'] = (np.array([0, 1 / s / 3, h / 6]), 'L')
        points['Z'] = (np.array([1 / 3**.5 / s, 1 / s / 3, h / 6]), 'Z')
        points['Q'] = (np.array([0, 0, l2 / (h / 6)]), 'Q')
        points['Q1'] = (np.array([0, 2 / 3 / s, h / 3 - l2 / (h / 6)]), '$Q_1$')
        points['P1'] = (np.array([0, 2 / 3 / s, (h / 6 - l2 / (h / 6)) / 2]), '$P_1$')
        points['P'] = (
            np.array([1 / 3**.5 / s, 1 / s / 3, -(h / 6 - l2 / (h / 6)) / 2]), 'P')
    if not mirror_x:
        # switch xy of the points
        points = {k: (np.array(v[0][[1, 0, 2]]), v[1])
                  for k, v in points.items()}
    return points
