from typing import Iterator
import numpy as np
import _collections_abc

points = {}
points['G'] = np.array([[0, 0, 0]])
points['O'] = np.array([[0, 0, 0]])
# face centered cubic in tpiba_b units:
points['X2'] = np.array([[0, 1, 0]]) # Delta line
points['K'] = np.array([[3/4, 3/4, 0]]) # Sigma line
points['L'] = np.array([[1/2, 1/2, 1/2]]) # Lambda line
points['W'] = np.array([[1/2, 1, 0]]) # between X and K
points['U'] = np.array([[1/4, 1, 1/4]]) # between X and L
# simple cubic points:
points['X'] = 0.5*np.array([[0, 1, 0]]) # Delta line
points['M'] = 0.5*np.array([[1, 1, 0]]) # Sigma line
points['R'] = 0.5*np.array([[1, 1, 1]]) # Lambda line
# body centered points:
points['H'] = np.array([[1, 0, 0]]) # Delta line
points['N'] = np.array([[1/2, 1/2, 0]]) # Sigma line
points['P'] = np.array([[1/2, 1/2, 1/2]]) # Lambda line
# 2D square symmetry points:
points['G2d'] = 0.5*np.array([[0, 0]])
points['X2d'] = 0.5*np.array([[0, 1]]) # Delta line
points['M2d'] = 0.5*np.array([[1, 1]]) # Sigma line

class KPath(_collections_abc.Sequence):
    def __init__(self, start):
        """create new path in k-space.
        The dimension of the k-space is inferred from the start point.

        Args:
            start (str, array-like): Either a pointname from `kpath.points.keys()` like "G" or an explicit point in k-space like (0, 0, 0).
        """
        # TODO accept a list of points instead of start
        self.path = [(points[start] if start in points else np.array(start)).reshape(-1)]
        self.indices = [0] # symmetry point indices
        self.names = [str(start).replace("G", "Γ")]
    
    def to(self, point, N=32):
        """add a new (symmetry) waypoint to the end of the path.

        Args:
            point (str, array-like): Either a pointname from `kpath.points.keys()` or an explicit point in k-space.
            N (int, optional): The number of interpolated points from the last point in the path. Defaults to 32.

        Returns:
            Self: self
        """
        self.names.append(str(point).replace("G", "Γ"))
        point = points[point] if point in points else np.array(point).reshape(1,-1)
        t = np.linspace(0, 1, N, endpoint=False).reshape(-1, 1) + 1/N
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
            x.append(x[-1] + np.linalg.norm(self.path[i] - self.path[i-1]))
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
            l += np.linalg.norm(self.path[i] - self.path[i-1])
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

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if label_bands not in {None, "", "left", "right"}:
            raise ValueError("label_bands must be left or right")
        from matplotlib import pyplot as plt
        ibands = func(np.array(self))
        x_smpl = self.x()
        sym_x_smpl = self.sym_x()
        ax1 = plt.gca()
        if label_bands == "left":
            ax2 = plt.gca().twinx() # plot to twinx (now the gca)
        plt.gca().set_prop_cycle(None)
        for _ in range(band_offset):
            next(plt.gca()._get_lines.prop_cycler)
        for i in range(len(ibands[0])):
            plt.plot(x_smpl, ibands[:,i], *args, **kwargs)
        for sym_x in sym_x_smpl:
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
                ax2 = plt.gca().twinx() # plot to twinx (now the gca)
            edge_bands = np.round(ibands[-1 if label_bands == "right" else 0], 1) # rounded to display precision to avoid label overlap
            y_ticks, inv = np.unique(edge_bands, return_inverse=True)
            y_bands = [[] for i in range(len(y_ticks))]
            for i, j in enumerate(inv):
                y_bands[j].append(i)
            # detect sorting of bands. If bands are sorted format the labels as 10-12 instead of 10,11,12
            y_labels = [""]*len(y_ticks)
            for j, bands in enumerate(y_bands):
                # turn bands into ranges
                ranges = [[bands[0], bands[0]]] # both sided inclusive ranges
                for i in bands[1:]:
                    if ranges[-1][1] + 1 == i:
                        ranges[-1][1] += 1
                    else:
                        ranges.append([i, i])
                # now format ranges properly
                y_labels[j] = ",".join([str(r[0]) if r[0] == r[1] else (f"{r[0]},{r[1]}" if r[0] == r[1] - 1 else f"{r[0]}-{r[1]}") for r in ranges])
            plt.gca().set_yticks(y_ticks, labels=y_labels, fontsize=8)
            ax1.set_ylim(*ylim)
            ax2.set_ylim(*ylim)
            return ax1, ax2
        elif ylim != None:
            plt.ylim(*ylim)

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
            k_points = k_points + f"{x*2} {y*2} {z*2} 1\n"
        return k_points

# given band structure data and (non hexagonal) symmetry, return an interpolator for the bandstructure.
# This works only for data, which is arranged in a rectilinear grid after symmetrization.
# sym needs to be a Symmetry instance from symmetry.py
def interpolate(k_smpl, bands, sym, method="cubic"):
    import scipy.interpolate as interp
    k_smpl, bands = sym.realize_symmetric_data(k_smpl, bands)
    dim = sym.dim()
    assert len(k_smpl[0]) == dim, "dimensions of the symmetry and the data don't match"
    n = round(len(k_smpl)**(1/dim))
    assert n**dim == len(k_smpl), "could reconstruct full square/cubic volume"
    
    # sort (again) by x, y, z compatible with reshape to meshgrid
    bands = np.array(bands)
    k_smpl = np.array(k_smpl)
    for i in range(dim):
        # the round here is annoying as it can break at wrong places
        # + np.pi makes it less likely, but it can still happen
        reorder = np.argsort(np.round(k_smpl[:,i] + np.pi, 4), kind='stable')
        k_smpl = k_smpl[reorder]
        bands = bands[reorder]
    
    used_k_smpl = k_smpl.reshape((n,)*dim + (dim,)).T
    if dim == 1:
        return interp.RegularGridInterpolator(used_k_smpl,
                                                bands.reshape((n,)*dim + (-1,)),
                                                method=method)
    if dim == 2:
        return interp.RegularGridInterpolator((used_k_smpl[0][:,0], used_k_smpl[1][0,:]),
                                                bands.reshape((n,)*dim + (-1,)),
                                                method=method)
    if dim == 3:
        return interp.RegularGridInterpolator((used_k_smpl[0][:,0,0], used_k_smpl[1][0,:,0], used_k_smpl[2][0,0,:]),
                                                bands.reshape((n,)*dim + (-1,)),
                                                method=method)

# given band structure data and symmetry, return an interpolator for the bandstructure.
# The interpolator will return NaN if the point was not in the data.
# So this function doesn't really interpolate.
# This is useful for plotting the data along a path
def interpolate_unstructured(k_smpl, bands, sym, max_error=1e-3):
    from scipy.spatial import KDTree
    k_smpl, bands = sym.realize_symmetric_data(k_smpl, bands)
    # add a NaN value as last entry
    bands = np.append(bands, [[np.nan]*len(bands[0])], axis=0)
    kdtree = KDTree(k_smpl)
    def interp(k):
        _dist, index = kdtree.query(k, distance_upper_bound=max_error)
        # missing neighbors are indicated by infinite distance and index outside of range
        return np.reshape(bands[index], (len(k), -1))
    return interp

# TODO add tests
