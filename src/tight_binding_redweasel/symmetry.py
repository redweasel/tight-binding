import numpy as np
from scipy.spatial import KDTree
from typing import Self, Callable, Tuple
import warnings

# My own symmetries.
# The following has a lot of it, but is missing a part which I deemed crucial for the performance.
# https://qsymm.readthedocs.io/en/latest/tutorial/bloch_generator.html


def pointcloud_distance(pointcloud1, pointcloud2):
    # TODO try using scipy.optimize.linear_sum_assignment
    pointcloud1 = np.asarray(pointcloud1)
    pointcloud1 = pointcloud1.reshape(pointcloud1.shape[0], -1)
    pointcloud2 = np.asarray(pointcloud2)
    pointcloud2 = pointcloud2.reshape(pointcloud2.shape[0], -1)
    # for each point in pointcloud1 find the closest in pointcloud2, add the distance, then remove that
    dist = 0.0
    for i, p1 in enumerate(pointcloud1):
        d = np.linalg.norm(p1 - pointcloud2, axis=-1)
        min_index = np.argmin(d.flat)
        dist += d.flat[min_index]
        d = np.delete(d.flat, min_index)
    return dist

# a function that returns a function that maps positions to (neighbor_index, is_mirrored)
# and raises a ValueError if the neighbor isn't found.


def neighbor_function(neighbors, err=1e-4) -> Callable[..., Tuple[int, bool]]:
    kdtree = KDTree(neighbors)

    def find(r):
        mirror = False
        _dist, j = kdtree.query(r, distance_upper_bound=err)
        if j < 0 or j >= len(neighbors):
            _dist, j = kdtree.query(-r, distance_upper_bound=err)
            mirror = True
            if not 0 <= j < len(neighbors):
                raise ValueError("neighbors don't match the symmetry")
        return j, mirror
    return find

# a function that returns a function that maps positions to (neighbor_index, is_mirrored)
# and returns None, None if the neighbor isn't found


def try_neighbor_function(neighbors, err=1e-4) -> Callable[..., Tuple[int, bool]]:
    kdtree = KDTree(neighbors)

    def find(r):
        mirror = False
        _dist, j = kdtree.query(r, distance_upper_bound=err)
        if j < 0 or j >= len(neighbors):
            _dist, j = kdtree.query(-r, distance_upper_bound=err)
            mirror = True
            if not 0 <= j < len(neighbors):
                return None, None
        return j, mirror
    return find


# class for symmetries. All symmetries (except inversion symmetry) are saved as unitary/orthogonal transformation matrices
# TODO do they need to be orthogonal? Why not SL(3) or -SL(3)? Those would also create closed sets of symmetries.
class Symmetry:
    # initialize symmetry from orthogonal matrices
    def __init__(self, S, inversion=False):
        self.S = np.asarray(S)
        self.inversion = inversion
        if np.linalg.norm(self.S[0] - np.eye(len(self.S[0]))) > 1e-7:
            raise ValueError(
                "The first entry in the symmetry list needs to be the identity element")
        if [s for s in self.S if abs(abs(np.linalg.det(s)) - 1) > 1e-7]:
            raise ValueError(
                "invalid matrix in parameter. All matrices need to have |det| = 1, otherwise the group is infinite")
        # check if S has inversion symmetry build in
        if pointcloud_distance(self.S, -self.S) < 1e-6:
            # if not inversion:
            #    print("found inversion symmetry")
            if len(S[0]) % 2 == 1:
                # keep only the S with positive determinants (works for real matrices in odd dimensions)
                self.inversion = True
                self.S = np.array([s for s in self.S if np.linalg.det(s) > 0])
            else:
                # self.S = np.array([s for s in self.S if next((x for x in np.ravel(s) if abs(x) > 1e-4), 1) > 0])
                # here inversion can be a problem, as it may be in the center, but it's not always a normal subgroup!
                # for now, disable the inversion symmetry reduction in even dimensions.
                pass
        # TODO add lists for broken symmetries
        # each symmetry(except for inversion) has a line or a plane on which it's unbroken
        # -> add information about that and add a way to get all the unbroken/broken symmetries for a point

    def copy(self) -> Self:
        """Deep copy of the symmetry group."""
        return Symmetry(np.array(self.S), self.inversion)

    def dim(self) -> int:
        """Dimension of the symmetry operations"""
        return len(self.S[0])

    def check(self) -> bool:
        """Check if the symmetry group is closed."""
        gen_sym = Symmetry.from_generator(self.S, self.inversion)
        return self == gen_sym

    def is_orthogonal(self) -> bool:
        """Check if the symmetry operations are orthogonal/unitary matrices"""
        return np.linalg.norm(np.einsum("nij,nkj->nik", self.S, np.conj(self.S)) - np.eye(self.dim())[None,...]) < 1e-6

    def transpose(self) -> Self:
        """Get the symmetry with all the symmetry operations transposed. That is the symmetry group for the dual lattice/vector space."""
        return Symmetry(np.array(np.swapaxes(self.S, -1, -2)), self.inversion)

    def __eq__(self, other: Self) -> bool:
        if self.dim() != other.dim():
            return False
        if self.inversion != other.inversion:
            return False
        if len(self) != len(other):
            return False
        return pointcloud_distance(self.S, other.S) < len(self.S) * 1e-7

    def __len__(self) -> int:
        """Cardinality of the group (including inversion symmetry if present!)"""
        return len(self.S) * (2 if self.inversion else 1)

    def get_symmetry_operations(self) -> np.ndarray:
        """Get all symmetry operations, including inversion symmetry.

        Returns:
            ndarray(len, dim, dim): All symmetry operation matrices
        """
        S = list(self.S)
        if self.inversion:
            S = S + list(-self.S)
        return np.array(S)

    # initialize the symmetry group from the lattice matrix A and the basis atoms b
    # b is a list of lists of basis positions with the meaning bpos = b[type][index]
    # TODO unfinished
    def from_lattice(A, b) -> Self:
        assert False
        # TODO find the integer valued symmetry operations in crystal space,
        # which get transformed to orthogonal symmetries using A
        # TODO find translational symmetries

        # simplify the problem using qr
        _, A = np.linalg.qr(A)
        A /= A[0, 0] # scaling doesn't change symmetries
        if len(A) == 2:
            S = [np.eye(2)]
            # check which rotation symmetry is given by just testing them
            # for that, generate the full pointcloud in a circle
            points = (A @ b.T).T
            r = np.max(np.linalg.norm(points, axis=-1))
            # TODO generate a full circle of these points
            for x in [4, 3]:
                angle = 2*np.pi/x
                rot = np.array(((np.cos(angle), np.sin(angle)),
                               (-np.sin(angle), np.cos(angle))))
                if pointcloud_distance(rot @ points, points) < 1e-7:
                    # add symmetry
                    S.append(rot)
                    if x == 3:
                        S.append(rot.T)
                    break
            # check if inversion symmetry is given
            # inversion is given, if the basis is equivalent to 1 - basis
            inv = True
            for b_ in b:
                if not pointcloud_distance(b_ % 1.0, (1 - b_) % 1.0) < 1e-7:
                    inv = False
            return Symmetry(S, inversion=inv)
        elif len(A) == 3:
            S = [np.eye(3)]
            # collect symmetries
            # TODO
            # check if inversion symmetry is given
            inv = pointcloud_distance(b % 1.0, (1 - b) % 1.0) < 1e-7
            return Symmetry(S, inversion=inv)
        else:
            raise NotImplementedError(
                f"not implemented for dimension {len(A)}")

    def transform(self, basis_transform) -> Self:
        """Apply a basis transformation to all symmetry operations.
        This is useful to convert reciprocal crystal space symmetries, which only contain integer entries,
        into reciprocal space symmetries using `transform(B)` where
        B is the matrix with the reciprocal lattice vectors as columns.

        `S' = B @ S @ inv(B)`

        NOTE: this does not change the instance it is called on.
        
        Args:
            basis_transform (matrix): Transform matrix B
        
        Returns:
            Self: The transformed symmetry group
        """
        assert np.shape(basis_transform) == (self.dim(),)*2, "basis_transform needs to be a square matrix with matching dimension"
        self.S = np.einsum("nij,mi,jk->nmk", self.S,
                           basis_transform, np.linalg.inv(basis_transform))
        return 

    def one_dim(inversion: bool) -> Self:
        """1D lattice symmetry (inversion or nothing)"""
        S = [np.array(((1,),))]
        return Symmetry(S, inversion)

    def two_dim_rotation(count: int) -> Self:
        """2D lattice rotation symmetry (n-fold rotation symmetry (includes inversion) where n is from {1, 2, 3, 4, 6})"""
        assert count in {1, 2, 3, 4, 6}
        inversion = False
        if count % 2 == 0:
            inversion = True
            count /= 2
        if count == 2:
            S = [np.eye(2), np.array(((0, 1), (-1, 0)))]
        elif count == 3:
            c = np.cos(2*np.pi/3)
            s = np.sin(2*np.pi/3)
            S = [np.eye(2), np.array(((c, s), (-s, c))),
                 np.array(((c, -s), (s, c)))]
        else:
            S = [np.eye(2)]
        return Symmetry(S, inversion)

    def from_generator(G, inversion: bool) -> Self:
        """Create the symmetry group from a set of unique generators.

        Args:
            G (arraylike(n, dim, dim)): n generating elements for the symmetry group.
            inversion (bool): If True, add inversion as a generating element

        Raises:
            ValueError: Raised if the group is not closed.

        Returns:
            Self: The symmetry generated by the generating set
        """
        N = len(G[0])
        G = np.array(G) + 0.0
        assert len(
            G) > 0, "Need at least one generator (e.g. the neutral element) to determine the dimension"
        d = len(G[0])
        # remove the neutral element
        G = [s for s in G if np.linalg.norm(s - np.eye(d)) > 1e-7]
        if len(G) == 0:
            return Symmetry(np.eye(d)[None, ...], inversion)
        if [s for s in G if abs(abs(np.linalg.det(s)) - 1) > 1e-10]:
            raise ValueError(
                "invalid matrix in parameter. All matrices need to have |det| = 1, otherwise the group is infinite")
        # TODO check uniqueness of G
        # add the neutral element back in
        S = [np.eye(d)] + G
        # exponentiate the group n times to form the full group
        for _ in range(1000):
            prev_len = len(S)
            S_new = np.array(S)
            S_new = np.reshape(S_new.reshape(-1, 1, N, N) @
                               S_new.reshape(1, -1, N, N), (-1, N, N))
            # find unique with a margin of error, assuming the S where unique before
            for s in S_new:
                is_new = True
                for s2 in S:
                    if np.linalg.norm(s - s2) < 1e-7:
                        is_new = False
                        break
                if is_new:
                    S.append(s)
            if len(S) >= 1000:
                raise ValueError("group size limitation to avoid endless loops")
            if len(S) <= prev_len:
                break
        return Symmetry(np.array(S), inversion)

    def none(dim=3) -> Self:
        """No symmetry for any dimension.

        Args:
            dim (int, optional): Dimension of the symmetry. Defaults to 3.

        Returns:
            Self: The empty symmetry group
        """
        return Symmetry([np.eye(dim)], False)

    def inv(dim=3) -> Self:
        """Inversion symmetry for any dimension.

        Args:
            dim (int, optional): Dimension of the symmetry. Defaults to 3.

        Returns:
            Self: The inversion symmetry group
        """
        return Symmetry([np.eye(dim)], True)

    def cubic(inversion: bool) -> Self:
        """Octahedral group https://en.wikipedia.org/wiki/Octahedral_symmetry

        Args:
            inversion (bool): If True, the inversion symmetry is included (O_h group instead of O).

        Returns:
            Self: The cubic symmetry group
        """
        return Symmetry.even_perm3() * Symmetry.mirror3(inversion)

    def perm3(inversion=False) -> Self:
        """The 3D symmetry created by permuting the axes.

        Args:
            inversion (bool, optional): If True, inversion symmetry is added. Defaults to False.

        Returns:
            Self: The 3D permutation group
        """
        S = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
             ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (0, 1, 0), (1, 0, 0))]
        return Symmetry(S, inversion=inversion)

    def even_perm3(inversion=False) -> Self:
        """The 3D permutation group with positive determinant.
        This group is generated by
        ```
        [[[ 0, 0, 1],
          [ 0,-1, 0],
          [ 1, 0, 0]],
         [[-1, 0, 0],
          [ 0, 0, 1],
          [ 0, 1, 0]]]
        ```

        Args:
            inversion (bool, optional): _description_. Defaults to False.

        Returns:
            Self: The positive 3D permutation symmetry group
        """
        S = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((0, 1, 0), (0, 0, -1), (-1, 0, 0)),
             ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
             ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
             ((0, 0, 1), (0, -1, 0), (1, 0, 0))]
        return Symmetry(S, inversion=inversion)

    def mirror3(inversion=False) -> Self:
        """3D Point reflections in all 3 planes (Klein four group V_4),
        or mirror symmetries for all axes if inversion is added.

        Args:
            inversion (bool, optional): If True, add inversion symmetry -> full xyz mirror symmetry. Defaults to False.

        Returns:
            Self: The klein four symmetry group (V_4) or the full xyz-mirror symmetry group
        """
        S = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
             ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
             ((-1, 0, 0), (0, 1, 0), (0, 0, -1))]
        return Symmetry(S, inversion=inversion)

    def mirror_x(inversion=False, dim=3) -> Self:
        """Mirror symmetry on x-axis.

        Args:
            inversion (bool, optional): If True, add inversion symmetry. Defaults to False.
            dim (int, optional): Dimension of the symmetry. Defaults to 3.

        Returns:
            Self: The x-mirror symmetry group with 2/4 elements.
        """
        S = [np.eye(dim), np.eye(dim)]
        S[1][0,0] = -1
        return Symmetry(S, inversion=inversion)

    def square() -> Self:
        """The symmetry of a 2D square.

        Returns:
            Self: The square symmetry group
        """
        S = [((0, 1), (-1, 0)),
             ((1, 0), (0, -1))]
        return Symmetry.from_generator(S, False)

    def monoclinic_x(inversion: bool) -> Self:
        """Monoclinic crystal (inversion symmetry + 180° rotation in yz)"""
        D = [np.eye(3), np.diag((1, -1, -1))]
        return Symmetry(D, inversion)

    def monoclinic_y(inversion: bool) -> Self:
        """Monoclinic crystal (inversion symmetry + 180° rotation in xz)"""
        D = [np.eye(3), np.diag((-1, 1, -1))]
        return Symmetry(D, inversion)

    def monoclinic_z(inversion: bool) -> Self:
        """Monoclinic crystal (inversion symmetry + 180° rotation in yz)"""
        D = [np.eye(3), np.diag((-1, -1, 1))]
        return Symmetry(D, inversion)

    def check_symmetry(self, foo: Callable, verbose=True) -> bool:
        """Check if a space dependent function satisfies the symmetry.
        
        Args:
            foo (Callable[[arraylike(dim)], arraylike(...)]): The function to be checked. Takes a spacial position and returns some kind of value.
            verbose (bool, optional): If True, a message is printed showing the standard deviation
                of the output for all symmetrically equivalent positions of a random position. Defaults to True.

        Returns:
            bool: True if the symmetry wasn't broken, otherwise False.
        """
        # random sample points
        r_smpl = np.random.random((50, len(self.S[0])))
        for r in r_smpl:
            values = []
            for s in self.S:
                values.append(foo(s @ r))
            if np.linalg.norm(np.std(values, axis=0)) > 1e-7:
                if verbose:
                    print("symmetry error")
                    print(np.std(values, axis=0))
                return False
        return True

    def realize_symmetric(self, k_smpl, unit_cell=False) -> Tuple[np.ndarray, np.ndarray]:
        """Takes a reduced set of k-points and computes
        the full set of k points that can be inferred using symmetry.
        The result is sorted, such that if it's a grid.
        It will be ordered appropriately for a reshape into a grid (useful for interpolation).
        
        If the input list of k-points contains symmetrically equivalent points,
        the output will have duplicates! There is a warning printed if this happens.

        Args:
            k_smpl (arraylike(N_k, dim(k))): List of points, that this symmetry is applied to.
                This is not allowed to contain symmetric equivalent points.
            unit_cell (bool, optional): If True, the results will be transformed into the unit cell [-1/2, 1/2[^dim. No duplicates are introduced by this.

        Returns:
            array(N'_k, dim(k)): List of points with the symmetric k-points added and sorted.
            array(N'_k): Index list of the original k-point for each new k-points
        """
        order = list(range(len(k_smpl)))
        full_k = list(k_smpl)
        for i, k in enumerate(k_smpl):
            # asymtotically slow* algorithm to find all symmetric points
            # (fine because it's never going above 48, so asymtotic behaviour is irrelevant)
            scale = np.linalg.norm(k)
            if scale == 0:
                continue
            used_k = [k]
            for inv in [1, -1] if self.inversion else [1]:
                for s in self.S:
                    k_ = s @ k * inv
                    # * this is why this algorithm is asymtotically slow
                    if np.min(np.linalg.norm(used_k - k_.reshape(1, -1), axis=-1)) < scale * 1e-6:
                        continue
                    used_k.append(k_)
                    full_k.append(k_)
                    order.append(i)
        order = np.asarray(order)
        full_k = np.asarray(full_k)
        if unit_cell:
            # make sure all k are inside the [-0.5, 0.5[ region
            # first restrict it to [0, 1[
            full_k %= 1.0
            full_k %= 0.9999999 # fix the ones which are basically 1.0 but with numerical error
            full_k = np.where(full_k <= 1e-6, 0.0, full_k) # make them exactly 0
            # shift to [-0.5, 0.5[
            full_k = np.where(full_k >= 0.4999999, full_k - 1.0, full_k)
            full_k = np.where(full_k <= 1e-6 - 0.5, -0.5, full_k) # make them exactly -0.5 if they exist.
        # sort by x, y, z compatible with reshape to meshgrid
        for i in range(len(k_smpl[0])):
            # the round here is annoying as it can break at wrong places
            # + np.pi makes it less likely, but it can still happen
            reorder = np.argsort(
                np.round(full_k[:, i] + np.pi, 4), kind='stable')
            full_k = full_k[reorder]
            order = order[reorder]
        # if there is duplicate k, then they would be next to one another at this point
        # -> quick check and warning if duplicates are generated, which refer to different source indices!
        unique_full_k = [full_k[0]]
        unique_order = [order[0]]
        bad_duplicates = False
        for i in range(1, len(full_k)):
            if np.linalg.norm(full_k[i-1] - full_k[i]) < 1e-6:
                # duplicate found
                if order[i-1] == order[i]:
                    # can be handled by ignoring it
                    continue
                else:
                    # can not be handled. Add to the list of unique_full_k -> output is not unique anymore
                    bad_duplicates = True
            unique_full_k.append(full_k[i])
            unique_order.append(order[i])
        # warn if duplicates fromm different k_smpl are found
        if bad_duplicates:
            warnings.warn("duplicate k-points generated in realize_symmetric")
        return np.asarray(unique_full_k), np.asarray(unique_order)

    def realize_symmetric_data(self, k_smpl, data, unit_cell=False) -> Tuple[np.ndarray, np.ndarray]:
        """Same as `realize_symmetric` but instead of returning the source indices,
        the information is immediately used to copy the given data for the new k-points.

        Args:
            k_smpl (arraylike(N_k, dim)): List of points, that this symmetry is applied to.
                This is not allowed to contain symmetric equivalent points. 
            data (arraylike(N_k, ...)): Data to be copied along with the k-points.
            unit_cell (bool, optional): If True, the results will be transformed into the unit cell [-1/2, 1/2[^dim. No duplicates are introduced by this.

        Returns:
            Tuple[np.ndarray, np.ndarray]: new k-points, data at the new k-points
        """
        full_k, order = self.realize_symmetric(k_smpl, unit_cell=unit_cell)
        return full_k, np.array(np.asarray(data)[order])

    def reduce_symmetric_data(self, k_smpl, data, checked=True) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce symmetric data to only include one representant for each k equivalence class.
        NOTE: the result can be unstable but usually it's good.

        Args:
            k_smpl (arraylike(N_k, dim)): The spacial positions of the data
            data (arraylike(N_k, ...)): The data for each k-point
            checked (bool, optional): If True, the data gets checked for symmetry and an exception is raises if it is not symmetric. This costs almost no extra time. Defaults to True.

        Raises:
            ValueError: Raised if the data didn't respect the symmetry.

        Returns:
            array(N'_k, dim(k)): List of points with the symmetric k-points removed.
            array(N'_k, ...): List of data values for the reduced points.
        """
        # same sort as in realize_symmetric_data
        k_smpl = np.asarray(k_smpl)
        data = np.asarray(data)
        for i in range(len(k_smpl[0])):
            order = np.argsort(
                np.round(k_smpl[:, i] + np.pi, 4), kind='stable')
            k_smpl = k_smpl[order]
            data = data[order]
        reduced_k = np.asarray(k_smpl)
        reduced = np.asarray(data)
        # sort by length of k, since all symmetry operations keep length equal
        assert self.is_orthogonal(), "The algorithm here only works for orthogonal symmetry operations"
        # this stable works really well
        order = np.argsort(np.round(np.linalg.norm(
            reduced_k, axis=-1)**2, 2), kind='stable')
        reduced_k = list(reduced_k[order])
        reduced = list(reduced[order])
        i = 0
        while i < len(reduced_k):
            # good* algorithm to find all symmetric points
            k = reduced_k[i]
            value = reduced[i]
            i += 1
            scale = np.linalg.norm(k)
            if scale == 0:
                continue
            for inv in [1, -1] if self.inversion else [1]:
                for s in self.S:
                    k_ = s @ k * inv
                    if np.linalg.norm(k - k_) < scale * 1e-6:
                        continue
                    # find all instances of k_ in reduced_k and delete them
                    j = i
                    while j < len(reduced_k):
                        if np.linalg.norm(reduced_k[j]) > scale * 1.0001:
                            break  # * this is what makes this a good algorithm
                        if np.linalg.norm(reduced_k[j] - k_) < scale * 1e-6:
                            if checked and np.linalg.norm(reduced[j] - value) > 1e-7:
                                raise ValueError(
                                    "Symmetry check found asymmetric data.")
                            del reduced[j]
                            del reduced_k[j]
                        else:
                            j += 1
        return np.array(reduced_k), np.array(reduced)

    def complete_neighbors(self, neighbors, return_order=False):
        """Make a complete set of neighbors for this symmetry based on the given neighbor positions.

        Args:
            neighbors (arraylike(N_R, dim)): A list of real lattice points.
            return_order (bool, optional): If True, the source index for each of the result indices is returned as well. Defaults to False.

        Returns:
            list: List of real lattice positions, but reduced by inversion symmetry. The 0 position is at index 0.
            list: Returned if return_order=True, List of source indices for each of the returned positions.
        """
        # this is done with the transposed symmetries, as the whole symmetry is supposed to be for k-space.
        neighbors, order = self.transpose().realize_symmetric(neighbors)
        # deduplicate (but keep the original ordering if there is no duplicates)
        new_neighbors = []
        used_neighbors = set()
        for n, orig_index in zip(neighbors, order):
            if tuple(n) not in used_neighbors:  # works only for exact matches
                new_neighbors.append((n, orig_index))
                used_neighbors.add(tuple(n))
        # remove negative versions
        neighbors = [(n, orig_index) for n, orig_index in new_neighbors if next(
            (x for x in n if abs(x) > 1e-7), 1) > 0]
        neighbors = sorted(neighbors, key=lambda ni: np.linalg.norm(ni[0]))
        order = [orig_index for _, orig_index in neighbors]
        neighbors = [n for n, _ in neighbors]
        if return_order:
            return neighbors, order
        return neighbors

    def check_neighbors(self, neighbors):
        """Check 3D real lattice positions, if they fit the symmetry.

        Args:
            neighbors (arraylike(N_R, 3)): The real lattice positions to be checked.

        Raises:
            ValueError: Raised if the given positions don't fit the symmetry.
        """
        assert self.dim() == 3
        # only half of the neighbor terms are present, but the symmetry is important
        neighbors = np.asarray(neighbors)
        neighbors = neighbors[np.argsort(np.linalg.norm(neighbors, axis=-1))]
        count = [0]*len(neighbors)
        for i, r in enumerate(neighbors):
            for s in zip(self.S):
                # using transposed symmetry operations, as the whole symmetry is meant to be in k-space
                r_ = s.T @ r
                # here only the neighbors with an index similar to i need to be checked
                # TODO replace this search by the KD-Trees
                start = max(i - len(self.S) * 3, 0)
                end = min(i + len(self.S) * 3, len(neighbors))
                j = start + np.argmin(np.minimum(np.linalg.norm(neighbors[start:end] - r_[None, :], axis=-1),
                                                 np.linalg.norm(neighbors[start:end] + r_[None, :], axis=-1)))
                count[j] += 1
        for c in count:
            if c != len(self.S):
                raise ValueError(
                    "neighbors need to be choosen to fit the symmetry, however counting occurences has found the numbers " + str(count))

    def k_weight(self, k_smpl) -> np.ndarray:
        """Calculate the weight of a k-point in a 1-periodic lattice with this symmetry.
        The weight is the fraction of space angle around the point, which is owned by the point.
        E.g. the weight of the point 0 is always 1/len(self).

        Args:
            k_smpl (arraylike(N_k, dim)): A list of k-points for which to compute the weight. Each k-point is considered unique.

        Returns:
            ndarray(N_k): List of weights
        """
        weights = np.zeros(len(k_smpl), dtype=np.int32)

        def pingpong_distance(x):
            return 0.5 - np.abs(x % 1.0 - 0.5)
        # NOTE this could be done faster using the right set of generators
        for sign in [-1, 1] if self.inversion else [1]:
            for s in self.S:
                weights += pingpong_distance(np.einsum("ij,nj->ni",
                                             sign*s - np.eye(len(s)), k_smpl)).sum(-1) < 1e-7
        return 1 / weights

    def r_class_size(self, k_smpl):
        """Number of points in the symmetry equivalence class for each k-point without periodicity.

        Args:
            k_smpl (arraylike(N_k, dim)): A list of k-points for which to compute the size of the equivalence class.

        Returns:
            ndarray(N_k): An integer (the format is float) for each point.
        """
        # instead of generating the symmetric points, check how many symmetries fail to produce new points.
        # NOTE every symmetric point can be generated from using the application of just one symmetry operation,
        # therefore the number of points is the number of symmetries divided by the symmetries which leave the initial point invariant.
        weights = np.zeros(len(k_smpl), dtype=np.int32)
        for sign in [-1, 1] if self.inversion else [1]:
            for s in self.S:
                weights += np.linalg.norm(np.einsum("ij,nj->ni",
                                          sign*s - np.eye(len(s)), k_smpl), axis=-1) < 1e-7
        return len(self) / weights

    def find_classes(self, points):
        """Find the groups of symmetrically equivalent points.

        Args:
            points (arraylike(N, dim)): The point array to find equivalent sets in.

        Returns:
            dict: {representative_index: { all indices of the equivalence group } }
        """
        points = np.asarray(points)
        classes = {}  # {representative_index: { index }}
        covered = set()
        # build the symmetry graph for the neighbor list
        for i, r in enumerate(points):
            rep_class = set()
            for s in self.S:
                r_ = s @ r
                dist = np.linalg.norm(points - r_[None, :], axis=-1)
                j = np.argmin(dist)
                if dist[j] <= 1e-7:
                    rep_class.add(j)
            if i not in covered:
                classes[i] = rep_class
            covered = covered.union(rep_class)
        return classes

    def symmetrize(self, tensor):
        """Symmetrize a tensor accoding to this symmetry using the group mean.
        This is a linear operation, which is a projection.

        Args:
            tensor (arraylike(dim, dim)): The input tensor to be symmetrized.

        Returns:
            ndarray(dim, dim): The symmetrized tensor
        """
        orig = np.array(tensor) * 1.0
        res = np.zeros_like(orig)
        for s in self.S:
            res += np.linalg.inv(s) @ orig @ s
        res /= len(self.S)
        return res

    def equivalence_classes(self, equiv_relation: Callable[[np.ndarray, np.ndarray], bool]) -> list:
        """Compute equivalence classes with a given equivalence relation.
        This function realizes the inversion symmetry for the result.

        Args:
            equiv_relation (Callable[[ndarray(dim, dim), ndarray(dim, dim)], bool]): _description_

        Returns:
            list: List of equivalence classes, which are represented as lists with the symmetry operations.
        """
        rem = list(self.S)
        if self.inversion:
            rem = rem + list(-self.S)
        classes = []
        while rem:
            s = rem.pop()
            # find equivalent elements
            unique = [s]
            for i in range(len(rem)-1, -1, -1):
                s2 = rem[i]
                if equiv_relation(s, s2):
                    unique.append(s2)
                    del rem[i]
            classes.append(unique)
        return classes

    def conjugacy_classes(self) -> list:
        """Compute the conjugacy classes of the symmetry group.

        Returns:
            list: List of equivalence classes, which are represented as lists with the symmetry operations.
        """
        def conjugated(a, b):
            cc = np.einsum("nij,jm,nmk->nik", self.S, a, np.linalg.inv(self.S))
            for c in cc:
                if np.linalg.norm(c - b) < 1e-7:
                    return True
            return False
        return self.equivalence_classes(conjugated)

    def __mul__(self, other: Self) -> Self:
        """Combine two symmetries by finding the smallest symmetry group, that is generated by them."""
        assert self.dim() == other.dim()
        return Symmetry.from_generator(list(self.S) + list(other.S), inversion=(self.inversion or other.inversion))

    # find the factor group or raise a ValueError
    def __truediv__(self, rhs: Self) -> Self:
        assert self.dim() == rhs.dim()
        d = self.dim()
        assert (self.inversion or not rhs.inversion) and len(self.S) % len(
            rhs.S) == 0, "The righthand side needs to be a subgroup"
        # TODO check subgroup

        def left(a, b):
            # to check a*rhs1=b*rhs2, check a^{-1}*b in rhs
            c = np.linalg.inv(a) @ b
            for r in rhs.S:
                if np.linalg.norm(r - c) < 1e-7:
                    return True
            return False
        classes = self.equivalence_classes(left)
        assert len(classes) == len(
            self) // len(rhs), "The number of left equivalence classes doesn't match the assumption of a normal subgroup"
        # TODO check the classes for the group properties (if they have it, rhs is a normal subgroup)
        # reduce the classes, while keeping the group property.
        # 1. find the class with the identity and reduce it to just that
        # 2. for each class, find the order and remove all elements,
        #    which do not result in the identity if taken to that power
        while True:
            for i in range(len(classes)):
                # takes increasing powers for each element of the class until one reaches the identity.
                # if more than one element reaches the identity at the same time, keep all of them in this step.
                base = np.array(classes[i])
                powers = np.zeros_like(base) + np.eye(d)[None, :, :]
                for _ in range(len(self.S)+1):  # no order bigger than the group size
                    powers = powers @ base
                    classes[i] = np.array([a for j, a in enumerate(
                        base) if np.linalg.norm(powers[j] - np.eye(d)) < 1e-7])
                    if len(classes[i]) > 0:
                        break
            # the class with the identity now has size 1, all the others can have a different size
            # -> take a class with size > 1, select an element and multiply it onto all classes, then repeat
            # -> this creates a second class of size 1
            # -> all classes will have size 1 in the end and obey the group structure
            big_classes = [c for c in classes if len(c) > 1]
            if len(big_classes) == 0:
                break
            # else:
            #    print("rerun with", len(big_classes), "big classes remaining")
            # guarantees that this will not be an endless loop
            selected = np.linalg.inv(big_classes[0][0])
            new_classes = []
            for c in classes:
                c_unique = []
                for s in c:
                    s = selected @ s
                    is_new = True
                    for u in c_unique:
                        if np.linalg.norm(u - s) < 1e-7:
                            is_new = False
                            break
                    if is_new:
                        c_unique.append(s)
                new_classes.append(c_unique)
            classes = new_classes
        # element to permute the identity to the first element
        id_inv = np.linalg.inv(classes[0][0])
        classes = np.array([c[0] @ id_inv for c in classes])
        sym = Symmetry(classes, inversion=self.inversion and rhs.inversion)
        if not sym.check():
            raise ValueError("The righthand side is no normal subgroup")
        return sym
