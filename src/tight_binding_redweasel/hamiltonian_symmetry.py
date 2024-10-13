import numpy as np
from typing import Self, Callable
from . import symmetry as _sym
from . import unitary_representations as _urep

# This file contains a structure very similar to
# unitary representations, however it only works for direct products
# of irrep unitary representations and additionally has information
# about the position of the represented atomic orbitals in the unit cell.
# This way it can represent a more general set of symmetries

# TODO handle projective symmetries! -> correctly symmetrize the diamond structure!

class HamiltonianSymmetry:
    """
    This class represents a symmtery for a hermitian operator that
    is dependent on a periodic spacial parameter.
    The hermitian operator is represented as a fourier series:

    `H(k) = sum_R(H_R exp(2πikR))`

    The symmetry is based on a fixed spacial symmetry group and represents a
    direct sum of unitary representations of that symmetry group.
    There is an additional phase added to each individual unitary representation.
    The total symmetry can be expressed as:

    `H((S^T)^{-1} k) = U_S H(k) U^+_S exp(2πik(1+S)(r_1-r_2))`

    In solid state physics, the most important example for this
    is the **system hamiltonian**, which gives the class its name.
    """

    def __init__(self, sym: _sym.Symmetry):
        """
        Initialize a new empty HamiltonianSymmetry based on a spacial symmetry group.
        The corresponding hermitian operator starts out as 0-dimensional
        and get extended to n-dim. by appending new unitary representations.

        Args:
            sym (Symmetry): the spacial symmetry group (call sym.dual() if needed to construct this from the reciprocal symmetry group)
        """
        # translation is actually not important for the symmetry.
        # It just plays a role in deciding which orbitals are equal,
        # but I let the user do that with the names of the orbitals.
        self.sym = sym.ignore_translation()
        self.U = [] # unitary representations
        self.pos = [] # e.g. [[0,0,0], [1/4,1/4,1/4]] for k-dependence of symmetry
        self.names = [] # e.g. ["C", "C"] for exchange symmetriesCu
        # translation symmetries (for S=1) can be handled separately (TODO)
        self.translation = np.ones((sym.dim(), 0))
    
    def dim(self):
        """
        Returns:
            int: dimension (also called degree) of the full unitary representation/symmetry
        """
        # sum of the representations, used in the direct sum
        return sum((u.dim() for u in self.U))
    
    def copy(self) -> Self:
        u_repr = HamiltonianSymmetry(self.sym.copy(), 1)
        u_repr.inv = np.array(self.inv)
        u_repr.U = [u.copy() for u in self.U]
        u_repr.pos = [x.copy() for x in self.pos]
        u_repr.names = self.names.copy()
        return u_repr
    
    def __len__(self):
        return len(self.sym)
    
    def append(self, urepr: _urep.UnitaryRepresentation, pos, name=""):
        """Append a unitary representation to the full symmetry.
        Additional to the representation, there is a position,
        which introduces a phase after each symmetry operation.
        See class description for more information.

        Args:
            urepr (UnitaryRepresentation): a unitary representation to be added as a direct sum to the existing representation. A copy is done before it is appended.
                Only unitary representation with skalar behavior under inversion are allowed. This is always the case for irreps.
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        if len(pos) != self.sym.dim():
            raise ValueError(f"position (dimension {len(pos)}) needs to match the dimension of the symmetry ({self.sym.dim()})")
        if not (urepr.inv_split == 0 or urepr.inv_split == urepr.dim()):
            raise ValueError("only unitary representations with uniform inversion behavior are allowed. Otherwise they are reducible and can be added separately.")
        urepr = urepr.copy()
        urepr.match_sym(self.sym)
        self.U.append(urepr)
        self.pos.append(np.array(pos))
        self.names.append(name)
    
    def append_s(self, pos, name: str):
        """Append an automatically created s-orbital.
        This is a notion from solid state physics that means the representation
        is a one dimensional representation that does nothing (identity).
        This works with any symmetry group.

        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        self.append(_urep.UnitaryRepresentation(self.sym, 1), pos, name)
    
    def append_diamond_s(self, pos, name: str):
        """Append 2 automatically created s-orbitals, which are considered to get swapped for 90° rotations and inversion.
        
        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        # TODO figure out the positions...
        s = np.einsum("nij,j->n", self.sym.S, [1,1,1])
        swap = np.abs(np.abs(s - 1) - 2) > 1e-5
        if self.sym.inversion:
            # Here the swap also needs to happen on inversion.
            # Therefore the basis needs to be changed, as only
            # diagonal operations are allowed on inversion.
            # mixed inversion symmetry is not allowed here, so this is split up
            urepr = _urep.UnitaryRepresentation(self.sym, 1)
            urepr.inv_split = 0 # -1 on inversion
            self.append(urepr, pos, name+"0")
            urepr.inv_split = 1
            urepr.U[swap,0,0] *= -1 # -1 on rotation
            self.append(urepr, pos, name+"1")
        else:
            urepr = _urep.UnitaryRepresentation(self.sym, 2)
            urepr.U = np.where(swap[:,None,None], np.roll(urepr.U, 1, axis=1), urepr.U)
            self.append(urepr, pos, name)

    def append_p(self, pos, name: str):
        """Append automatically created p-orbitals in axis order. (xyz...)
        This is a notion from solid state physics that means the representation
        is the same matrix as the symmetry operation itself.
        This works with any symmetry group.

        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        urepr = _urep.UnitaryRepresentation(self.sym, self.sym.dim())
        urepr.U = np.array(self.sym.S)
        urepr.inv_split = 0
        self.append(urepr, pos, name)

    def append_diamond_p(self, pos, name: str):
        """Append 2 automatically created p-orbitals, which are considered to get swapped for 90° rotations and inversion.
        
        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        # TODO figure out the positions...
        urepr = _urep.UnitaryRepresentation(self.sym, self.sym.dim())
        urepr.U = np.array(self.sym.S)
        urepr.inv_split = 0 # -1 on inversion
        s = np.einsum("nij,j->n", self.sym.S, [1,1,1])
        swap = np.abs(np.abs(s - 1) - 2) > 1e-5
        if self.sym.inversion:
            # see append_diamond_s
            # mixed inversion symmetry is not allowed here, so this is split up
            urepr.inv_split = 3 #  1 on inversion
            self.append(urepr, pos, name+"0")
            urepr.inv_split = 0 # -1 on inversion
            urepr.U[swap] *= -1 # -1 on rotation
            self.append(urepr, pos, name+"1")
        else:
            urepr = urepr * 2
            urepr.permute([0, 2, 4, 1, 3, 5]) # normal order
            urepr.U = np.where(swap[:,None,None], urepr.U[:,[3,4,5,0,1,2]], urepr.U)
            self.append(urepr, pos, name)
    
    def append_d2(self, pos, name: str):
        """Append d_{z^2}, d_{x^2-y^2} orbitals in that order.
        This is a notion from solid state physics that comes from the
        symmetry of the real spherical harmonics with the same name.
        This currently only works with the cubic symmetry group.

        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        # TODO make this work for all symmetries
        self.append(_urep.UnitaryRepresentation.d3(False, inversion=self.sym.inversion), pos, name)

    def append_d3(self, pos, name: str):
        """Append d_yz, d_xz, d_xy orbitals in that order.
        This is a notion from solid state physics that comes from the
        symmetry of the real spherical harmonics with the same name.
        The symmetry of these function is same as that of p-orbitals,
        but without the negative sign on inversion.
        As such this works for any symmetry group.

        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        urepr = _urep.UnitaryRepresentation(self.sym, self.sym.dim())
        urepr.U = self.sym.S / np.linalg.det(self.sym.S)[:,None,None]
        urepr.inv_split = urepr.dim() # don't use -1 on inversion
        self.append(urepr, pos, name)

    def append_diamond_d3(self, pos, name: str):
        """Append 2 automatically created d-orbitals (yz, xz, xy), which are considered to get swapped for 90° rotations and inversion.
        
        Args:
            pos (arraylike): the position used for the added phase.
            name (str, optional): a name that is just used for annotation. Defaults to empty string.
        """
        # TODO figure out the positions...
        urepr = _urep.UnitaryRepresentation(self.sym, self.sym.dim())
        urepr.U = np.array(self.sym.S)
        urepr.inv_split = 3 # 1 on inversion
        s = np.einsum("nij,j->n", self.sym.S, [1,1,1])
        swap = np.abs(np.abs(s - 1) - 2) > 1e-5
        if self.sym.inversion:
            # see append_diamond_s
            # mixed inversion symmetry is not allowed here, so this is split up
            urepr.inv_split = 0 # -1 on inversion
            self.append(urepr, pos, name+"0")
            urepr.inv_split = 3 #  1 on inversion
            urepr.U[swap] *= -1 # -1 on rotation
            self.append(urepr, pos, name+"1")
        else:
            urepr = urepr * 2
            urepr.permute([0, 2, 4, 1, 3, 5]) # normal order
            urepr.U = np.where(swap[:,None,None], urepr.U[:,[3,4,5,0,1,2]], urepr.U)
            self.append(urepr, pos, name)
    
    def get_band_name(self, index):
        for u, name in zip(self.U, self.names):
            if index < u.dim():
                if u.dim() == 1:
                    return name
                else:
                    return f"{name}{index}"
            index -= u.dim()
        raise IndexError(f"index {index} is out of bounds")

    def apply(self, k, hamiltonian, s_index, inversion=False):
        """Apply the unitary transformation for one symmetry operation
        at the given k-point to the hermitian operator to get

        `H((S^T)^{-1} k) = U_S H(k) U^+_S exp(2πik(1+S)(r_1-r_2))`

        Args:
            k (arraylike): The spacial point
            hamiltonian (arraylike(dim, dim)): The hermitian operator H(k) on which the operation is performed.
            s_index (int): The index of the symmetry operation in the associated symmetry.
            inversion (bool, optional): Whether to add an inversion to the symmetry operation
                                        (not covered by the index alone). Defaults to False.

        Returns:
            arraylike(dim, dim): the resulting hermitian operator `U_S H(k) U^+_S exp(ik(1+S)(r_1-r_2))`
        """
        # apply inversion first, as that is the simple part
        hamiltonian_inv = np.array(hamiltonian)
        n1 = 0
        for u1, r1 in zip(self.U, self.pos):
            d1 = u1.dim()
            u1 = 1 if u1.inv_split > 0 else -1
            if u1 < 0:
                # TODO use r1 !?!
                hamiltonian_inv[n1:n1+d1,:] *= -1
                hamiltonian_inv[:,n1:n1+d1] *= -1
        # now apply the symmetry operation
        result = np.zeros_like(hamiltonian)
        s = self.sym.S[s_index]
        n1 = 0
        for u1, r1 in zip(self.U, self.pos):
            d1 = u1.dim()
            n2 = 0
            for u2, r2 in zip(self.U, self.pos):
                d2 = u2.dim()
                u1_ = u1.U[s_index]
                u2_ = u2.U[s_index]
                fac = np.exp(2j*np.pi * k @ (r1 - s @ r2)) # TODO check again!
                fac *= np.exp(-2j*np.pi * k @ (r2 - s @ r1))
                result[n1:n1+d1,n2:n2+d2] += fac * u1_ @ hamiltonian_inv[n1:n1+d1,n2:n2+d2] @ np.conj(u2_.T)
                n2 += d2
            n1 += d1
        return result

    def symmetrize(self, H_r, neighbors):
        """Symmetrize a hermitian operator of the form

        `H(k) = sum_r(H_r exp(2πikr))`

        where `H_r` is a list of matrices, with `r` as index
        and `r` comes from a list of positions.
        The `r` list is called the neighbors list in solid state physics.

        The symmetrisation is a group mean over the entire group,
        such that the whole operation is an orthogonal projection
        onto the space in which the symmetry operation described
        by this class holds.
        
        Subgroups of the symmetry are not exploited and everything is
        computed in the most verbose way to make this function a good
        reference implementation.
        For a faster function see `self.symmetrizer`.

        Args:
            H_r (arraylike(N_r, dim, dim)): A list of matrices, that make up the hermitian operator.
            neighbors (arraylike(N_r, k-dim)): A list of positions for the matrices in `H_r`.

        Returns:
            arraylike(N_r, dim, dim): The changed H_r, which respects the symmetry.
        """
        # add up all symmetries
        assert len(neighbors) == len(H_r)
        assert len(neighbors[0]) == self.sym.dim()
        neighbors = np.asarray(neighbors)
        neighbor_func = _sym.try_neighbor_function(neighbors)
        
        H_r = np.array(H_r)
        for i, r in enumerate(neighbors):
            if np.linalg.norm(r) == 0:
                H_r[i] = (H_r[i] + np.conj(H_r[i].T)) / 2
                break
        # symmetrize just inversion
        if self.sym.inversion:
            H_r2 = np.array(H_r)
            for i, r in enumerate(neighbors):
                n1 = 0
                p = np.zeros_like(H_r[0])
                for u1, r1 in zip(self.U, self.pos):
                    d1 = u1.dim()
                    u1 = 1 if u1.inv_split > 0 else -1
                    n2 = 0
                    for u2, r2 in zip(self.U, self.pos):
                        d2 = u2.dim()
                        u2 = 1 if u2.inv_split > 0 else -1
                        j, mirror = neighbor_func(-r - 2*(r2 - r1))
                        if j is not None:
                            h = H_r[j]
                            if mirror:
                                h = np.conj(h.T)
                            p[n1:n1+d1,n2:n2+d2] += h[n1:n1+d1,n2:n2+d2] * (u1 * u2)
                        else:
                            p[n1:n1+d1,n2:n2+d2] -= H_r[i,n1:n1+d1,n2:n2+d2]
                        n2 += d2
                    n1 += d1
                H_r2[i] += p
            H_r2 /= 2
        else:
            H_r2 = H_r
        # TODO symmetrize equal orbitals by using their names

        # Now do translation symmetry. It's a normal subgroup, so it can be done separately like this
        # TODO
        #H_r3 = np.zeros_like(H_r2)
        #d = self.sym.dim()
        #for i in range(d):
        #    shift = [neighbor_func(n + self.A[:,i]) for n in neighbors]
        #    H_r3[shift] += self.translation[i][None,:] * H_r2 * self.translation[i][:,None]
        #H_r3 /= self.sym.dim()
        H_r3 = H_r2
        result = np.zeros_like(H_r3) # all U_S are real, so no worries about type here
        # symmetrise with the subgroup sym/inversion (inversion is always a normal subgroup)
        for i, r in enumerate(neighbors):
            # the neighbors are reduced by inversion symmetry
            # now compute result[i] += np.conj(u.T) @ params[j] @ u
            n1 = 0
            for u1, r1 in zip(self.U, self.pos):
                d1 = u1.dim()
                n2 = 0
                for u2, r2 in zip(self.U, self.pos):
                    d2 = u2.dim()
                    p = np.zeros_like(H_r3[i,n1:n1+d1,n2:n2+d2])
                    for k, s in enumerate(self.sym.S):
                        u1_ = u1.U[k]
                        u2_ = u2.U[k]
                        j, mirror = neighbor_func(s @ (r + r2 - r1) - r2 + r1)
                        if j is not None:
                            h = H_r3[j]
                            if mirror:
                                h = np.conj(h.T)
                            p += np.conj(u1_.T) @ h[n1:n1+d1,n2:n2+d2] @ u2_
                        else:
                            p = 0
                            break
                    result[i,n1:n1+d1,n2:n2+d2] += p
                    n2 += d2
                n1 += d1
        return result / len(self.sym.S)
    
    def symmetrizer(self, neighbors) -> Callable:
        """Symmetrize a hermitian operator of the form

        `H(k) = sum_r(H_r exp(2πikr))`

        where `H_r` is a list of matrices, with `r` as index
        and `r` comes from a list of positions.
        The `r` list is called the neighbors list in solid state physics.

        The symmetrisation is a group mean over the entire group,
        such that the whole operation is an orthogonal projection
        onto the space in which the symmetry operation described
        by this class holds.
        
        This function prepares an internal function to apply the symmetrization quickly.
        This is useful if the symmetrization is repeated, since then the preparation pays off.

        Args:
            neighbors (arraylike(N_r, k-dim)): A list of positions for the matrices in `H_r`.

        Returns:
            function: A function which takes a `H_r` and returns the symmetrized version, just like `self.symmetrize` would.
        """
        neighbors = np.asarray(neighbors)
        # prepare neighbor lookup
        neighbor_func = _sym.try_neighbor_function(neighbors)
        # index lookup
        u_index_lookup = []
        n1 = 0
        for u1 in self.U:
            d1 = u1.dim()
            u_index_lookup.append([n1, n1+d1])
            n1 += d1
        # TODO to speed this up 4x, use the normal subgroup relations for the group mean!
        # lookup all the combinations once here
        total_table = []
        inv_table = []
        origin = None
        for i, r in enumerate(neighbors):
            if np.linalg.norm(r) == 0:
                origin = i
            for i1, r1 in enumerate(self.pos):
                u1 = self.U[i1].inv_split
                u1 = 1 if u1 > 0 else -1
                for i2, r2 in enumerate(self.pos):
                    if self.sym.inversion:
                        u2 = self.U[i2].inv_split
                        u2 = 1 if u2 > 0 else -1
                        j, mirror = neighbor_func(-(r - r1 + r2) + r1 - r2)
                        if j is not None:
                            inv_table.append((i, i1, i2, j, 1 if mirror else 0, u1 * u2))
                        else:
                            # to zero out this component, remove what has been added by the identity
                            inv_table.append((i, i1, i2, i, 0, -1))
                    s_table = {}
                    for k, s in enumerate(self.sym.S):
                        j, mirror = neighbor_func(s @ (r - r1 + r2) + r1 - r2)
                        if j is not None:
                            # reduce total_table by merging entries which have the same effect (based on the given self.U)
                            # doing this makes it go from 8448 to 8019 entries in my testcase
                            # NOTE: only works if the U are exact... rounding causes a race condition
                            u1_ = np.round(self.U[i1].U[k], 4)
                            u2_ = np.round(self.U[i2].U[k], 4)
                            u1_ = tuple([tuple(row) for row in u1_])
                            u2_ = tuple([tuple(row) for row in u2_])
                            key = (j, 1 if mirror else 0, u1_, u2_)
                            if key in s_table:
                                #print("reduction found")
                                value = s_table[key]
                                s_table[key] = (value[0], value[1] + 1)
                            else:
                                s_table[key] = (k, 1)
                        else:
                            # this i, i1, i2 needs to be left at 0, any other value would break symmetry
                            # it's okay to clear here, as this step starts at 0 and includes the identity as individual step
                            s_table.clear()
                            break
                    if len(s_table):
                        total_table.extend([(i, k, i1, i2, j, mirror, fac) for (j, mirror, _, _), (k, fac) in s_table.items()])
        #total_table = sorted(total_table, key=lambda x: x[6] + 100 * x[5] + 10000 * x[4] + 1000000 * x[3] + 100000000 * x[2] + 10000000000 * x[1] + 10000000000 * x[0])
        reduced_table = total_table
        # TODO reduce the table by various methods. E.g. i just specifies where to put the calculation.
        # calculations for different i are equal, but repeated. The following statement shows, that's not happening though...
        reduced_table = np.array(reduced_table, dtype=np.int32)
        #print(*np.unique(np.unique(reduced_table[:,1:-1], axis=0, return_counts=True)[1], return_counts=True))

        # TODO reduce inv_table significantly (increasing precision) by cummulating "inv" for equal parameters
        inv_table = np.array(inv_table, dtype=np.int32)
        #print(len(reduced_table)) # can quickly become > 8019
        #print(len(inv_table)) # stays reasonable in size ~ 500
        #print([list(l) for l in reduced_table])
        #print("checksum: ", np.prod(reduced_table + 1, axis=-1).sum()) # checksum
        def symmetrizer_func(H_r):
            assert len(neighbors) == len(H_r)
            if origin is not None:
                H_r = np.array(H_r)
                # this is the only matrix which is allowed to have its inverse in the parameterset
                H_r[origin] = (H_r[origin] + np.conj(H_r[origin].T)) / 2
            # symmetrize just inversion
            if self.sym.inversion:
                H_r2 = np.array(H_r)
                H_r_mirror = np.conj(np.swapaxes(H_r, -1, -2))
                for i, i1, i2, j, mirror, inv in inv_table:
                    start1, end1 = u_index_lookup[i1]
                    start2, end2 = u_index_lookup[i2]
                    h = H_r_mirror[j] if mirror else H_r[j]
                    H_r2[i,start1:end1,start2:end2] += inv * h[start1:end1,start2:end2]
                H_r2 /= 2
            else:
                H_r2 = H_r
            # Now do translation symmetry. It's a normal subgroup, so it can be done separately like this
            #H_r3 = np.zeros_like(H_r)
            #for i in range(self.sym.dim()):
            #    H_r3 += self.translation[i][None,:] * H_r2 * self.translation[i][:,None]
            #H_r3 /= self.sym.dim()
            H_r3 = H_r2
            H_r3_mirror = np.conj(np.swapaxes(H_r3, -1, -2))
            result = np.zeros_like(H_r) # all U_S are real, so no worries about type here
            # symmetrise with the subgroup sym/inversion (inversion is always a normal subgroup)
            # TODO find a way to sort these operations to make it more efficient
            # e.g. sort by u2 so the last matrix multiplication needs to be performed less often.
            # (ideally, remove some iterations from the for loop and move them to numpy)
            for i, k, i1, i2, j, mirror, fac in reduced_table:
                start1, end1 = u_index_lookup[i1]
                start2, end2 = u_index_lookup[i2]
                u1, u2 = self.U[i1].U[k], self.U[i2].U[k]
                h = H_r3_mirror[j] if mirror else H_r3[j]
                result[i,start1:end1,start2:end2] += (fac * np.conj(u1.T)) @ h[start1:end1,start2:end2] @ u2
            return result / len(self.sym.S)
        return symmetrizer_func
    
    def realize_symmetric(self, k_smpl, hamiltonian):
        """Same as `Symmetry.realize_symmetric` but here the data consists
        of hermitian operators that transform with the symmetry described by this class.

        NOTE: this method ONLY works for symmetry reduced points, otherwise it will create duplicates.

        Args:
            k_smpl (arraylike(N_k, k-dim)): The positions that are referred to as k.
            hamiltonian (arraylike(N_k, dim, dim)): The hermitian operator H(k) for the given k positions.

        Returns:
            (ndarray(N_k', k-dim), ndarray(N_k', dim, dim), ndarray(N_k')): (The full k-samples with all symmetries, The full list of hermitian operators with all symmetries, The index of the source k-sample for each new k-smpl)
        """
        order = list(range(len(k_smpl)))
        full_k = list(k_smpl)
        full_hamiltonian = list(hamiltonian)
        for i, (k, H_k) in enumerate(zip(k_smpl, hamiltonian)):
            # asymtotically slow* algorithm to find all symmetric points
            # (fine because it's never going above 48, so asymtotic behaviour is irrelevant)
            scale = np.linalg.norm(k)
            if scale == 0:
                continue
            used_k = [k]
            for inv in [1, -1] if self.sym.inversion else [1]:
                for s_index, s in enumerate(self.sym.S):
                    k_ = s @ k * inv
                    # * this is why this algorithm is asymtotically slow
                    if np.min(np.linalg.norm(used_k - k_.reshape(1, -1), axis=-1)) < scale * 1e-6:
                        continue
                    used_k.append(k_)
                    full_k.append(k_)
                    full_hamiltonian.append(self.apply(k, H_k, s_index, inv < 0))
                    order.append(i)
        # sort by x, y, z compatible with reshape to meshgrid
        order = np.array(order)
        full_k = np.array(full_k)
        full_hamiltonian = np.array(full_hamiltonian)
        for i in range(len(k_smpl[0])):
            # the round here is annoying as it can break at wrong places
            # + np.pi makes it less likely, but it can still happen
            reorder = np.argsort(np.round(full_k[:,i] + np.pi, 4), kind='stable')
            full_k = full_k[reorder]
            full_hamiltonian = full_hamiltonian[reorder]
            order = order[reorder]
        return np.array(full_k), np.array(full_hamiltonian), order


# function for comparing this reference implementation with other implementation
def _compare_hamiltonian_symmetry():
    np.set_printoptions(precision=3, suppress=True)
    ref_bands = [
        [-0.657, -0.657, -0.657, -0.208, 0.25, 0.509, 0.509, 0.509],
        [-0.226, -0.226, -0.191, -0.16, -0.109, -0.077, 0.195, 0.195],
        [-0.321, -0.321, -0.293, -0.268, -0.268, 0.568, 0.568, 0.836],
        [-1.019, -0.375, -0.176, -0.176, -0.176, 0.008, 0.008, 0.008],
    ]
    k_smpl = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
    ]
    hsym = HamiltonianSymmetry(_sym.Symmetry.cubic(True))
    hsym.append_s((0.0, 0.0, 0.0), "A")
    hsym.append_p((0.0, 0.0, 0.0), "A")
    hsym.append_s((0.5, 0.5, 0.5), "B")
    hsym.append_d3((0.5, 0.5, 0.5), "B")
    #for s in hsym.sym.S:
    #    print(s)
    
    neighbors = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)) # works well
    neighbors = _sym.Symmetry.cubic(True).complete_neighbors(neighbors)
    
    # make a testcase to validate on a particular result
    params = []
    for ni in range(0, len(neighbors)):
        params.append([[float(i * (j + ni)) for j in range(hsym.dim())] for i in range(hsym.dim())])
    #params = hsym.symmetrizer(neighbors)(params)
    params = hsym.symmetrize(params, neighbors)
    print(params)

    # make a testcase to test the bandstructure calculation based on thes symmetrizer
    np.set_printoptions(linewidth=1000)
    import bandstructure
    tb = bandstructure.BandStructureModel.init_tight_binding_from_ref(_sym.Symmetry.none(), neighbors, k_smpl, np.zeros_like(ref_bands), 0, 0, cos_reduced=False, exp=True)
    tb.symmetrizer = hsym.symmetrizer(neighbors) # for this: cos_reduced=False, exp=True
    tb.params = np.zeros_like(tb.params)
    np.random.seed(837258985)
    tb.params = np.random.standard_normal(tb.params.shape) + np.random.standard_normal(tb.params.shape) * 1j
    for i in range(10):
        tb.normalize()
        tb.params = np.round(tb.params, 3)
        tb.normalize()
    tb.params = np.round(tb.params, 3)
    print("vec![")
    for i in range(len(tb.params)):
        print(repr(tb.params[i]).replace("array", "arr!").replace(" ", "").replace(",", ", ").replace(" \n", "\n").replace("(", "").replace(")", "").replace("j", " i"), ",", sep="")
    print("]")
    tb.params *= 0.1
    print("initial loss:", tb.loss(k_smpl, ref_bands, np.ones_like(ref_bands[0]), 0))
    #print(tb.f((0.1,0.2,0.3)))
    iterations = 100
    tb.optimize(k_smpl, 1.0, ref_bands, 1.0, 0, iterations, 1, use_pinv=True)
    np.set_printoptions(precision=8)
    print(tb.params)
    print("final loss:", tb.loss(k_smpl, ref_bands, np.ones_like(ref_bands[0]), 0))
    #print()
    #print(tb(k_smpl))

if __name__ == "__main__":
    _compare_hamiltonian_symmetry()