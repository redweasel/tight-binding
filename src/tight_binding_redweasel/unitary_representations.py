import numpy as np
from typing import Self, Callable
from . import symmetry as _sym
from .linalg import kron, direct_sum
from numbers import Integral

class UnitaryRepresentation:
    """
    This class is a unitary representation from group representation theory.
    A representation for a symmetry S is a matrix set D(S) such that

    `D(S1) D(S2) = D(S1 S2)`

    These representation can always be transformed by a change of basis into a
    representation, that is a direct sum of unitary irreduzible representations (unitary irreps).
    
    This class can represent any unitary representation, however inversion symmetry is handled separately.
    """

    def __init__(self, sym: _sym.Symmetry, N):
        self.U = np.array([np.eye(N) for _ in sym.S])
        # inversion symmetry can be handled separately, as it is in the centrum of the group
        self.inv_split = N # number of 1 eigenvalues of the inversion symmetry unitary representations
        self.sym = sym
    
    def dim(self):
        """
        Returns:
            int: Dimension (also called degree) of the unitary representation.
        """
        return len(self.U[0])
    
    def copy(self) -> Self:
        u_repr = UnitaryRepresentation(self.sym.copy(), 1)
        u_repr.inv_split = self.inv_split
        u_repr.U = np.array(self.U)
        return u_repr
    
    @staticmethod
    def from_generator(S_G, U_G, inversion=True, negate_inversion=False) -> Self:
        """Create the symmetry group and the representation from a generator set of size N_G.
        Fails if the generators structure is not closed.

        Args:
            S_G (arraylike(N_G, k-dim, k-dim)): The symmetry operations in the generating set.
            U_G (arraylike(N_G, dim, dim)): The unitary operations in the generating set.
            inversion (bool, optional): If True, the inversion will be added to the symmetry. Defaults to True.
            negate_inversion (bool, optional): If True, the factor for inversion will be -1, otherwise 1. Defaults to False.

        Returns:
            UnitaryRepresentation: The generated unitary representation.
        """
        Nk = len(S_G[0])
        N = len(U_G[0])
        S = S_G.copy()
        U = U_G.copy()
        # exponentiate the group n times to form the full group
        for _ in range(1000):
            S_new = np.array(S)
            S_new = np.reshape(S_new.reshape(-1, 1, Nk, Nk) @ S_new.reshape(1, -1, Nk, Nk), (-1, Nk, Nk))
            U_new = np.array(U)
            U_new = np.reshape(U_new.reshape(-1, 1, N, N) @ U_new.reshape(1, -1, N, N), (-1, N, N))
            # TODO why does the above not work?
            # convert back to tuples and reduce
            prev_len = len(S)
            S = {}
            for s, u in zip(S_new, U_new):
                s_key = tuple([tuple(s_) for s_ in s])
                if s_key in S:
                    # the symmetry already exists
                    # check if the U_new symmetry matches
                    u_existing = S[s_key][1]
                    assert np.sum((u_existing - u)**2) < 1e-14
                else:
                    S[s_key] = (s_key, u)
            pairs = list(S.values())
            S = []
            U = []
            for s, u in pairs:
                S.append(s)
                U.append(u)
            assert len(S) < 1000 # limitation to avoid endless loops
            if len(S) <= prev_len:
                break
        u_repr = UnitaryRepresentation(_sym.Symmetry(np.array(S), inversion), N)
        u_repr.inv_split = 0 if negate_inversion else N
        u_repr.U = np.array(U)
        return u_repr
    
    def match_sym(self, sym: _sym.Symmetry):
        """Check if this symmetry matches the given one
        and if so, reorder the elements of this symmetry (`self`)
        to match the order of the given symmetry.

        Args:
            sym (Symmetry): The other symmetry.

        Raises:
            ValueError: If the symmetries can not be matched.
        """
        if len(sym.S) != len(self.sym.S):
            raise ValueError("Symmetries don't match (unequal group order)")
        if len(sym.S[0]) != len(self.sym.S[0]):
            raise ValueError("Symmetries don't match (unequal matrix dimension)")
        if sym.inversion != self.sym.inversion:
            raise ValueError("Symmetries don't match (unequal inversion property)")
        Nk = len(sym.S[0])
        S1 = np.reshape(sym.S, (-1, Nk*Nk))
        S2 = np.reshape(self.sym.S, (-1, Nk*Nk))
        # for each point in S1 find the closest in S2
        order = []
        for p1 in S1:
            d = np.linalg.norm(p1[None,:] - S2, axis=-1)
            min_index = np.argmin(d)
            if d[min_index] > 1e-7:
                raise ValueError(f"Symmetries don't match (didn't find {p1} in self)")
            order.append(min_index)
        if len(set(order)) != len(order):
            raise ValueError("Symmetries don't match (duplicates???)")
        self.sym = sym
        self.U = np.array(self.U[order])

    # TODO create an iterator, which lists a representant for each of the possible unitary representations of the symmetry given a partition (multiplicities of band energies at k=0)
    
    def check(self) -> bool:
        N = self.dim()
        # check the structure of the unitary matrices
        # i.e. check if they are a representation of the symmetry by checking 
        for s1, u1 in zip(self.sym.S, self.U):
            for s2, u2 in zip(self.sym.S, self.U):
                s = s1 @ s2
                u = u1 @ u2
                # find s and u in the lists (use approximate equality)
                found = False
                for s3, u3 in zip(self.sym.S, self.U):
                    if np.linalg.norm(s3 - s) < 1e-7:
                        found = True
                        if np.linalg.norm(u3 - u) > 1e-7:
                            print("error in U (U matrices don't fit the structure of the symmetry group)")
                            return False
                if not found:
                    print("error in symmetry (symmetry group is not closed)")
                    return False
        if self.sym.inversion:
            # just check if the inversion element is in the centrum
            inv_U = np.diag([1] * self.inv_split + [-1] * (N - self.inv_split))
            for u in self.U:
                if np.linalg.norm(inv_U @ u - u @ inv_U) > 1e-7:
                    print("error in U (inversion isn't in the group center)")
                    return False
        return True
    
    # symmetrize a collection of matrices, such that they obey  where r is one of the neighbor positions
    # use symmetrizer(neighbors)(H_r) for repeated use of symmetrize
    # IMPORTANT: this one is very slow and is only kept for testing the symmetrizer
    def symmetrize(self, H_r, neighbors):
        """Symmetrize a hermitian operator, such that

        'H(Sr) = U_S H(r) (U_S)^+'

        where `H(r)` is a list of matrices, with `r` as index
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
            H_r (arraylike(N_r, dim, dim)): A list of matrices. One for each r-position.
            neighbors (arraylike(N_r, r-dim)): A list of positions for the matrices in `H_r`.

        Returns:
            arraylike(N_r, dim, dim): The changed H_r, which respects the symmetry.
        """
        if len(self.U) <= 1:
            return H_r # do nothing if self.U is empty, which stands for all U being the unit matrix
        result = np.zeros_like(H_r) # all U_S are real, so no worries about type here
        # add up all symmetries
        assert len(neighbors) == len(H_r) # TODO this doesn't match my definition without inversion symmetry...
        assert len(self.sym.S) == len(self.U)
        neighbor_func = _sym.neighbor_function(neighbors)
        k = self.inv_split
        if self.sym.inversion:
            # only half of the neighbor terms are present, but the symmetry is important
            for i, r in enumerate(neighbors):
                for s, u in zip(self.sym.S, self.U):
                    r_ = s @ r
                    j, mirror = neighbor_func(r_)
                    if np.linalg.norm(r_) == 0:
                        # center case
                        p = np.array(H_r[j])
                        p[:k,k:] = 0 # could also do this at the end
                        p[k:,:k] = 0
                        result[i] += np.conj(u.T) @ p @ u
                    elif mirror:
                        # mirrored case, read the mirrored matrix
                        p = np.array(H_r[j])
                        p[:k,k:] *= -1
                        p[k:,:k] *= -1
                        result[i] += np.conj(u.T) @ p @ u
                    else:
                        result[i] += np.conj(u.T) @ H_r[j] @ u
        else:
            for i, r in enumerate(neighbors):
                for s, u in zip(self.sym.S, self.U):
                    r_ = s @ r
                    # even here the neighbors are reduced by inversion symmetry,
                    # but they are duplicated to cos and sin terms instead
                    j, mirror = neighbor_func(r_)
                    # TODO check
                    result[i] += np.conj(u.T) @ H_r[j] @ u
        return result / len(self.U)
    
    # symmetrize a collection of matrices, such that they obey H(Sr) = U_S H(r) (U_S)^+ where r is one of the neighbor positions
    # returns a function that does that ^
    # (has a build in sym.neighbors_check)
    def symmetrizer(self, neighbors) -> Callable:
        """Symmetrize a hermitian operator, such that

        'H(Sr) = U_S H(r) (U_S)^+'

        where `H(r)` is a list of matrices, with `r` as index
        and `r` comes from a list of positions.
        The `r` list is called the neighbors list in solid state physics.

        The symmetrisation is a group mean over the entire group,
        such that the whole operation is an orthogonal projection
        onto the space in which the symmetry operation described
        by this class holds.
        
        This function prepares an internal function to apply the symmetrization quickly.
        This is useful if the symmetrization is repeated, since then the preparation pays off.

        Args:
            neighbors (arraylike(N_r, r-dim)): A list of positions for the matrices in `H_r`.

        Returns:
            function: A function which takes a `H_r` and returns the symmetrized version, just like `self.symmetrize` would.
        """
        if len(self.U) <= 1:
            return lambda x: x # do nothing if self.U is empty, which stands for all U being the unit matrix
        assert len(self.sym.S) == len(self.U)
        neighbor_func = _sym.neighbor_function(neighbors)
        k = self.inv_split
        task_list_r0 = []
        task_list_rinv = []
        task_list_r = []
        # for each symmetry class of neighbors, only compute one
        # representative and then compute the others from that one directly
        classes = {} # {representative_index: { other_index: transform_index }}
        covered = set()
        # build the symmetry graph for the neighbor list
        neighbors = np.asarray(neighbors)
        count = [0]*len(neighbors)
        for i, r in enumerate(neighbors):
            rep_class = {}
            for m, s in enumerate(self.sym.S):
                r_ = s @ r
                # find r_ or -r_ in neighbors
                j, mirror = neighbor_func(r_)
                count[j] += 1
                if np.linalg.norm(r_) == 0 and self.sym.inversion:
                    # center case
                    task_list_r0.append((i, j, m))
                elif mirror and self.sym.inversion:
                    # mirrored case, read the mirrored matrix
                    if j != i and not j in rep_class:
                        # save which transform is used to generate this non representative
                        rep_class[j] = (m, True)
                    task_list_rinv.append((i, j, m))
                else:
                    assert not mirror
                    if j != i:
                        # save which transform is used to generate this non representative
                        rep_class[j] = (m, False)
                    task_list_r.append((i, j, m))
            if i not in covered:
                classes[i] = rep_class
            covered = covered.union(rep_class.keys())
        for c in count:
            if c != len(self.sym.S):
                raise ValueError("neighbors need to be choosen to fit the symmetry, however counting occurences has found the numbers " + str(count))
        # task_list_r0 should only contain one index, so check that and simplify
        task_list_r0 = tuple(np.reshape(task_list_r0, (-1, 3)).T)
        assert np.all(task_list_r0[0] == task_list_r0[1]) and np.all(task_list_r0[0][0] == task_list_r0[0])
        r0_index = task_list_r0[0][0] if len(task_list_r0) else None
        # the rest can stay as is, just numpyify them
        task_list_rinv = np.reshape(task_list_rinv, (-1, 3))
        task_list_rinv = tuple(task_list_rinv[np.array([i in classes for i in task_list_rinv[:,0]])].T)
        task_list_r = np.reshape(task_list_r, (-1, 3))
        task_list_r = tuple(task_list_r[np.array([i in classes for i in task_list_r[:,0]])].T)
        class_copy = []
        class_invcopy = []
        for i, class_other in classes.items():
            for j, (m, mirrored) in sorted(class_other.items(), key=lambda x: x[0]):
                if mirrored:
                    class_invcopy.append((i, j, m))
                else:
                    class_copy.append((i, j, m))
        class_invcopy = tuple(np.reshape(class_invcopy, (-1, 3)).T) if class_invcopy else None
        class_copy = tuple(np.reshape(class_copy, (-1, 3)).T) if class_copy else None
        # this function is still the performance bottleneck!
        # even though it has no for loops, just because of the size of the symmetry positions.
        # It is O(neighbor_representatives * sym)
        def symmetrizer_func(params):
            result = np.zeros_like(params) # all U_S are real, so no worries about type here
            assert len(neighbors) == len(params) # TODO this doesn't match my definition without inversion symmetry...
            # add up all symmetries
            # center case
            if not r0_index is None:
                result[r0_index] = np.einsum("nki,kl,nlj->ij", np.conj(self.U), params[r0_index], self.U)
                # center case post processing
                result[r0_index,:k,k:] = 0
                result[r0_index,k:,:k] = 0
            # mirrored
            i, j, m = task_list_rinv
            u = self.U[m]
            np.add.at(result, i, np.conj(np.swapaxes(u, -1, -2)) @ params[j] @ u)
            # mirror post processing
            result[:,:k,k:] *= -1
            result[:,k:,:k] *= -1
            # normal
            i, j, m = task_list_r
            u = self.U[m]
            np.add.at(result, i, np.conj(np.swapaxes(u, -1, -2)) @ params[j] @ u)
            # reconstruct all the non representatives now
            # mirrored
            if class_invcopy is not None:
                i, j, m = class_invcopy
                u = self.U[m]
                result[j] = u @ result[i] @ np.conj(np.swapaxes(u, -1, -2))
                # mirrored post processing
                result[j,:k,k:] *= -1
                result[j,k:,:k] *= -1
            # normal
            if class_copy is not None:
                i, j, m = class_copy
                u = self.U[m]
                result[j] = u @ result[i] @ np.conj(np.swapaxes(u, -1, -2))
            return result / len(self.U)
        return symmetrizer_func
    
    # 
    def symmetrize2(self, r_smpl, M_r):
        """Symmetrize matrices `M_r` such that they are
        invariant under `U_S M_r U_S^+` if `Sr=r`.
        This is a subgroup symmetrisation of each individual matrix.

        Args:
            r_smpl (arraylike(N_r, r-dim)): A list of positions for the matrices in `M_r`.
            M_r (arraylike(N_r, dim, dim)): A matrix for each position.

        Returns:
            arraylike(N_r, dim, dim): The changed M_r, which fullfils the subgroup symmetry.
        """
        count = np.zeros(len(r_smpl), dtype=np.int32)
        result = np.zeros_like(M_r)
        for sign in [-1, 1] if self.sym.inversion else [1]:
            if sign == 1:
                # apply U_I (commutes with all other U_S)
                k = self.inv_split
                result[:,:k,k:] *= -1
                result[:,k:,:k] *= -1
            for s, u in zip(self.sym.S, self.U):
                invariants = (np.einsum("ij,nj->ni", sign*s - np.eye(len(s)), r_smpl)**2).sum(-1) < 1e-7
                count += invariants
                result[invariants] += np.einsum("nij,li,mj->nlm", M_r[invariants], u, np.conj(u))
        return result / count[:,None,None]

    # Calculate all the k_smpl and matrices, which can be computed from the given set.
    # This uses the unitary representation to transform the matrices like U_S @ mat or U_S @ mat @ U_S^+ in case Sk=k
    # if inverse is True, U_S^+ is used instead of U_S
    def realize_symmetric_matrices(self, k_smpl, reduced, sorted=True, inverse=False):
        full = list(reduced)
        full_k = list(k_smpl)
        for data, k in zip(reduced, k_smpl):
            # asymtotically slow* algorithm to find all symmetric points
            # (fine because it's never going above 48, so asymtotic behaviour is irrelevant)
            scale = np.linalg.norm(k)
            if scale == 0:
                continue
            used_k = [k]
            for inv in [1, -1] if self.sym.inversion else [1]:
                for s, u in zip(self.sym.S, self.U):
                    k_ = s @ k * inv
                    # * this is why this algorithm is asymtotically slow for large symmetry groups
                    if np.min(np.linalg.norm(used_k - k_.reshape(1, -1), axis=-1)) < scale * 1e-6:
                        continue
                    used_k.append(k_)
                    if inverse:
                        full.append(u @ data @ np.conj(u.T))
                    else:
                        full.append(np.conj(u.T) @ data @ u)
                    full_k.append(k_)
        # sort by x, y, z compatible with reshape to meshgrid
        full_k = np.array(full_k)
        full = np.array(full)
        if sorted:
            for i in range(len(k_smpl[0])):
                # the round here is annoying as it can break at wrong places
                # + np.pi makes it less likely, but it can still happen
                order = np.argsort(np.round(full_k[:,i] + np.pi, 4), kind='stable')
                full_k = full_k[order]
                full = full[order]
        return full_k, full

    def check_symmetry(self, hamilton) -> bool:
        """
        Check if a hermitian operator satisfies the symmetry of this unitary representation.

        Args:
            hamilton (function(k) -> arraylike(dim, dim)): A function that returns a hermitian matrix.

        Returns:
            bool: True if the hermitian operator had this exact symmetry.
        """
        # random sample points
        k_smpl = np.random.random((50, len(self.sym.S[0])))
        k_smpl = ((0.0, 0.5, 0.0),)
        for k in k_smpl:
            values = []
            for s, u in zip(self.sym.S, self.U):
                values.append(np.conj(u.T) @ hamilton([s @ k])[0] @ u)
            if np.linalg.norm(np.std(values, axis=0)) > 1e-7:
                print("symmetry error")
                #print(np.std(values, axis=0))
                print((np.std(values, axis=0) > 1e-7).astype(np.int8)) # patterns more clear
                return False
        # TODO check inversion symmetry
        return True
    
    def subspaces(self):
        """Get the subspace structure and dimension for
        all irreducible representations that were
        part of the direct sum/product construction."""
        # finding the symmetric spaces in U_S in general is much much harder
        # -> assuming U_S is a direct sum of irreducible unitary representations
        # (for a more general construction one can use characters (traces of the representation matrices))
        occupancy_table = np.sum(np.abs(self.U), axis=0) > 1e-7
        groups, counts = np.unique(occupancy_table, return_counts=True, axis=0)
        assert np.all(groups.astype(np.int32).sum(1) == counts) and np.all(groups.astype(np.int32).sum(0) == 1), "group size didn't match dimension, this has probably not been constructed using direct sums/products"
        return groups, counts

    def permute(self, order):
        """Apply a permutation to the basis of the representation"""
        # check that the permutation doesn't mix inversion symmetry
        for i, j in enumerate(order):
            assert (i < self.inv_split) == (j < self.inv_split), "The permutation is not allowed to mix the inversion symmetry"
        for i in range(len(self.U)):
            self.U[i] = self.U[i][order]
            self.U[i] = self.U[i][:, order]
    
    # direct sum of representations
    def __add__(self, rhs):
        rhs = rhs.copy()
        rhs.match_sym(self.sym)
        u_repr = UnitaryRepresentation(self.sym, self.dim() + rhs.dim())
        # one of the two representations must fully commit to -1 or 1 on inversion
        if self.inv_split == 0:
            # self is commited to -1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum(u2, u1)
        elif self.inv_split == self.dim():
            # self is commited to 1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum(u1, u2)
        elif rhs.inv_split == 0:
            # rhs is commited to -1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum(u1, u2)
        elif rhs.inv_split == rhs.dim():
            # rhs is commited to 1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum(u2, u1)
        else:
            raise ValueError("one of the added representations needs to full commit to 1 or -1 on inversion.")
        u_repr.inv_split = self.inv_split + rhs.inv_split
        return u_repr

    # direct product with an n-dim identity matrix from the left
    # = direct sum of n times this representation
    def __mul__(self, rhs):
        if isinstance(rhs, Integral):
            rhs = int(rhs)
            res = self.copy()
            # The order of the kron is a bit unfortunate,
            # but is required to get inv_split working for the general case.
            res.U = np.kron(res.U, np.eye(rhs)[None, ...])
            res.inv_split *= rhs
            return res
        if type(rhs) == type(self):
            # TODO test
            res = rhs.copy()
            res.match_sym(self.sym) # match to self symmetry, like in __add__
            # this is only supported if one of the representations has inv_split at 0 or dim.
            if rhs.inv_split == 0 or rhs.inv_split == rhs.dim():
                # case 1: kron is ordered correctly
                for i in range(len(res)):
                    res.U[i] = np.kron(self.U[i], res.U[i])
                res.inv_split = self.inv_split * rhs.dim()
                if rhs.inv_split == 0:
                    # flip inversion symmetry if self has it
                    res.inv_split = res.dim() - res.inv_split
                    res.U = np.flip(res.U, axis=(1, 2))
            elif self.inv_split == 0 or self.inv_split == self.dim():
                # case 2: kron is flipped
                # TODO make the kron in linalg do this loop correctly and generally
                for i in range(len(res)):
                    res.U[i] = np.kron(res.U[i], self.U[i])
                res.inv_split *= self.dim()
                if self.inv_split == 0:
                    # flip inversion symmetry if self has it
                    res.inv_split = res.dim() - res.inv_split
                    res.U = np.flip(res.U, axis=(1, 2))
            else:
                raise ValueError("Multiplication is not supported for two unitary representations with both mixed inversion symmetry")
            return res
        raise ValueError(f"Unsupported type for rhs: {type(rhs)}")

    ### examples for unitary irreducible representations

    @staticmethod
    def o3() -> Self:
        """O(3) with inversion -1 for cubic symmetry O_h."""
        sym = _sym.Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = np.asarray(sym.S)
        u_repr.inv_split = 0 # no 1 eigenvalues in inversion
        return u_repr
    
    @staticmethod
    def so3() -> Self:
        """SO(3) with 1 on inversion for cubic symmetry O_h."""
        sym = _sym.Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = np.asarray(sym.S)
        u_repr.inv_split = 3 # all 1 eigenvalues in inversion
        return u_repr
    
    @staticmethod
    def o3ri() -> Self:
        """O(3) with 90° rotation -1 and inversion -1 for cubic symmetry O_h."""
        sym = _sym.Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = np.array(sym.S)
        u_repr.U *= np.where((np.abs(np.einsum("nii->n", u_repr.U)) == 1) & (np.prod(np.einsum("nii->ni", u_repr.U), axis=-1) == 0), -1, 1)[:,None,None]
        u_repr.inv_split = 0 # no 1 eigenvalues in inversion
        return u_repr
    
    @staticmethod
    def o3r() -> Self:
        """O(3) with 90° rotation -1 and 1 on inversion for cubic symmetry O_h."""
        sym = _sym.Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = np.array(sym.S)
        u_repr.U *= np.where((np.abs(np.einsum("nii->n", u_repr.U)) == 1) & (np.prod(np.einsum("nii->ni", u_repr.U), axis=-1) == 0), -1, 1)[:,None,None]
        u_repr.inv_split = 3 # all 1 eigenvalues in inversion
        return u_repr
    
    @staticmethod
    def d3(negate_inversion: bool, inversion=True, sqrt3=3**.5) -> Self:
        """D_3 dihedral triangle symmetry for cubic symmetry O_h.
        sqrt3 can be given in arbitrary precision if needed."""
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((0,1,0), (-1,0,0), (0,0,1)), # R_z
             ((0,0,-1), (0,1,0), (1,0,0)), # R_y
             ((1,0,0), (0,0,1), (0,-1,0))] # R_x
        U = [((1,0), (0,1)),
             ((1,0), (0,-1)),
             ((-.5,.5*sqrt3), (.5*sqrt3,.5)),
             ((-.5,-.5*sqrt3), (-.5*sqrt3,.5))]
        return UnitaryRepresentation.from_generator(S, U, inversion, negate_inversion)
    
    @staticmethod
    def one_dim(invert: bool, negate_inversion: bool, inversion=True) -> Self:
        """1d representations for cubic symmetry O_h."""
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((1,0,0), (0,0,1), (0,-1,0)),
             ((0,0,-1), (0,1,0), (1,0,0)),
             ((0,1,0), (-1,0,0), (0,0,1))]
        x = -1 if invert else 1
        U = [((1,),),
             ((x,),),
             ((x,),),
             ((x,),)]
        return UnitaryRepresentation.from_generator(S, U, inversion=inversion, negate_inversion=negate_inversion)
