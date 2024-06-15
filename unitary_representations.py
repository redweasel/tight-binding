import numpy as np
from symmetry import *

# direct sum of two matrices (block diagonal concatenation)
def direct_sum2(a, b):
    return np.block([[a, np.zeros((np.shape(a)[0], np.shape(b)[1]))], [np.zeros((np.shape(b)[0], np.shape(a)[1])), b]])

def kron(*a):
    if len(a) == 0:
        raise ValueError("no parameters given")
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return np.kron(*a)
    return kron(np.kron(a[0], a[1]), *a[2:])

def direct_sum(*a):
    if len(a) == 0:
        raise ValueError("no parameters given")
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return direct_sum2(*a)
    return direct_sum(direct_sum2(a[0], a[1]), *a[2:])


class UnitaryRepresentation:
    def __init__(self, sym: Symmetry, N):
        self.U = np.array([np.eye(N) for s in sym.S])
        # inversion symmetry can be handled separately, as it is in the centrum of the group
        self.inv_split = N # number of 1 eigenvalues of the inversion symmetry unitary representations
        self.sym = sym
    
    # dimensions (also called degree) of the representation
    def dim(self):
        return len(self.U[0])
    
    def copy(self):
        u_repr = UnitaryRepresentation(self.sym.copy(), 1)
        u_repr.inv_split = self.inv_split
        u_repr.U = np.array(self.U)
        return u_repr
    
    # create the symmetry group and the representation from a generator set
    # fails if the generators structure doesn't match
    def from_generator(S_G, U_G, inversion=True, negate_inversion=False):
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
        u_repr = UnitaryRepresentation(Symmetry(np.array(S), inversion), N)
        u_repr.inv_split = 0 if negate_inversion else N
        u_repr.U = np.array(U)
        return u_repr
    
    # check if this symmetry matches the given one and if so, reorder the elements to match the order of the given symmetry
    def match_sym(self, sym: Symmetry):
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
    
    def check_U(self):
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
    
    # symmetrize a collection of matrices, such that they obey H(Sr) = U_S H(r) (U_S)^+ where r is one of the neighbor positions
    # use symmetrizer(neighbors)(params) for repeated use of symmetrize
    # IMPORTANT: this one is very slow and is only kept for testing the symmetrizer
    def symmetrize(self, params, neighbors):
        if len(self.U) <= 1:
            return params # do nothing if self.U is empty, which stands for all U being the unit matrix
        result = np.zeros_like(params) # all U_S are real, so no worries about type here
        # add up all symmetries
        assert len(neighbors) == len(params) # TODO this doesn't match my definition without inversion symmetry...
        assert len(self.sym.S) == len(self.U)
        neighbor_func = neighbor_function(neighbors)
        k = self.inv_split
        if self.sym.inversion:
            # only half of the neighbor terms are present, but the symmetry is important
            for i, r in enumerate(neighbors):
                for s, u in zip(self.sym.S, self.U):
                    r_ = s @ r
                    j, mirror = neighbor_func(r_)
                    if np.linalg.norm(r_) == 0:
                        # center case
                        p = np.array(params[j])
                        p[:k,k:] = 0 # could also do this at the end
                        p[k:,:k] = 0
                        result[i] += np.conj(u.T) @ p @ u
                    elif mirror:
                        # mirrored case, read the mirrored matrix
                        p = np.array(params[j])
                        p[:k,k:] *= -1
                        p[k:,:k] *= -1
                        result[i] += np.conj(u.T) @ p @ u
                    else:
                        result[i] += np.conj(u.T) @ params[j] @ u
        else:
            for i, r in enumerate(neighbors):
                for s, u in zip(self.sym.S, self.U):
                    r_ = s @ r
                    # even here the neighbors are reduced by inversion symmetry,
                    # but they are duplicated to cos and sin terms instead
                    j, mirror = neighbor_func(r_)
                    # TODO check
                    result[i] += np.conj(u.T) @ params[j] @ u
        return result / len(self.U)
    
    # symmetrize a collection of matrices, such that they obey H(Sr) = U_S H(r) (U_S)^+ where r is one of the neighbor positions
    # returns a function that does that ^
    # (has a build in sym.neighbors_check)
    def symmetrizer(self, neighbors):
        if len(self.U) <= 1:
            return lambda x: x # do nothing if self.U is empty, which stands for all U being the unit matrix
        assert len(self.sym.S) == len(self.U)
        neighbor_func = neighbor_function(neighbors)
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
    
    # symmetrize matrices such that they are invariant under U_S @ mat @ U_S^+ if Sk=k
    def symmetrize2(self, k_smpl, mat):
        count = np.zeros(len(k_smpl), dtype=np.int32)
        result = np.zeros_like(mat)
        for sign in [-1, 1] if self.sym.inversion else [1]:
            if sign == 1:
                # apply U_I (commutes with all other U_S)
                k = self.inv_split
                result[:,:k,k:] *= -1
                result[:,k:,:k] *= -1
            for s, u in zip(self.sym.S, self.U):
                invariants = (np.einsum("ij,nj->ni", sign*s - np.eye(len(s)), k_smpl)**2).sum(-1) < 1e-7
                count += invariants
                result[invariants] += np.einsum("nij,li,mj->nlm", mat[invariants], u, np.conj(u))
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

    # check if a hamilton operator satisfies the symmetry of this unitary representation
    # hamilon is a function k -> matrix
    def check_symmetry(self, hamilton):
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
    
    # get the subspace structure and dimension for all irreducible representations that were part of the direct sum/product construction
    def subspaces(self):
        # finding the symmetric spaces in U_S in general is much much harder
        # -> assuming U_S is a direct sum of irreducible unitary representations
        # (for a more general construction one can use characters (traces of the representation matrices))
        occupancy_table = np.sum(np.abs(self.U), axis=0) > 1e-7
        groups, counts = np.unique(occupancy_table, return_counts=True, axis=0)
        assert np.all(groups.astype(np.int32).sum(1) == counts) and np.all(groups.astype(np.int32).sum(0) == 1), "group size didn't match dimension, this has probably not been constructed using direct sums/products"
        return groups, counts

    # direct sum of representations
    def __add__(self, rhs):
        rhs = rhs.copy()
        rhs.match_sym(self.sym)
        u_repr = UnitaryRepresentation(self.sym, self.dim() + rhs.dim())
        # one of the two representations must fully commit to -1 or 1 on inversion
        if self.inv_split == 0:
            # self is commited to -1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum2(u2, u1)
        elif self.inv_split == self.dim():
            # self is commited to 1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum2(u1, u2)
        elif rhs.inv_split == 0:
            # rhs is commited to -1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum2(u1, u2)
        elif rhs.inv_split == rhs.dim():
            # rhs is commited to 1 on inversion
            for i, (u1, u2) in enumerate(zip(self.U, rhs.U)):
                u_repr.U[i] = direct_sum2(u2, u1)
        else:
            raise ValueError("one of the added representations needs to full commit to 1 or -1 on inversion.")
        u_repr.inv_split = self.inv_split + rhs.inv_split
        return u_repr

    # direct product with an n-dim identity matrix from the left
    # = direct sum of n times this representation
    def __mul__(self, rhs):
        if int(rhs) != rhs:
            print("can only multiply with an integer number")
        rhs = int(rhs)
        res = self.copy()
        res.U = np.kron(res.U, np.eye(rhs)[None, ...])
        res.inv_split *= rhs
        return res

    ### unitary irreducible representations

    # O(3) with inversion -1 for cubic symmetry O_h
    def o3():
        sym = Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = np.asarray(sym.S)
        u_repr.inv_split = 0 # no 1 eigenvalues in inversion
        return u_repr
    # SO(3) with 1 on inversion for cubic symmetry O_h
    def so3():
        sym = Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = np.asarray(sym.S)
        u_repr.inv_split = 3 # all 1 eigenvalues in inversion
        return u_repr
    # reflected O(3) with inversion -1 for cubic symmetry O_h
    def o3ri():
        sym = Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = -np.asarray(sym.S)
        u_repr.inv_split = 0 # no 1 eigenvalues in inversion
        return u_repr
    # reflected O(3) with 1 on inversion for cubic symmetry O_h
    def o3r():
        sym = Symmetry.cubic(True)
        u_repr = UnitaryRepresentation(sym, 3)
        u_repr.U = -np.asarray(sym.S)
        u_repr.inv_split = 3 # all 1 eigenvalues in inversion
        return u_repr
    # D_3 dihedral triangle symmetry for cubic symmetry O_h
    # sqrt3 can be given in arbitrary precision if needed
    def d3(negate_inversion, inversion=True, sqrt3=3**.5):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((0,1,0), (-1,0,0), (0,0,1)), # R_z
             ((0,0,-1), (0,1,0), (1,0,0)), # R_y
             ((1,0,0), (0,0,1), (0,-1,0))] # R_x
        U = [((1,0), (0,1)),
             ((1,0), (0,-1)),
             ((-.5,.5*sqrt3), (.5*sqrt3,.5)),
             ((-.5,-.5*sqrt3), (-.5*sqrt3,.5))]
        return UnitaryRepresentation.from_generator(S, U, inversion, negate_inversion)
    # 1d representations for cubic symmetry O_h
    def one_dim(invert, negate_inversion, inversion=True):
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

def test_unitary_representations():
    UR = UnitaryRepresentation
    test_u_repr = UR.one_dim(True, True, True) + UR.d3(True) + UR.so3()
    #test_u_repr = UR.one_dim(False, False, True) + UR.d3(False) + UR.so3() + UR.o3()
    assert test_u_repr.check_U()
    neighbors = test_u_repr.sym.complete_neighbors([(0,0,0), (0,0,1), (1,1,1), (2,1,0)]) # with 2,1,3 it will use invcopy
    test_params = np.random.random((len(neighbors),)+(test_u_repr.dim(),)*2)
    test_params = test_params + np.random.random((len(neighbors),)+(test_u_repr.dim(),)*2) * 1j
    test_params = test_params + np.conj(np.swapaxes(test_params, -1, -2))

    test_params2 = test_u_repr.symmetrize(test_params, neighbors)
    test_params3 = test_u_repr.symmetrize(test_params2, neighbors)
    assert np.linalg.norm(test_params2 - test_params3) < 1e-7
    _test_symmetrizer = test_u_repr.symmetrizer(neighbors)
    test_params2 = _test_symmetrizer(test_params)
    test_params4 = _test_symmetrizer(test_params2)
    assert np.linalg.norm(test_params2 - test_params4) < 1e-7
    #print(np.linalg.norm(test_params3 - test_params4, axis=(-1,-2)))
    #print((np.linalg.norm(test_params3 - test_params4, axis=0) > 0.0001).astype(int))
    assert np.linalg.norm(test_params3 - test_params4) < 1e-7
