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
    def __init__(self, sym, N):
        self.U = np.array([np.eye(N) for s in sym.S])
        # inversion symmetry can be handled separately, as it is in the centrum of the group
        self.inv_split = N # number of 1 eigenvalues of the inversion symmetry unitary representations
        self.sym = sym
    
    def dim(self):
        return len(self.U[0])
    
    def copy(self):
        u_repr = UnitaryRepresentation(self.sym.copy(), 1)
        u_repr.inv_split = self.inv_split
        u_repr.U = np.array(self.U)
        return u_repr
    
    # create the symmetry group and the representation from a generator set
    # fails if the generators structure doesn't match
    def from_generator(S_G, U_G, inversion=True, invert_inversion=False):
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
        u_repr.inv_split = 0 if invert_inversion else N
        u_repr.U = U
        return u_repr
    
    # check if this symmetry matches the given one and if so, reorder the elements to match the order of the given symmetry
    def match_sym(self, sym):
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
                u = u1 @ u2 # order reversed
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
    def symmetrize2(self, params, neighbors):
        if len(self.U) <= 1:
            return params # do nothing if self.U is empty, which stands for all U being the unit matrix
        result = np.zeros_like(params) # all U_S are real, so no worries about type here
        # add up all symmetries
        assert len(neighbors) == len(params) # TODO this doesn't match my definition without inversion symmetry...
        assert len(self.sym.S) == len(self.U)
        k = self.inv_split
        if self.sym.inversion:
            # only half of the neighbor terms are present, but the symmetry is important
            for i, r in enumerate(neighbors):
                for s, u in zip(self.sym.S, self.U):
                    r_ = s @ r
                    j = np.argmin(np.minimum(np.linalg.norm(neighbors - r_[None, :], axis=-1),
                                             np.linalg.norm(neighbors + r_[None, :], axis=-1)))
                    if np.linalg.norm(r_) == 0:
                        # center case
                        p = np.array(params[j])
                        p[:k,k:] = 0 # could also do this at the end
                        p[k:,:k] = 0
                        result[i] += np.conj(u.T) @ p @ u
                    elif np.linalg.norm(neighbors[j] + r_) < 1e-7:
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
                    r_ = s.T @ r
                    # even here the neighbors are reduced by inversion symmetry, but they are duplicated to cos and sin terms instead
                    j = np.argmin(np.minimum(np.linalg.norm(neighbors - r_[None, :], axis=-1),
                                             np.linalg.norm(neighbors + r_[None, :], axis=-1)))
                    result[i] += u @ params[j] @ np.conj(u.T)
        return result / len(self.U)
    
    # check if a hamilton operator satisfies the symmetry of this unitary representation
    # hamilon is a function k -> matrix
    def check_symmetry(self, hamilton):
        # random sample points
        k_smpl = np.random.random((50, len(self.sym.S[0])))
        k_smpl = ((0.0, 0.5, 0.0),)
        for k in k_smpl:
            values = []
            for s, u in zip(self.sym.S, self.U):
                values.append(np.conj(u.T) @ hamilton(s @ k) @ u)
            if np.linalg.norm(np.std(values, axis=0)) > 1e-7:
                print("symmetry error")
                #print(np.std(values, axis=0))
                print((np.std(values, axis=0) > 1e-7).astype(np.int8)) # patterns more clear
                return False
        # TODO check inversion symmetry
        return True
    
    def check_neighbors(self, neighbors):
        # only half of the neighbor terms are present, but the symmetry is important
        count = [0]*len(neighbors)
        for i, r in enumerate(neighbors):
            for s, u in zip(self.sym.S, self.U):
                r_ = s @ r
                j = np.argmin(np.minimum(np.linalg.norm(neighbors - r_[None, :], axis=-1), np.linalg.norm(neighbors + r_[None, :], axis=-1)))
                count[j] += 1
        for c in count:
            if c != len(self.U):
                raise ValueError("neighbors need to be choosen to fit the symmetry, however counting occurences has found the numbers " + str(count))

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
    def d3(invert_inversion, sqrt3=3**.5):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((1,0,0), (0,0,1), (0,-1,0)),
             ((0,0,-1), (0,1,0), (1,0,0)),
             ((0,1,0), (-1,0,0), (0,0,1))]
        U = [((1,0), (0,1)),
             ((1,0), (0,-1)),
             ((-.5,.5*sqrt3), (.5*sqrt3,.5)),
             ((-.5,-.5*sqrt3), (-.5*sqrt3,.5))]
        return UnitaryRepresentation.from_generator(S, U, True, invert_inversion)
    # 1d representations for cubic symmetry O_h
    def one_dim(invert, invert_inversion, inversion=True):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((1,0,0), (0,0,1), (0,-1,0)),
             ((0,0,-1), (0,1,0), (1,0,0)),
             ((0,1,0), (-1,0,0), (0,0,1))]
        x = -1 if invert else 1
        U = [((1,),),
             ((x,),),
             ((x,),),
             ((x,),)]
        return UnitaryRepresentation.from_generator(S, U, inversion=inversion, invert_inversion=invert_inversion)

test_u_repr = UnitaryRepresentation.one_dim(True, True, True) + UnitaryRepresentation.d3(True) + UnitaryRepresentation.so3()
neighbors = ((0,0,0), (1,0,0), (0,1,0), (0,0,1))
test_params = np.random.random((len(neighbors), 1+2+3, 1+2+3))
test_params2 = test_u_repr.symmetrize2(test_params, neighbors)
test_params3 = test_u_repr.symmetrize2(test_params2, neighbors)
assert np.linalg.norm(test_params2 - test_params3) < 1e-7