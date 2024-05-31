import numpy as np
from symmetry import *
from unitary_representations import *

# This file contains a structure very similar to
# unitary representations, however it only works for direct products
# of irrep unitary representations and additionally has information
# about the position of the represented atomic orbitals in the unit cell.
# This way it can represent a more general set of symmetries


# direct sum of two matrices (block diagonal concatenation)
def direct_sum2(a, b):
    return np.block([[a, np.zeros((np.shape(a)[0], np.shape(b)[1]))], [np.zeros((np.shape(b)[0], np.shape(a)[1])), b]])


def hamiltonian_symmetry(sym: Symmetry, urepr: list = [], pos: list = []):
    hsym = HamiltonianSymmetry(sym)
    for u, p in zip(urepr, pos):
        hsym.append(u, p)
    return hsym


class HamiltonianSymmetry:
    # sym - symmetry group
    # A - basis matrix
    def __init__(self, sym: Symmetry, A):
        self.sym = sym
        self.U = [] # unitary representations
        self.pos = [] # e.g. [[0,0,0], [1/4,1/4,1/4]] for k-dependence of symmetry
        self.names = [] # e.g. ["C", "C"] for exchange symmetriesCu
        # translation symmetries can also be handled separately
        self.translation = np.ones((sym.dim(), 0))
    
    # dimension (also called degree) of the representation
    def dim(self):
        # sum of the representations, used in the direct sum
        return sum((u.dim() for u in self.U))
    
    def copy(self):
        u_repr = HamiltonianSymmetry(self.sym.copy(), 1)
        u_repr.inv = np.array(self.inv)
        u_repr.U = [u.copy() for u in self.U]
        return u_repr
    
    def __len__(self):
        return len(self.sym)
    
    def append(self, urepr: UnitaryRepresentation, pos, name):
        if len(pos) != self.sym.dim():
            raise ValueError(f"position (dimension {len(pos)}) needs to match the dimension of the symmetry ({self.sym.dim()})")
        if not (urepr.inv_split == 0 or urepr.inv_split == urepr.dim()):
            raise ValueError("only unitary representations with uniform inversion behavior are allowed. Otherwise they are reducible and can be added separately.")
        urepr = urepr.copy()
        urepr.match_sym(self.sym)
        self.U.append(urepr)
        self.pos.append(np.array(pos))
        self.names.append(name)
    
    def append_s(self, pos, name):
        self.append(UnitaryRepresentation(self.sym, 1), pos, name)

    def append_p(self, pos, name):
        # TODO check this!
        urepr = UnitaryRepresentation(self.sym, self.sym.dim())
        urepr.U = self.sym.S
        urepr.inv_split = 0
        self.append(urepr, pos, name)

    def append_d2(self, pos, name):
        # TODO make this work for all symmetries
        self.append(UnitaryRepresentation.d3(False), pos, name)

    def append_d3(self, pos, name):
        # TODO check this!
        urepr = UnitaryRepresentation(self.sym, self.sym.dim())
        urepr.U = self.sym.S / np.linalg.det(self.sym.S)[:,None,None]
        urepr.inv_split = self.sym.dim() # don't use -1 on inversion
        self.append(urepr, pos, name)
    
    # parameter symmetrisation with H_r type
    def symmetrize(self, H_r, neighbors):
        if len(self.U) <= 1:
            return H_r # do nothing if self.U is empty, which stands for all U being the unit matrix
        # add up all symmetries
        assert len(neighbors) == len(H_r)
        assert len(neighbors[0]) == self.sym.dim()
        neighbors = np.asarray(neighbors)
        neighbor_func = try_neighbor_function(neighbors)
        
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
                        j, mirror = neighbor_func(-r + 2*(r2 - r1))
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
        # Now do translation symmetry. It's a normal subgroup, so it can be done separately like this
        H_r3 = np.zeros_like(H_r)
        d = self.sym.dim()
        for i in range(d):
            shift = [neighbor_func(n + self.A[:,i]) for n in neighbors]
            H_r3[shift] += self.translation[i][None,:] * H_r2 * self.translation[i][:,None]
        H_r3 /= self.sym.dim()
        result = np.zeros_like(H_r) # all U_S are real, so no worries about type here
        # symmetrise with the subgroup sym/inversion (inversion is always a normal subgroup)
        for i, r in enumerate(neighbors):
            # the neighbors are reduced by inversion symmetry
            # now compute result[i] += np.conj(u.T) @ params[j] @ u
            # but with u = direct_sum(u1, u2, ...)
            # a @ u = (a @ direct_sum(u1, zeros...) + a @ direct_sum(zeros, u2, zeros...) + ...)
            #       = a @ direct_sum(u1, eye...) @ direct_sum(eye, u2, eye...) @ ...
            #for k, s in enumerate(self.sym.S):
            #    j, mirror = neighbor_func(r_)
            #    p = H_r3[j]
            #    if mirror:
            #        p = np.conj(p.T) # get copy with H_{-r}=H_r^+
            #    else:
            #        p = np.array(p) # copy
            #    n = 0
            #    for u in self.U:
            #        d = u.dim()
            #        u = u.U[k]
            #        p[:,n:n+d] = p[:,n:n+d] @ u
            #        p[n:n+d,:] = np.conj(u.T) @ p[n:n+d,:]
            #        n += d
            n1 = 0
            for u1, r1 in zip(self.U, self.pos):
                d1 = u1.dim()
                n2 = 0
                for u2, r2 in zip(self.U, self.pos):
                    d2 = u2.dim()
                    p = np.zeros_like(H_r[i,n1:n1+d1,n2:n2+d2])
                    for k, s in enumerate(self.sym.S):
                        u1_ = u1.U[k]
                        u2_ = u2.U[k]
                        j, mirror = neighbor_func(s @ (r + r1 - r2) - r1 + r2)
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
    
    def symmetrizer(self, neighbors):
        neighbors = np.asarray(neighbors)
        # prepare neighbor lookup
        neighbor_func = try_neighbor_function(neighbors)
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
                        j, mirror = neighbor_func(-(r + r1 - r2) - r1 + r2)
                        if j is not None:
                            inv_table.append((i, i1, i2, j, 1 if mirror else 0, u1 * u2))
                        else:
                            # to zero out this component, remove what has been added by the identity
                            inv_table.append((i, i1, i2, i, 0, -1))
                    s_table = {}
                    for k, s in enumerate(self.sym.S):
                        j, mirror = neighbor_func(s @ (r + r1 - r2) - r1 + r2)
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
            H_r3 = np.zeros_like(H_r)
            for i in range(self.sym.dim()):
                H_r3 += self.translation[i][None,:] * H_r2 * self.translation[i][:,None]
            H_r3 /= self.sym.dim()
            H_r3_mirror = np.conj(np.swapaxes(H_r2, -1, -2))
            result = np.zeros_like(H_r) # all U_S are real, so no worries about type here
            # symmetrise with the subgroup sym/inversion (inversion is always a normal subgroup)
            # TODO find a way to sort these operations to make it more efficient
            # (remove some iterations from the for loop and move them to numpy)
            for i, k, i1, i2, j, mirror, fac in reduced_table:
                start1, end1 = u_index_lookup[i1]
                start2, end2 = u_index_lookup[i2]
                u1, u2 = self.U[i1].U[k], self.U[i2].U[k]
                h = H_r3_mirror[j] if mirror else H_r3[j]
                result[i,start1:end1,start2:end2] += (fac * np.conj(u1.T)) @ h[start1:end1,start2:end2] @ u2
            return result / len(self.sym.S)
        return symmetrizer_func
    
def test_hamiltonian_symmetry():
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)
    UR = UnitaryRepresentation
    # make equivalent unitary representation and hamilton symmetry
    test_u_repr = UR.one_dim(True, False) + UR.d3(False) + UR.o3()
    assert test_u_repr.check_U()
    test_h_sym = HamiltonianSymmetry(Symmetry.cubic(True))
    test_h_sym.append(UR.one_dim(True, False), [0, 0, 0], "")
    test_h_sym.append(UR.d3(False), [0, 0, 0], "")
    test_h_sym.append(UR.o3(), [0, 0, 0], "")
    # test if they are the same
    neighbors = test_u_repr.sym.complete_neighbors([(0,0,0), (0,0,1), (1,1,1), (2,1,0)]) # with 2,1,3 is also good for the test
    test_rnd = np.random.random((len(neighbors),)+(test_u_repr.dim(),)*2)
    test_rnd = test_rnd + np.random.random((len(neighbors),)+(test_u_repr.dim(),)*2) * 1j
    test_C_r = test_rnd + np.conj(np.swapaxes(test_rnd, -1, -2))
    # convert C_r to H_r
    test_H_r = np.array(test_C_r)
    n = test_u_repr.inv_split
    test_H_r[:,n:,:n] *= 1j
    test_H_r[:,:n,n:] *= 1j

    test_C_r2 = test_u_repr.symmetrize(test_C_r, neighbors)
    test_H_r3 = test_h_sym.symmetrize(test_H_r, neighbors)
    hsym_symm = test_h_sym.symmetrizer(neighbors)
    test_H_r4 = hsym_symm(test_H_r)
    test_H_r5 = hsym_symm(test_H_r4)
    assert np.linalg.norm(test_H_r4 - test_H_r5) < 1e-7
    test_C_r3 = np.array(test_H_r3)
    test_C_r3[:,n:,:n] *= -1j
    test_C_r3[:,:n,n:] *= -1j
    test_C_r4 = np.array(test_H_r4)
    test_C_r4[:,n:,:n] *= -1j
    test_C_r4[:,:n,n:] *= -1j
    #np.set_printoptions(precision=3, suppress=True, linewidth=1000)
    assert np.linalg.norm(test_C_r2 - test_C_r4) < 1e-7
    assert np.linalg.norm(test_C_r2 - test_C_r3) < 1e-7

    # test a 1D problem with inversion symmetry
    sym = Symmetry.inv(1)
    hsym = HamiltonianSymmetry(sym)
    hsym.append_s((0,), "s")
    hsym.append_p((0.5,), "p")
    # now symmetrize a parameterset where I already know the correct symmetrized version
    neighbors = [(0,), (1,), (2,)]
    H_r = np.random.random((3, 2, 2))
    H_r_sym = np.zeros_like(H_r)
    H_r_sym[0] = (H_r[0] + H_r[0].T) / 2
    H_r_sym[1] = H_r[1]
    H_r_sym[2] = H_r[2]
    H_r_sym[1,1,0] = (H_r[1,1,0] - H_r_sym[0,0,1]) / 2
    H_r_sym[1,0,1] = (H_r[1,0,1] - H_r_sym[0,1,0]) / 2
    H_r_sym[0,0,1] = H_r_sym[0,1,0] = -H_r_sym[1,0,1]
    H_r_sym[2,0,1] = (H_r[2,0,1] - H_r[1,1,0]) / 2
    H_r_sym[1,1,0] = -H_r_sym[2,0,1]
    H_r_sym[2,1,0] = 0
    H_r_sym2 = hsym.symmetrizer(neighbors)(H_r)
    H_r_sym3 = hsym.symmetrize(H_r, neighbors)
    #print(f"symmetrizer\n{H_r_sym2}\nvs written out\n{H_r_sym}")
    assert np.linalg.norm(H_r_sym - H_r_sym2) < 1e-7
    assert np.linalg.norm(H_r_sym - H_r_sym3) < 1e-7

    # test a 2D problem with rotation symmetry
    sym = Symmetry.o2()
    hsym = HamiltonianSymmetry(sym)
    hsym.append_s((0,0), "s")
    hsym.append_p((0.5,0.5), "p")
    neighbors = [(0,0), (1,0), (0,1), (1,1), (1,-1)]
    H_r = np.random.random((5, 3, 3))
    hsym = hsym.symmetrizer(neighbors)
    H_r_sym = hsym(H_r)
    H_r_sym2 = hsym(H_r_sym)
    assert np.linalg.norm(H_r_sym - H_r_sym2) < 1e-7
    # test symmetry of the eigenvalueproblem
    # TODO

# function for comparing this reference implementation with other implementation
def compare_hamiltonian_symmetry():
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
    hsym = HamiltonianSymmetry(Symmetry.cubic(True))
    hsym.append_s((0.0, 0.0, 0.0), "A")
    hsym.append_p((0.0, 0.0, 0.0), "A")
    hsym.append_s((0.5, 0.5, 0.5), "B")
    hsym.append_d3((0.5, 0.5, 0.5), "B")
    #for s in hsym.sym.S:
    #    print(s)
    
    neighbors = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)) # works well
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    
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
    tb = bandstructure.BandStructureModel.init_tight_binding_from_ref(Symmetry.none(), neighbors, k_smpl, np.zeros_like(ref_bands), 0, 0, cos_reduced=False, exp=True)
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
