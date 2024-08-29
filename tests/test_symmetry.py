
import numpy as np

from tight_binding_redweasel.symmetry import *
from tight_binding_redweasel.unitary_representations import *
from tight_binding_redweasel.hamiltonian_symmetry import *

def test_symmetry():
    assert pointcloud_distance([(1, 2), (3, 4)], [(1, 2), (1, 2)]) > 2
    assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 3)]) == 1
    assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 2)]) == 0
    # TODO add automated tests for realize_symmetric_data, reduce_symmetric_data

    # smallest set of generators for O
    S = [[[0,1,0], [-1,0,0], [0,0,1]],
        [[0,0,1], [1,0,0], [0,1,0]]]
    O = Symmetry.from_generator(S, False)
    assert len(O) == 24
    assert Symmetry.cubic(False) == O
    # generators for a klein four subgroup V_4
    S = [[[-1,0,0], [0,-1,0], [0,0,1]],
        [[1,0,0], [0,-1,0], [0,0,-1]]]
    V_4 = Symmetry.from_generator(S, False)
    assert len(V_4) == 4
    # generators for the factor group ig
    S = [[[0,0,1], [0,-1,0], [1,0,0]],
        [[-1,0,0], [0,0,1], [0,1,0]]]
    F = Symmetry.from_generator(S, False)
    assert len(F) == 6
    assert Symmetry.even_perm3(False) == F
    assert V_4 * F == O
    assert O / V_4 == F
    assert O / Symmetry.mirror3(False) == F
    # 2D rotation
    S = [((0, 1), (-1, 0))]
    R = Symmetry.from_generator(S, False)
    assert Symmetry.o2() == R
    assert len(Symmetry.square() / Symmetry.o2()) == 2

    # test realize_symmetric
    neighbors = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    full_neighbors, order = Symmetry.cubic(True).realize_symmetric(neighbors)
    assert np.linalg.norm(np.linalg.norm(full_neighbors, axis=-1) - np.linalg.norm(np.asarray(neighbors)[order], axis=-1)) < 1e-7
    assert len(order) == 27, f"the symmetrization has yielded {len(order)} elements"
    full_neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    assert len(full_neighbors) == (27+1)//2, f"the symmetrization has yielded {len(full_neighbors)} elements"

    # test symmetrize
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mat2 = Symmetry.cubic(False).symmetrize(mat)
    assert np.all(mat2 == np.eye(3) * 5), "symmetrize with cubic symmetry failed"
    mat3 = Symmetry.mirror3(False).symmetrize(mat)
    assert np.all(mat3 == np.diag([1, 5, 9])), "symmetrize with mirror symmetry failed"

    # test conjugacy classes
    classes = Symmetry.cubic(True).conjugacy_classes()
    assert sorted([len(c) for c in classes]) == [1, 1, 3, 3, 6, 6, 6, 6, 8, 8]


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


def test_hamiltonian_symmetry():
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)
    UR = UnitaryRepresentation
    # make equivalent unitary representation and hamilton symmetry
    test_u_repr = UR.one_dim(True, False) + UR.d3(False) + UR.o3()
    assert test_u_repr.check_U()
    test_h_sym = HamiltonianSymmetry(Symmetry.cubic(True))
    # append all at rho=0 to compare with the unitary repr directly
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
    assert np.linalg.norm(test_H_r3 - test_H_r4) < 1e-7, "symmetrize and symmetrizer don't match"
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
