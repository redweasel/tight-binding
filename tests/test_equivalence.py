# all models that are implemented in this package should
# produce the same output, if the same input data is loaded.

import numpy as np

from tight_binding_redweasel import (AsymTightBindingModel, BandStructureModel,
                                     Symmetry, TightBindingModel, urep)


def test_bands():
    # generate k_smpl (random with fixed seed)
    N = 10
    np.random.seed(13**5)
    k_smpl = np.random.random((N, 3))
    # initialize the models
    # Note, that the data has a hermitian first matrix, because that is a symmetry requirement.
    # If that first matrix is not hermitian, this test fails!
    band_model = BandStructureModel.load("tests/test_model.json")
    tb_model = TightBindingModel(
        urep.UnitaryRepresentation(Symmetry.none(), band_model.band_count()),
        band_model.neighbors,
        None
    )
    tb_model.set_from_complex(band_model.params_complex())
    asym_tb_model = AsymTightBindingModel.load("tests/test_model.json")
    #np.set_printoptions(precision=3, suppress=True, linewidth=10000)
    #print(band_model.ddf([[0.1, 0.1, 0.1]])[0, :5, :5] - tb_model.ddf([[0.1, 0.1, 0.1]])[0, :5, :5])
    #print(band_model.ddf([[0., 0., 0.]])[0, :5, :5] - tb_model.ddf([[0., 0., 0.]])[0, :5, :5])
    # TODO compare with numerical derivatives!!!
    # compute the bandstructure and all available derivatives using all models
    bands1, grads1, hess1 = band_model.bands_grad_hess(k_smpl)
    bands2, grads2, hess2 = tb_model.bands_grad_hess(k_smpl)
    bands3, grads3, hess3 = asym_tb_model.bands_grad_hess(k_smpl)
    # now compute the differences
    bands_norm = np.linalg.norm(bands1) # trust bands1 the most here (not that important, just for display)
    bands_diff_12 = np.linalg.norm(bands1 - bands2) / bands_norm
    bands_diff_23 = np.linalg.norm(bands2 - bands3) / bands_norm
    bands_diff_13 = np.linalg.norm(bands3 - bands1) / bands_norm
    grads_norm = np.linalg.norm(grads1) # trust bands1 the most here (not that important, just for display)
    grads_diff_12 = np.linalg.norm(grads1 - grads2) / grads_norm
    grads_diff_23 = np.linalg.norm(grads2 - grads3) / grads_norm
    grads_diff_13 = np.linalg.norm(grads3 - grads1) / grads_norm
    hess_norm = np.linalg.norm(hess1) # trust bands1 the most here (not that important, just for display)
    hess_diff_12 = np.linalg.norm(hess1 - hess2) / hess_norm
    hess_diff_23 = np.linalg.norm(hess2 - hess3) / hess_norm
    hess_diff_13 = np.linalg.norm(hess3 - hess1) / hess_norm
    # now check if any is above 1e-8 percent, which is the precision of the eigenvalue algorithms
    max_error = 1e-8
    fail = False
    fail |= bands_diff_12 > max_error
    fail |= bands_diff_23 > max_error
    fail |= bands_diff_13 > max_error
    fail |= grads_diff_12 > max_error
    fail |= grads_diff_23 > max_error
    fail |= grads_diff_13 > max_error
    fail |= hess_diff_12 > max_error
    fail |= hess_diff_23 > max_error
    fail |= hess_diff_13 > max_error
    if fail:
        print("errors |   1-2   |   2-3   |   3-1")
        print(f"bands  | {bands_diff_12:7.2e} | {bands_diff_23:7.2e} | {bands_diff_13:7.2e}")
        print(f"grads  | {grads_diff_12:7.2e} | {grads_diff_23:7.2e} | {grads_diff_13:7.2e}")
        print(f"hess   | {hess_diff_12:7.2e} | {hess_diff_23:7.2e} | {hess_diff_13:7.2e}")
        raise ValueError("Test failed, errors are printed to stdout")


def test_loss():
    # generate k_smpl (random with fixed seed)
    N = 10
    N_B = 5
    np.random.seed(13**5)
    k_smpl = np.random.random((N, 3))
    k_smpl[0] *= 0.0
    ref_bands = np.random.random((N, N_B))
    band_weights = np.random.random(N_B) + 1.0
    # initialize the models
    # Note, that the data has a hermitian first matrix, because that is a symmetry requirement.
    # If that first matrix is not hermitian, this test fails!
    # test loss functions of all models, as they too should be equivalent in the most basic form
    neighbors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    band_model = BandStructureModel.init_tight_binding_from_ref(Symmetry.none(), neighbors, k_smpl, ref_bands)
    tb_model = TightBindingModel(
        urep.UnitaryRepresentation(Symmetry.none(), band_model.band_count()),
        band_model.neighbors,
        None
    )
    tb_model.set_from_complex(band_model.params_complex())
    asym_tb_model = AsymTightBindingModel.init_from_ref(neighbors, k_smpl, ref_bands, use_S=False)
    asym_tb_model.set_from_complex(band_model.params_complex())
    # now compute the loss and max_err of all 3 models and compare
    loss1, max_err1 = band_model.error(k_smpl, ref_bands, band_weights, 0)
    loss2, max_err2 = tb_model.error(k_smpl, ref_bands, band_weights, 0)
    loss3, max_err3 = asym_tb_model.error(k_smpl, ref_bands, band_weights, 0)
    loss1_ = band_model.loss(k_smpl, ref_bands, band_weights, 0)
    loss2_ = tb_model.loss(k_smpl, ref_bands, band_weights, 0)
    loss3_ = asym_tb_model.loss(k_smpl, ref_bands, band_weights, 0)
    assert loss1_ == loss1, "inconsistent loss in BandStructureModel"
    assert loss2_ == loss2, "inconsistent loss in TightBindingModel"
    assert loss3_ == loss3, "inconsistent loss in AsymTightBindingModel"
    # now compute differences in max_err
    max_err_norm = np.linalg.norm(max_err1)
    max_err_diff_12 = np.linalg.norm(max_err1 - max_err2) / max_err_norm
    max_err_diff_23 = np.linalg.norm(max_err2 - max_err3) / max_err_norm
    max_err_diff_13 = np.linalg.norm(max_err3 - max_err1) / max_err_norm
    max_error = 1e-8
    fail = False
    fail |= abs(loss1 - loss2) > max_error
    fail |= abs(loss2 - loss3) > max_error
    fail |= abs(loss3 - loss1) > max_error
    fail |= max_err_diff_12 > max_error
    fail |= max_err_diff_23 > max_error
    fail |= max_err_diff_13 > max_error
    if fail:
        print(f"loss was: {loss1:.3e}, {loss2:.3e}, {loss3:.3e}")
        print("errors |   1-2   |   2-3   |   3-1")
        print(f"loss   | {abs(loss1 - loss2):7.2e} | {abs(loss2 - loss3):7.2e} | {abs(loss3 - loss1):7.2e}")
        print(f"maxerr | {max_err_diff_12:7.2e} | {max_err_diff_23:7.2e} | {max_err_diff_13:7.2e}")
        raise ValueError("Test failed, errors are printed to stdout")


def test_bandstructure_params():
    neighbors = ((0, 0, 0), (1, 0, 0))#, (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    k_smpl = np.array([(0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.4, 0.5, 0.6), (0.6, 0.7, 0.8), (0.7, 0.8, 0.9)])
    k_smpl, _ = Symmetry.cubic(True).realize_symmetric(k_smpl)

    for sym in [Symmetry.none(), Symmetry.inv()]:
        tb00 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=False, exp=False) # 0
        tb01 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=False, exp=True)  # 1
        tb10 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=True,  exp=False) # 2
        tb11 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=True,  exp=True)  # 3
        tb = [tb00, tb01, tb10, tb11]

        # set one model and convert to normal H_r matrices and apply them to the other models
        # then test if the hamiltonians at some non symmetric k-points are the same
        tb_other = list(enumerate(tb)) # even check with itself
        for i in range(len(tb)):
            tb_test = tb[i]
            tb_test.params = np.random.standard_normal(tb_test.params.shape) + 1j*np.random.standard_normal(tb_test.params.shape)
            tb_test.normalize() # fixes non symmetric matrices from random start
            tb_test_f = tb_test.f(k_smpl)
            tb_test_df = tb_test.df(k_smpl)
            tb_test_ddf = tb_test.ddf(k_smpl)
            #print()
            #print("first params", np.ravel(tb_test.params))
            #print(tb_test.f([(0,0,0)]), tb_test.f([(0,0,1/2)]), tb_test.f([(0,0,1/4)]))
            H_r = tb_test.params_complex()
            #print("first H_r", np.ravel(H_r))
            for j, tb_test2 in tb_other:
                tb_test2.set_params_complex(H_r)
                #print("tb_test2", np.ravel(tb_test2.params))
                #print(tb_test2.f([(0,0,0)]), tb_test2.f([(0,0,1/2)]), tb_test2.f([(0,0,1/4)]))
                assert np.linalg.norm(tb_test_f - tb_test2.f(k_smpl)) < 1e-7, f"reconstructed Hamiltonians don't match ({i} vs {j}, inversion: {sym.inversion})"
                assert np.linalg.norm(tb_test_df - tb_test2.df(k_smpl)) < 1e-7, f"reconstructed Hamiltonian 1. derivative doesn't match ({i} vs {j}, inversion: {sym.inversion})"
                assert np.linalg.norm(tb_test_ddf - tb_test2.ddf(k_smpl)) < 1e-7, f"reconstructed Hamiltonian 2. derivative doesn't match ({i} vs {j}, inversion: {sym.inversion})"
                H_r2 = tb_test2.params_complex()
                assert np.linalg.norm(H_r - H_r2) < 1e-14, f"Reconstruction doesn't match ({i} vs {j}, inversion: {sym.inversion})"


if __name__ == "__main__":
    test_bands()
    test_loss()