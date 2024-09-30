import numpy as np

from tight_binding_redweasel import (AsymTightBindingModel, BandStructureModel,
                                     Symmetry, TightBindingModel, urep, HermitianFourierSeries)


BANDS_N_K = 1000
SEED = 13**5

def test_bands_bands(benchmark):
    # generate k_smpl (random with fixed seed)
    np.random.seed(SEED)
    k_smpl = np.random.random((BANDS_N_K, 3))
    k_smpl[0] *= 0.0
    # initialize the models
    # Note, that the data has a hermitian first matrix, because that is a symmetry requirement.
    # If that first matrix is not hermitian, this test fails!
    band_model = BandStructureModel.load("tests/test_model.json")
    
    @benchmark
    def benchmarked_function():
        return band_model.bands(k_smpl)

def test_tb_bands(benchmark):
    # generate k_smpl (random with fixed seed)
    np.random.seed(SEED)
    k_smpl = np.random.random((BANDS_N_K, 3))
    k_smpl[0] *= 0.0
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
    
    @benchmark
    def benchmarked_function():
        return tb_model.bands(k_smpl)

def test_asym_tb_bands(benchmark):
    # generate k_smpl (random with fixed seed)
    np.random.seed(SEED)
    k_smpl = np.random.random((BANDS_N_K, 3))
    k_smpl[0] *= 0.0
    # initialize the models
    # Note, that the data has a hermitian first matrix, because that is a symmetry requirement.
    # If that first matrix is not hermitian, this test fails!
    asym_tb_model = AsymTightBindingModel.load("tests/test_model.json")
    
    @benchmark
    def benchmarked_function():
        return asym_tb_model.bands(k_smpl)


def optimize_example():
    # generate k_smpl (random with fixed seed)
    N = 5
    np.random.seed(SEED)
    k_smpl = np.random.random((N, 3))
    k_smpl[0] *= 0.0

    N_B = 3 # small for fast convergence
    band_weights = np.random.random(N_B) + 2.0
    #band_weights = band_weights * 0 + 1 # To test without bandweights during development

    neighbors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    H_r = np.random.random((len(neighbors), N_B, N_B)) + 0j
    H_r[0] += np.conj(H_r[0].T)

    ref_model = BandStructureModel.init_tight_binding(Symmetry.none(), neighbors, N_B, cos_reduced=False, exp=True)
    ref_model.set_params_complex(H_r)
    ref_bands = ref_model.bands(k_smpl)
    assert len(ref_bands[0]) == N_B
    assert len(band_weights) == N_B

    start_H_r = np.random.random((len(neighbors), N_B, N_B)) + 0j
    start_H_r[0] += np.conj(start_H_r[0].T)

    return neighbors, start_H_r, N_B, k_smpl, ref_bands, band_weights


def test_bands_optimizer(benchmark):
    # init the reference data
    neighbors, H_r, band_count, k_smpl, ref_bands, band_weights = optimize_example()
    
    @benchmark
    def benchmarked_function():
        # initialize the model
        band_model = BandStructureModel.init_tight_binding(Symmetry.none(), neighbors, band_count, cos_reduced=False, exp=True) # type: BandStructureModel
        band_model.set_params_complex(H_r)
        band_model.optimize(k_smpl, 1, ref_bands, band_weights, 0, 1000, log=False, use_pinv=True, max_accel_global=16.0, loss_threshold=1e-6)
        loss = band_model.loss(k_smpl, ref_bands, band_weights, 0)
        assert loss < 1e-6, f"not converged (loss: {loss:.2e})"
    # if I want the vscode ui to show proper times I have to fix the rounds...
    #benchmark.pedantic(benchmarked_function, iterations=1, rounds=20)


def test_asym_tb_optimizer_cg(benchmark):
    # init the reference data
    neighbors, H_r, band_count, k_smpl, ref_bands, band_weights = optimize_example()
    
    @benchmark
    def benchmarked_function():
        # initialize the model
        asym_tb_model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        log = asym_tb_model.optimize_cg(k_smpl, ref_bands, band_weights, 0, 100, log=False, max_cg_iterations=5, loss_threshold=1e-6)
        loss = asym_tb_model.loss(k_smpl, ref_bands, band_weights, 0)
        assert loss < 1e-6, f"not converged (loss: {loss:.2e})"

def test_asym_tb_optimizer(benchmark):
    # init the reference data
    neighbors, H_r, band_count, k_smpl, ref_bands, band_weights = optimize_example()
    
    @benchmark
    def benchmarked_function():
        # initialize the model
        asym_tb_model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        log = asym_tb_model.optimize(k_smpl, 1, ref_bands, band_weights, 0, 1000, log=False, max_accel_global=1.5, train_S=False, use_lstsq_stepsize=False, loss_threshold=1e-6)
        loss = asym_tb_model.loss(k_smpl, ref_bands, band_weights, 0)
        assert loss < 1e-6, f"not converged (loss: {loss:.2e})"


def test_asym_tb_optimizer_lstsq_stepsize(benchmark):
    # init the reference data
    neighbors, H_r, band_count, k_smpl, ref_bands, band_weights = optimize_example()
    
    @benchmark
    def benchmarked_function():
        # initialize the model
        asym_tb_model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        log = asym_tb_model.optimize(k_smpl, 1, ref_bands, band_weights, 0, 1000, log=False, max_accel_global=2.0, train_S=False, use_lstsq_stepsize=True, loss_threshold=1e-6)
        loss = asym_tb_model.loss(k_smpl, ref_bands, band_weights, 0)
        assert loss < 1e-6, f"not converged (loss: {loss:.2e})"

