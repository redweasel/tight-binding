 
import numpy as np

from tight_binding_redweasel import (AsymTightBindingModel, BandStructureModel,
                                     Symmetry, TightBindingModel, urep, HermitianFourierSeries)


def test_bandstructure_optimize():
    neighbors = ((0, 0, 0), (1, 0, 0))#, (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    sym = Symmetry.none()
    n = 5
    tb00 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=False, exp=False) # 0
    tb01 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=False, exp=True)  # 1
    tb10 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=True,  exp=False) # 2
    tb11 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=True,  exp=True)  # 3
    tb = [tb00, tb01, tb10, tb11]

    for k_count in [1]:
        # TODO use k_count
        k_smpl = np.array([(0.1, 0.2, 0.3),])
        ref_bands = np.arange(n).astype(np.float64)[None,:]

        # test whether a single optimize step solves the single k case
        for i, tb in enumerate(tb):
            for kw in [1, 1.5]:
                for bw in [1, 1.5]:
                    tb.params = np.random.random(tb.params.shape)
                    tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, log=False, use_pinv=False)
                    assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} without pinv was incorrect for model {i}, kw = {kw}, bw = {bw}"
                    tb.params = np.random.random(tb.params.shape)
                    tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, log=False, use_pinv=True)
                    assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} with pinv was incorrect for model {i}, kw = {kw}, bw = {bw}"


def test_asym_tight_binding_optimize():
    neighbors = ((0, 0, 0), (1, 0, 0))#, (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    n = 5
    tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, np.zeros((len(neighbors), n, n))))

    for k_count in [1]:
        # TODO use k_count
        k_smpl = np.array([(0.1, 0.2, 0.3),])
        ref_bands = np.arange(n).astype(np.float64)[None,:]

        # test whether a single optimize step solves the single k case
        for kw in [1, 1.5]:
            for bw in [1, 1.5]:
                for pinv in [False, True]:
                    tb.set_from_complex(np.random.random(tb.params_complex().shape))
                    tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, use_pinv=pinv, use_lstsq_stepsize=False, log=False)
                    assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} with pinv={pinv} was incorrect, kw = {kw}, bw = {bw}"



if __name__ == "__main__":
    test_bandstructure_optimize()
    test_asym_tight_binding_optimize()