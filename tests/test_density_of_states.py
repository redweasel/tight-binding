import numpy as np

from tight_binding_redweasel import (BandStructureModel, Symmetry, dos)
from tight_binding_redweasel import bulk_properties as bulk

def test_cubic_density_of_states():
    t = 2.0

    #model = lambda k: t/2 * np.sum(np.cos(2*np.pi*k), axis=-1, keepdims=True)
    model = BandStructureModel.init_tight_binding(Symmetry.cubic(True), ((0,0,0), (0,0,1)), 1, exp=False)
    model.params[1] = t/48*3/2

    dos_model = dos.DensityOfStates(model, N=15) # 15 -> the minimum band value is not in the grid
    assert abs(dos_model.bands_range[0][0] - -3) < 1e-8, f"lower band range was wrong/inprecise: {dos_model.bands_range[0][0]}"
    assert abs(dos_model.bands_range[0][1] - 3) < 1e-8, f"upper band range was wrong/inprecise: {dos_model.bands_range[0][1]}"
    _e_smpl, n, _rho = dos_model.full_curve(N=40)
    # test monotonicity of n
    assert np.all((np.roll(n, -1) - n)[:-1] >= 0), "states(energy) was not monotone"

    electrons = 0.5
    assert abs(dos_model.fermi_energy(electrons)) < 1e-3, "computed fermi energy was wrong"


def test_linear_density_of_states():
    for offset_b in [-10, 0, 10]:
        offset_a = 1.0

        class LinearModel:
            def __init__(self):
                pass

            def __call__(self, k_smpl):
                return (np.sum(np.abs(k_smpl), axis=-1, keepdims=True) + offset_a) * [1, 2, 3, 4] + offset_b
            
            def bands_grad(self, k_smpl):
                base = np.ones_like(k_smpl) * np.sign(k_smpl)
                base = base[...,None]
                return self(k_smpl), base * [1, 2, 3, 4]

        linear = LinearModel()
        dos_model = dos.DensityOfStates(linear, N=11, ranges=((0.0, 0.5),)*3, wrap=False)
        assert np.linalg.norm(np.array(dos_model.bands_range)-offset_b - [(1.0, 2.5), (2.0, 5.0), (3.0, 7.5), (4.0, 10.0)]) < 1e-10, f"band ranges are detected wrong: {dos_model.bands_range}"
        energy_smpl, _states, density = dos_model.full_curve(N=30)

        a = offset_a * np.array([1, 2, 3, 4]) + offset_b
        c = [1, 2, 3, 4]
        # analytic state density
        density2 = []
        density_bands = []
        for e in energy_smpl:
            b = []
            for c_, a_ in zip(c, a):
                x = (e - a_) / c_
                lim = (x > 0.0) & (x < 1.5)
                rho = 1/abs(c_) * 8/2 * (x**2 - 3*max(0, x - 0.5)**2 + 3*max(0, x - 1)**2) * lim
                b.append(rho)
            density2.append(sum(b))
            density_bands.append(b)
        # test the contributions of the individual bands
        # error is due to the added epsilon in dos.cube_cut_volume
        assert np.linalg.norm(density2 - density) < 1e-6, "total dos doesn't match"
        assert np.linalg.norm([dos_model.density_bands(e) for e in energy_smpl] - np.array(density_bands)) < 1e-6, "band resolved dos doesn't match"

        # analytic drude factor
        T = 300
        beta = 1 / (dos.k_B * T)
        mu = dos_model.chemical_potential(2, [T], N=40)
        assert abs(mu[0] - (4.200556218616366+offset_b)) < 1e-6, "chemical potential was wrong"
        #print("chemical potential:", *mu)
        cell_length = 2.993e-10
        k_unit = np.pi*2/cell_length # 1/m
        def foo(mu):
            sigma_T0 = 0
            for c_, a_ in zip(c, a):
                x = (mu - a_) / c_
                lim = (x > 0.0) & (x < 1.5)
                rho_i = 1/np.abs(c_) * 8/2 * (x**2 - 3*np.maximum(0, x - 0.5)**2 + 3*np.maximum(0, x - 1)**2) * lim
                sigma_T0 += rho_i * c_**2 * (bulk.eV / k_unit / bulk.hbar)**2
            return sigma_T0
        sigma_T0 = dos.gauss_7_df(foo, mu, beta)
        
        sigma1 = (2 * bulk.elementary_charge**2/bulk.eV / cell_length**3) * sigma_T0
        sigma2 = bulk.KIntegral(dos_model, 2, T).drude_factor(np.eye(3)*cell_length*1e10, 2)[0,0]
        sigma3 = bulk.KIntegral(dos_model, 2, T).conductivity_over_tau(cell_length, 2)[0,0]
        assert abs(sigma1 - sigma2) / abs(sigma1) < 1e-4, f"error in sigma {(sigma1 - sigma2)/sigma1:%}"
        assert abs(sigma2 - sigma3) / abs(sigma2) < 1e-6, f"error in sigma {(sigma2 - sigma3)/sigma2:%}"
