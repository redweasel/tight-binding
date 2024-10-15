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
    e_smpl, n, rho = dos_model.full_curve(N=40)
    # test monotonicity of n
    assert np.all((np.roll(n, -1) - n)[:-1] >= 0), "states(energy) was not monotone"

    electrons = 0.5
    assert abs(dos_model.fermi_energy(electrons)) < 1e-3, "computed fermi energy was wrong"

    # test the other functions to get the density and states. They should be equivalent
    assert np.linalg.norm([dos_model.states_below(e) for e in e_smpl] - n) < 1e-7, "states_below was different from full_curve"
    assert np.linalg.norm([dos_model.density(e) for e in e_smpl] - rho) < 1e-7, "density was different from full_curve"
    assert np.linalg.norm([dos_model.states_density(e)[0] for e in e_smpl] - n) < 1e-7, "states_below was different from full_curve"
    assert np.linalg.norm([dos_model.states_density(e)[1] for e in e_smpl] - rho) < 1e-7, "density was different from full_curve"


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
        dos_model = dos.DensityOfStates(linear, N=17, ranges=((0.0, 0.5),)*3, wrap=False)
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
        assert np.linalg.norm(density2 - density) < 1e-12, f"total dos doesn't match, error {np.linalg.norm(density2 - density)}"
        density_bands2 = np.array([dos_model.density_bands(e) for e in energy_smpl])
        assert np.linalg.norm(density_bands2 - density_bands) < 1e-12, f"band resolved dos doesn't match, error {np.linalg.norm(density_bands2 - density_bands)}"
        # TODO compare states as well!

        # test area measure in fermi_surface_samples
        for weight_by_gradient in [False, True]:
            _, _, _, area = dos_model.fermi_surface_samples(offset_b + 2*offset_a, improved_points=False, improved_weights=False, weight_by_gradient=weight_by_gradient, normalize=None)
            assert abs(area - offset_a**2*3**.5) < 1e-6, f"area of fermi_surface_samples is wrong, {area} != {offset_a**2*3**.5}"
            _, _, _, area = dos_model.fermi_surface_samples(offset_b + 2*offset_a, improved_points=True, improved_weights=False, weight_by_gradient=weight_by_gradient, normalize=None)
            assert abs(area - offset_a**2*3**.5) < 1e-6, f"area of fermi_surface_samples is wrong, {area} != {offset_a**2*3**.5}"
            _, _, _, area = dos_model.fermi_surface_samples(offset_b + 2*offset_a, improved_points=False, improved_weights=True, weight_by_gradient=weight_by_gradient, normalize=None)
            assert abs(area - offset_a**2*3**.5) < 1e-6, f"area of fermi_surface_samples is wrong, {area} != {offset_a**2*3**.5}"
            _, _, _, area = dos_model.fermi_surface_samples(offset_b + 2*offset_a, improved_points=True, improved_weights=True, weight_by_gradient=weight_by_gradient, normalize=None)
            assert abs(area - offset_a**2*3**.5) < 1e-6, f"area of fermi_surface_samples is wrong, {area} != {offset_a**2*3**.5}"
        # TODO test more!

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
        sigma2 = bulk.KIntegral(dos_model, 2, T).drude_factor(np.eye(3)*cell_length*1e10, 2)[0][0,0]
        sigma3 = bulk.KIntegral(dos_model, 2, T).conductivity_over_tau(cell_length, 2)[0,0]
        assert abs(sigma1 - sigma2) / abs(sigma1) < 1e-4, f"error in sigma {(sigma1 - sigma2)/sigma1:%}"
        assert abs(sigma2 - sigma3) / abs(sigma2) < 1e-6, f"error in sigma {(sigma2 - sigma3)/sigma2:%}"

test_linear_density_of_states()