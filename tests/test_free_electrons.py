# testing the methods by comparing to analytical results of the free electron model.

import numpy as np

from tight_binding_redweasel import (BandStructureModel, Symmetry, dos, bulk)

def test_dos():
    free_electron_model = bulk.FreeElectrons(k_neighbors=[(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,-1), (0,-1,0), (-1,0,0)])
    e = 30.0
    e_f = 36.4645006 # eV for N=1 electron (hbar^2/m_e/2×(3/8/pi)^(2/3)×(2×pi/angstrom)^2/1eV)
    states = 0.5*(e/e_f)**1.5
    density = 0.5*1.5*e_f**-1.5*e**.5
    seebeck_300 = -1.00495186e-7  # V/K for N=1 electron (-k_B^2×pi^2/6/elementary_charge/(hbar^2/m_e/2×(3/8/pi)^(2/3))×300K/(2×pi/angstrom)^2)
    c_v = 0
    for N, err in [(16, 0.02), (8, 0.08)]: # (32, 0.005),
        def models():
            yield False, dos.DensityOfStates(free_electron_model, N=N, ranges=3*((0,0.5),), wrap=False, use_gradients=False, smearing="cubes", check=False)
            yield False, dos.DensityOfStates(free_electron_model, N=N+1, use_gradients=False, smearing="cubes", check=False)
            yield True, dos.DensityOfStates(free_electron_model, N=N, use_gradients=True, smearing="cubes", check=False)
        for grads, dos_model in models():
            # both of the below give O(N^(-2)) for free electrons
            s, d = dos_model.states_density(e)
            assert abs(s - states)/states < err * 4
            assert abs(d - density)/density < err * 4
            # fermi energy
            assert abs(dos_model.fermi_energy(0.5) - e_f)/e_f < err
            # mu for some temperatures
            T_smpl = np.linspace(0, 10000, 5)
            mu_smpl = e_f*(1 - np.pi**2/12*(dos.k_B*T_smpl/e_f)**2)
            assert np.linalg.norm((dos_model.chemical_potential(0.5, T_smpl, N=10) - mu_smpl)/mu_smpl) < err * 4
            if not grads:
                seebeck_smpl = T_smpl * seebeck_300/300
                assert np.linalg.norm((dos_model.seebeck(0.5, T_smpl, N=10) - seebeck_smpl)/(seebeck_smpl + 1e-8)) < err * 20
            # energy and heat capacity
            c_v = np.pi**2/3 * dos.k_B**2 * 2*density * T_smpl
            total_energy = 21.8787004 + c_v/2 * T_smpl
            assert np.linalg.norm(([dos_model.energy(mu, T, N=10) for mu, T in zip(mu_smpl, T_smpl)] - total_energy)/total_energy) < err * 4
            # Note, this error is way to high and doesn't really converge yet...
            assert np.linalg.norm(([dos_model.heat_capacity(0.5, T, N=20) for T in T_smpl] - c_v)/(c_v + 1e-8)) < err * 400

def test_bulk():
    free_electron_model = bulk.FreeElectrons(k_neighbors=[(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,-1), (0,-1,0), (-1,0,0)])
    T = 300.0
    dos_model = dos.DensityOfStates(free_electron_model, N=32, check=False)
    N_list = [5, 7, 11, 13, 15, 17]
    for N in N_list:
        kint = bulk.KIntegral(dos_model, 0.5, [T], N=N)
        L_a = kint.transport_coefficients(np.eye(3))
        assert abs(np.trace(bulk.seebeck(L_a, T)[0])/3 - 3*1.00495186e-7) / (3*1.00495186e-7) < 0.03

