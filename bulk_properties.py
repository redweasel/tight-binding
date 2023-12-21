# this file computes properties of the bulk material.
# meaning properties in the volume of the material, not the surface.

import numpy as np
import density_of_states as dos

eV = elementary_charge = 1.602176634e-19 # in Coulomb
hbar = 1.05457181764616e-34 # in SI J*s

# compute the conductivity with an averaged scattering time tau
# the result of this needs to be multiplied with that tau (in seconds (SI)) to get the conductivity tensor (in SI)
# to use this function, dos_model needs to be based on a model with the function .bands_grad(k)
# TODO test more...
def conductivity_over_tau(dos_model, electrons, cell_length, T, mu=None, print_error=False):
    # calculate conductivity (Czycholl page 269)
    if mu is None:
        mu = dos_model.chemical_potential(electrons, [T], N=50)
    beta = 1 / (dos.k_B * T) # in 1/eV
    # TODO check this for non cubic structures
    k_unit = np.pi/2/cell_length # 1/m
    def int_e(e_smpl):
        res = []
        for e in e_smpl:
            k_smpl, indices, weights, _ = dos_model.fermi_surface_samples(e, improved=True, normalize=None)
            _, v = dos_model.model.bands_grad(k_smpl)
            v = np.take_along_axis(v, indices.reshape(-1, 1, 1), axis=-1)[:,:,0]
            res.append(np.einsum("ia,ib,i->ab", v, v, weights/(1e-8 + np.linalg.norm(v, axis=-1))))
        return np.array(res)
    # this only really works for metals for low temperatures with smooth state density.
    I = dos.gauss_7_df(int_e, mu, beta)
    if print_error:
        # for estimating the error
        # - overestimated if the integral method is good
        # - randomly underestimated if the integral method is bad
        I2 = dos.gauss_5_df(int_e, mu, beta)
        print(f"error: {np.abs(np.trace(I-I2))/np.trace(I):%}")
    sigma = (2 * elementary_charge**2/eV / cell_length**3 * (eV / k_unit / hbar)**2) * I # result is in 1/(Ohm*m)
    #avg_sigma = np.trace(sigma)/3
    #print(f"{sigma/avg_sigma}*{avg_sigma:.3e}\nwith error\n{err/np.abs(I)*100}%")
    return sigma
