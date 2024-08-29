# this file computes properties of the bulk material.
# meaning properties in the volume of the material, not the surface.

import numpy as np
from . import density_of_states as dos

eV = elementary_charge = 1.602176634e-19 # in Coulomb
hbar = 1.05457181764616e-34 # in SI J*s

# class that hold the data for bandstructure integrals together
# this is useful as a class, as it can reuse the same calculations for multiple integrals
class KIntegral:
    def __init__(self, dos_model, electrons, T, errors=False):
        self.errors = errors
        self.bandgap = dos_model.bandgap(electrons)
        self.metal = self.bandgap == 0.0
        self.model = dos_model.model
        self.mu = dos_model.chemical_potential(electrons, [T], N=50) # compute mu here with good enough precision
        self.beta = 1 / (dos.k_B * T) # in 1/eV
        self.e_smpl = []
        # prepared data for the integration
        self.k_smpl = []
        self.band_indices = []
        self.weights = []
        # function to collect the fermi surface data
        def collect_smpl(e_smpl):
            for e in e_smpl:
                k_smpl, band_indices, weights, _ = dos_model.fermi_surface_samples(e, improved=True, normalize=None)
                self.e_smpl.append(e)
                self.k_smpl.append(k_smpl)
                self.band_indices.append(band_indices)
                self.weights.append(weights)
            return e_smpl
        if self.metal:
            # this only really works for metals for low temperatures with smooth state density.
            dos.gauss_7_df(collect_smpl, self.mu, self.beta)
            if errors:
                # for estimating the error
                # - overestimated if the integral method is good
                # - randomly underestimated if the integral method is bad
                dos.gauss_5_df(collect_smpl, self.mu, self.beta)
        else:
            # for small temperatures:
            # do two gauss-laguerre integrations at the upper and lower bandgap boundary
            # for large temperatures:
            # do a normal gauss integration like for metals, but with a procedure with an even number of samples gauss_8_df, gauss_6_df
            # large/small temperature is defined by (self.bandgap * beta) small/large
            # TODO
            raise NotImplementedError("integrals for isolators (semiconductors) are currently not implemented")
        self.e_smpl = np.array(self.e_smpl)
        self.v = None # group velocities
        self.h = None # hessians
        self.w = None # weights divided by absolute group velocities
    
    # precompute/recompute the group velocities and hessians if hessians=True (False by default)
    def precompute(self, hessians=False):
        if hessians:
            self.v = []
            self.h = []
            self.w = []
            for k_smpl, band_indices, weights in zip(self.k_smpl, self.band_indices, self.weights):
                _, v, h = self.model.bands_grad_hess(k_smpl)
                v = np.take_along_axis(v, band_indices.reshape(-1, 1, 1), axis=-1)[:,:,0]
                self.v.append(v)
                self.h.append(np.take_along_axis(h, band_indices.reshape(-1, 1, 1, 1), axis=-1)[:,:,0])
                self.w.append(weights/(1e-8 + np.linalg.norm(v, axis=-1)))
        else:
            self.v = []
            self.w = []
            for k_smpl, band_indices, weights in zip(self.k_smpl, self.band_indices, self.weights):
                _, v = self.model.bands_grad(k_smpl)
                v = np.take_along_axis(v, band_indices.reshape(-1, 1, 1), axis=-1)[:,:,0]
                self.v.append(v)
                self.w.append(weights/(1e-8 + np.linalg.norm(v, axis=-1)))

    # integrate with the derivative of the fermi function as weight function
    # resulting unit: 1/eV * [f]
    def integrate_df(self, f, hessians=False, print_error=False):
        if self.v is None or (hessians and self.h is None):
            self.precompute(hessians)
        
        def int_e(e_smpl):
            res = []
            for e in e_smpl:
                # 1. find index of e in self.e_smpl
                index = np.argmin(np.abs(self.e_smpl - e))
                assert np.abs(self.e_smpl[index] - e) < 1e-8
                # 2. use data for that precomputed case
                if hessians:
                    f_res = f(e, self.v[index], self.h[index], self.k_smpl[index])
                    res.append(np.sum(f_res * self.w[index].reshape((-1,)+(1,)*len(np.shape(f_res)[1:])), axis=0))
                else:
                    f_res = f(e, self.v[index], self.k_smpl[index])
                    res.append(np.sum(f_res * self.w[index].reshape((-1,)+(1,)*len(np.shape(f_res)[1:])), axis=0))
            return res

        if self.metal:
            # this only really works for metals for low temperatures with smooth state density.
            I = dos.gauss_7_df(int_e, self.mu, self.beta)
            if self.errors and print_error:
                # for estimating the error
                # - overestimated if the integral method is good
                # - randomly underestimated if the integral method is bad
                I2 = dos.gauss_5_df(int_e, self.mu, self.beta)
                if len(np.shape(I)) <= 1:
                    print(f"error: {np.abs(I-I2)/I*100:f}%")
                if len(np.shape(I)) == 2:
                    print(f"error: {np.abs(np.trace(I-I2))/np.trace(I):%}")
        else:
            # TODO see comment in __init__
            raise NotImplementedError("integrals for isolators (semiconductors) are currently not implemented")
        return I
    
    # conductivity divided by the electron/phonon scattering time tau in 1/(Ohm*m*s)
    def conductivity_over_tau(self, cell_length, spin_factor=2, print_error=False):
        I = self.integrate_df(lambda _e, v, _k: v[:,None,:] * v[:,:,None], hessians=False, print_error=print_error)
        # TODO check this for non cubic structures
        k_unit = np.pi*2/cell_length # 1/m
        sigma = (spin_factor * elementary_charge**2/eV / cell_length**3 * (eV / k_unit / hbar)**2) * I # result is in 1/(Ohm*m)/s
        return sigma
    
    # electric part of the volumetric heat capacity c_V in J/m^3 (if cell_length is given in meters)
    # (This is better computed by the DoS itself)
    def heat_capacity(self, spin_factor=2, print_error=False):
        I = self.integrate_df(lambda e, _v, _k: e*(e - self.mu), hessians=False, print_error=print_error)
        cV_T = spin_factor * I * eV
        T = 1 / (self.beta * dos.k_B)
        return cV_T / T
    
    # electric part of the heat conductivity kappa in J/m^3 (if cell_length is given in meters)
    def heat_conductivity(self, cell_length, spin_factor=2, print_error=False):
        pass
