# this file computes properties of the bulk material.
# meaning properties in the volume of the material, not the surface.

import numpy as np
from typing import Callable
from . import density_of_states as _dos

elementary_charge = 1.602176634e-19 # in Coulomb
eV = elementary_charge # in Joule
hbar = 1.05457181764616e-34 # in SI J*s
c = 299792458.0 # speed of light in m/s

# TODO try to find it in numpy
antisym_tensor = np.array([
    [[0, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    [[0, 0, -1],
     [0, 0, 0],
     [1, 0, 0]],
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, 0]]
])

class KIntegral:
    """
    This class bundles the data for bandstructure integrals (k-space).
    This is useful as a class, as it can reuse the same calculations for multiple integrals.

    The integrals are of different forms, so look for the `integrate..` functions for details.
    """

    def __init__(self, dos_model: _dos.DensityOfStates, electrons: float, T: float, errors=False):
        """Initialize an Integral over the bandstructure (k-space) 

        Args:
            dos_model (DensityOfStates): Density of states (DoS) model which contains all important information.
            electrons (int): Number of electrons/filled bands in the model.
            T (float): Temperature at which the integral is computed.
            errors (bool, optional): If True, a second less precise integration will be done to compute the error of the result. Defaults to False.
        """
        self.errors = errors
        self.bandgap = dos_model.bandgap(electrons)
        self.metal = self.bandgap == 0.0
        self.model = dos_model.model
        self.A = dos_model.A
        self.mu = dos_model.chemical_potential(electrons, [T], N=50)[0] # compute mu here with good enough precision
        self.beta = 1 / (_dos.k_B * T) if T > 0 else 0.0 # in 1/eV
        self.e_smpl = []
        # prepared data for the integration
        self.k_smpl = []
        self.band_indices = []
        self.weights = []
        # function to collect the fermi surface data
        # TODO allow the use of the dos grid instead of the exact fermi surface
        # -> faster, less precise
        improved_points = False # True makes it much more robust -> extrapolation possible, however it doesn't increase the convergence order
        def collect_smpl(e_smpl):
            for e in e_smpl:
                k_smpl, band_indices, weights, _ = dos_model.fermi_surface_samples(e, improved_points=improved_points, improved_weights=False, normalize=None)
                self.e_smpl.append(e)
                self.k_smpl.append(k_smpl)
                self.band_indices.append(band_indices)
                self.weights.append(weights)
            return e_smpl
        if T > 0:
            if self.metal:
                # this only really works for metals for low temperatures with smooth state density.
                _dos.gauss_7_df(collect_smpl, self.mu, self.beta)
                if errors:
                    # for estimating the error
                    # - overestimated if the integral method is good
                    # - randomly underestimated if the integral method is bad
                    _dos.gauss_5_df(collect_smpl, self.mu, self.beta)
            else:
                # do two (3 point and 2 point) Gauss-Laguerre integrations at the upper and lower bandgap boundary
                # TODO
                raise NotImplementedError("integrals for isolators (semiconductors) are currently not implemented")
        else:
            assert T == 0, "No negative temperatures allowed"
            if self.metal:
                # just the Fermi-surface
                collect_smpl([self.mu])
            else:
                # isolator at 0 temperature has no Fermi-surface -> all integrals 0
                pass
        self.e_smpl = np.array(self.e_smpl)
        self.v = None # group velocities
        self.h = None # hessians
        self.w = None # weights divided by absolute group velocities
    
    def precompute(self, hessians=False):
        """Precompute/recompute the group velocities and hessians if needed.
        This function does not need to be called manually.
        It will be called as needed by the integrate functions.
        However note, that it will be called twice if the first integration has no hessians and the second has.
        In that case this function can be used to compute the needed hessians manually,
        before the first integration. The integration will not recompute this.

        Args:
            hessians (bool, optional): If True, the Hessian matrices (inverse mass tensors) will also be computed. Defaults to False.
        """
        if hessians:
            self.v = []
            self.h = []
            self.w = []
            for k_smpl, band_indices, weights in zip(self.k_smpl, self.band_indices, self.weights):
                _, v, h = self.model.bands_grad_hess(k_smpl)
                v = np.take_along_axis(v, band_indices.reshape(-1, 1, 1), axis=-1)[...,0]
                self.v.append(v)
                self.h.append(np.take_along_axis(h, band_indices.reshape(-1, 1, 1, 1), axis=-1)[...,0])
                self.w.append(weights/(1e-8 + np.linalg.norm(v, axis=-1)))
        else:
            self.v = []
            self.w = []
            for k_smpl, band_indices, weights in zip(self.k_smpl, self.band_indices, self.weights):
                _, v = self.model.bands_grad(k_smpl)
                v = np.take_along_axis(v, band_indices.reshape(-1, 1, 1), axis=-1)[...,0]
                self.v.append(v)
                self.w.append(weights/(1e-8 + np.linalg.norm(v, axis=-1)))

    def integrate_df(self, g: Callable, hessians=False):
        """Integrate with the derivative of the fermi function as weight function.

        I = sum_n 1/V_RZ integral_{V_RZ}( g(E_n(k), grad_k E_n(k), hess_k E_n(k), k) * (-df/de(e, mu, T)) dk)

        resulting unit: 1/eV * [unit of g]

        Args:
            g (Callable[[bandenergy: float, group_velocity, (inverse_mass_tensor), k], arraylike]):
                The function to be integrated. It can use
                the bandenergy, the energy gradients (~ group velocity),
                the energy hessians (~ inverse mass tensor) and the position in k-space.
                Note that the positions in k-space are given in crystal coordinates, meaning they are in the cube [-0.5,0.5]^3.
                This affects the gradients and hessians as well. Make sure to convert to proper k-space for the gradients.
            hessians (bool, optional): If True, the hessians are given to the function g, otherwise the (positional) argument is ommited. Defaults to False.

        Returns:
            skalar: The value of the integral.
            float: The absolute error of the integral, if available, otherwise 0.0.
                   This error is just the error if the integration. The larger error usually comes from the DensityOfStates.
        """
        if self.v is None or (hessians and self.h is None):
            self.precompute(hessians)
        
        def int_e(e_smpl):
            res = []
            for e in e_smpl:
                # 1. find index of e in self.e_smpl
                index = np.argmin(np.abs(self.e_smpl - e))
                assert np.abs(self.e_smpl[index] - e) < 1e-8, "internal error, precomputed energies don't match used energies"
                # 2. use data for that precomputed case
                if hessians:
                    f_res = g(e, self.v[index], self.h[index], self.k_smpl[index])
                    res.append(np.sum(f_res * self.w[index].reshape((-1,)+(1,)*len(np.shape(f_res)[1:])), axis=0))
                else:
                    f_res = g(e, self.v[index], self.k_smpl[index])
                    res.append(np.sum(f_res * self.w[index].reshape((-1,)+(1,)*len(np.shape(f_res)[1:])), axis=0))
            return res

        error = 0.0
        if self.beta == 0:
            # zero temperature case
            if self.metal:
                I = int_e([self.mu])
                assert len(I) == 1
                I = I[0]
                # TODO error estimation
            else:
                # no fermi surface!
                # to get the output shape right, evaluate g at one point
                if hessians:
                    I = 0.0 * np.asanyarray(g(self.mu, np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3))))
                else:
                    I = 0.0 * np.asanyarray(g(self.mu, np.zeros((1, 3)), np.zeros((1, 3))))
        else:
            if self.metal:
                # this only really works for metals for low temperatures with smooth state density.
                I = _dos.gauss_7_df(int_e, self.mu, self.beta)
                if self.errors:
                    # for estimating the error
                    # - overestimated if the integral method is good
                    # - randomly underestimated if the integral method is bad
                    I2 = _dos.gauss_5_df(int_e, self.mu, self.beta)
                    error = np.abs(I - I2)
            else:
                # TODO see comment in __init__
                raise NotImplementedError("integrals for isolators (semiconductors) are currently not implemented")
        return I, error
    
    def integrate_df_A(self, A, g: Callable, hessians=False):
        """Integrate with the derivative of the fermi function as weight function.
        In contrast to `integrate_df`, this function uses the correct k-space.

        I = sum_n 1/V_RZ integral_{V_RZ}( g(E_n(k), grad_k E_n(k), hess_k E_n(k), k) * (-df/de(e, mu, T)) dk)

        resulting unit: 1/eV * [unit of g]

        Args:
            A (matrix): lattice vectors in the columns of this row-major matrix in Ångström = 1e-10m.
            g (Callable[[bandenergy: float, group_velocity, (inverse_mass_tensor), k], arraylike]):
                The function to be integrated. It can use
                the bandenergy, the group velocity,
                the inverse mass tensor and the position in k-space.
                Note that the positions in k-space are given in actual reciprocal space coordinates in 2π/Å, computed using A.
                The group velocity is in m/s and the inverse_mass_tensor is in 1/kg
            hessians (bool, optional): If True, the hessians are given to the function g, otherwise the (positional) argument is ommited. Defaults to False.

        Returns:
            skalar: The value of the integral.
            float: The absolute error of the integral, if available, otherwise 0.0.
                   This error is just the error if the integration. The larger error usually comes from the DensityOfStates.
        """
        # consider DoS space matrix as an already applied A matrix
        A = A @ np.linalg.inv(self.A)
        # reciprocal space vectors
        B = 2*np.pi * np.linalg.inv(A.T)
        # transform the derivative, which live in the dual vector space -> dual transformation
        # this transformation also includes the unit conversion using Ångström and hbar in SI.
        # The unit of energy needs to be converted to Joule down below as well.
        grad_transform = (1/(2*np.pi) * 1e-10 / hbar) * A
        if hessians:
            def g2(e, v, h, k):
                k = np.einsum("ji,ni->nj", B, k)
                v = np.einsum("ji,ni->nj", grad_transform*eV, v)
                h = np.einsum("ki,lj,nij->nkl", grad_transform*eV, grad_transform, h)
                res = g(e, v, h, k)
                assert len(res) == len(k)
                return res
            value, error = self.integrate_df(g2, hessians=True)
        else:
            def g2(e, v, k):
                k = np.einsum("ji,ni->nj", B, k)
                v = np.einsum("ji,ni->nj", grad_transform*eV, v)
                res = g(e, v, k)
                assert len(res) == len(k)
                return res
            value, error = self.integrate_df(g2, hessians=False)
        return value, error
    
    # conductivity divided by the electron scattering time tau. Assuming constant tau. Result in 1/(Ohm*m*s)
    def conductivity_over_tau(self, cell_length: float, spin_factor=2):
        I, error = self.integrate_df(lambda _e, v, _k: v[:,None,:] * v[:,:,None], hessians=False)
        # TODO check this for non cubic structures
        k_unit = np.pi*2/cell_length # 1/m
        sigma = (spin_factor * elementary_charge**2/eV / cell_length**3 * (eV / k_unit / hbar)**2) * I # result is in 1/(Ohm*m)/s
        return sigma
    
    # conductivity divided by the electron scattering time tau. Assuming constant tau. Result in 1/(Ohm*m*s)
    def drude_factor(self, A, spin_factor=2):
        V_EZ = np.linalg.det(A*1e-10)
        I, error = self.integrate_df_A(A, lambda _e, v, _k: v[:,None,:] * v[:,:,None], hessians=False)
        D = spin_factor * elementary_charge**2 * I/eV / V_EZ
        return D, D/I*error # result is in 1/(Ohm*m)/s
    
    # electric part of the volumetric heat capacity c_V in J/m^3
    # (This is better computed by the DoS itself)
    def heat_capacity(self, spin_factor=2):
        I, error = self.integrate_df(lambda e, _v, _k: e*(e - self.mu), hessians=False)
        cV_T = spin_factor * I * eV
        T = 1 / (self.beta * _dos.k_B)
        return cV_T / T
    
    def hall_coefficient(self, A, spin_factor=2):
        # PhysRevB.82.035103 Hall effect formula with constant relaxation time for all bands over all k.
        # In reality the relaxation times are different over k-space or even just for different spins in the same band. (PhysRev.97.647)
        # Somehow their units don't match the expected result unit, so I removed the division by the speed of light to make it work.
        # I canceled one e from sigma_xx with e^2 from sigma_xyz
        # TODO remove! This is bad!
        V_EZ = np.linalg.det(A*1e-10)
        #sigma_xyz = elementary_charge/c/eV * self.integrate_df_A(A, lambda _e, v, h, _k: (v[:,0]**2*h[:,1,1] - v[:,0]*v[:,1]*h[:,1,0]), hessians=True)
        I, error = self.integrate_df_A(A, lambda _e, v, h, _k: (v[:,0]**2*h[:,1,1] - v[:,0]*v[:,1]*h[:,1,0]), hessians=True)
        I2, error2 = self.integrate_df_A(A, lambda _e, v, _k: v[:,0]**2, hessians=False)
        sigma_xyz = -elementary_charge/eV * spin_factor * I
        sigma_xx = elementary_charge/eV * spin_factor * I2
        R_H = sigma_xyz / sigma_xx**2 * V_EZ
        R_H_error = R_H * ((error/I)**2 + (2*error2/I2)**2)**.5
        return R_H, R_H_error # in m^3/C = Ohm m/T
    
    def hall_coefficient_metal_cubic(self, A, spin_factor=2):
        # PhysRevB.45.10886 Hall effect formula with constant relaxation time over k.
        # In reality the relaxation times are different over k-space or even just for different spins in the same band. (PhysRev.97.647)
        # I canceled one e from sigma_0 with e^2 from sigma_H
        V_EZ = np.linalg.det(A*1e-10)
        I, error = self.integrate_df_A(A, lambda _e, v, h, _k: np.einsum("ni,nij,nj->n", v, h, v) - np.einsum("nj,nj,nii->n", v, v, h), hessians=True)
        I2, error2 = self.integrate_df_A(A, lambda _e, v, _k: (v**2).sum(-1), hessians=False)
        sigma_H = elementary_charge/eV * spin_factor/6 * I
        sigma_0 = elementary_charge/eV * spin_factor/3  * I2
        R_H = sigma_H / sigma_0**2 * V_EZ
        R_H_error = R_H * ((error/I)**2 + (2*error2/I2)**2)**.5
        return R_H, R_H_error # in m^3/C = Ohm m/T
    
    def conductivity_hall_tensor(self, A, spin_factor=2):
        # From the book "C. Hurd, The Hall Coeffcient of Metals and Alloys (Plenum, New York, 1972)"
        # Assuming constant relaxation time for now -> independent of it
        V_EZ = np.linalg.det(A*1e-10)
        I, error = self.integrate_df_A(A, lambda _e, v, h, _k: np.einsum("yds,na,ns,nbd->naby", antisym_tensor, v, v, h), hessians=True)
        I2, error2 = self.integrate_df_A(A, lambda _e, v, _k: v[:,None,:] * v[:,:,None], hessians=False)
        sigma3 = 1/V_EZ * elementary_charge**3/eV * spin_factor * I
        sigma2 = 1/V_EZ * elementary_charge**2/eV * spin_factor * I2
        return (sigma2, sigma2/I2*error2), (sigma3, sigma3/I*error)
    
    # electric part of the heat conductivity kappa in J/m^3 (if cell_length is given in meters)
    def heat_conductivity(self, A, spin_factor=2, print_error=False):
        pass


""" Free electron model in eV for testing, reciprocal lattice k in 2pi/Å """
class FreeElectrons:
    def __init__(self, k_neighbors=((0,0,0),), limit_count=None):
        eV_unit = 3.80998211 * (2*np.pi)**2 # hbar^2/m_e/2 * (2pi/1Å)^2 / 1eV
        self.fac = eV_unit
        self.k_neighbors = np.asarray(k_neighbors).T
        self.limit_count = limit_count

    def __call__(self, k_smpl):
        k_smpl = k_smpl[:,:,None] - self.k_neighbors[None,:,:]
        bands = self.fac * np.linalg.norm(k_smpl, axis=-2)**2
        if self.limit_count:
            bands = np.sort(bands, axis=-1)[:,:self.limit_count]
        return bands
    
    def bands_grad(self, k_smpl):
        k_smpl = k_smpl[:,:,None] - self.k_neighbors[None,:,:]
        bands = np.linalg.norm(k_smpl, axis=-2)**2
        grad = 2*k_smpl
        if self.limit_count:
            order = np.argsort(bands, axis=-1)[:,:self.limit_count]
            bands = np.take_along_axis(bands, order, axis=-1)
            grad = np.take_along_axis(grad, order[:,None,:], axis=-1)
        return self.fac * bands, self.fac * grad
    
    def bands_grad_hess(self, k_smpl):
        k_smpl = k_smpl[:,:,None] - self.k_neighbors[None,:,:]
        bands = np.linalg.norm(k_smpl, axis=-2)**2
        grad = 2*k_smpl
        hess = np.array([np.eye(3) * 2] * len(self.k_neighbors[0])).T
        if self.limit_count:
            order = np.argsort(bands, axis=-1)[:,:self.limit_count]
            bands = np.take_along_axis(bands, order, axis=-1)
            grad = np.take_along_axis(grad, order[:,None,:], axis=-1)
        return self.fac * bands, self.fac * grad, self.fac * hess[None,...]
