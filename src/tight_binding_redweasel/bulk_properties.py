# this file computes properties of the bulk material.
# meaning properties in the volume of the material, not the surface.

import numpy as np
import scipy.special
import math
from collections.abc import Callable, Sequence, Iterable
from . import density_of_states as _dos
import warnings

elementary_charge = 1.602176634e-19  # in Coulomb
eV = elementary_charge  # in Joule
hbar = 1.05457181764616e-34  # in SI J*s
c = 299792458.0  # speed of light in m/s

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

# TODO constant mean free pathlength -> tau = l/v

def conductivity(L_a):
    """Compute the conductivity from L_a from `KIntegral.transport_coefficients`, where only the first L^(0) is needed.
    The conductivity sigma is defined as `j_e = sigma E`.

    Args:
        L_a (arraylike[>=1, N_d, N_d]): The transport coefficients.

    Returns:
        ndarray[N_d, N_d]: The conductivity (usually in SI units)
    """
    return L_a[0]


def peltier(L_a):
    """Compute the Peltier tensor from L_a from `KIntegral.transport_coefficients`, where only the first two L^(0), L^(1) are needed.
    The Peltier tensor P is defined as `j_Q = P j_e`.

    Args:
        L_a (arraylike[>=1, N_d, N_d]): The transport coefficients.

    Returns:
        ndarray[N_d, N_d]: The Peltier tensor (usually in SI units)
    """
    return L_a[1] @ np.linalg.inv(L_a[0])


def seebeck(L_a, T):
    """Compute the Seebeck tensor (aka thermopower) from L_a from `KIntegral.transport_coefficients`, where only the first two L^(0), L^(1) are needed.
    The Seebeck tensor S is defined as `E = S (grad T)`.

    Args:
        L_a (arraylike[>=1, N_d, N_d]): The transport coefficients.
        T (float): The temperature at which the transport coefficients were computed.

    Returns:
        ndarray[N_d, N_d]: The Seebeck tensor (usually in SI units)
    """
    return np.linalg.inv(L_a[0]) @ L_a[1] / T


def electronic_thermal_conductivity(L_a, T):
    """Compute the electronic part of the thermal conductivity tensor from L_a from `KIntegral.transport_coefficients`, where the first three L^(0), L^(1), L^(2) are needed.
    The thermal conductivity tensor kappa is defined as `j_Q = kappa (-grad T)`.

    Args:
        L_a (arraylike[>=1, N_d, N_d]): The transport coefficients.
        T (float): The temperature at which the transport coefficients were computed.

    Returns:
        ndarray[N_d, N_d]: The electronic thermal conductivity tensor (usually in SI units)
    """
    return (L_a[2] - L_a[1] @ np.linalg.inv(L_a[0]) @ L_a[1]) / T


def hall_coefficients(sigma2, sigma3):
    """Calculate the hall (pseudo-)tensor from the output of `KIntegral.conductivity_hall_tensor`.

    The result may be asymmetric. To symmetrize, use
    ```python
    sym = Symmetry(...)
    R_h = sym.symmetrize(R_h, rank=(0, 3), pseudo_tensor=True)
    ```

    Args:
        sigma2 ((arraylike[..., 3, 3])):    rank 2 conductivity tensor
        sigma3 ((arraylike[..., 3, 3, 3])): rank 3 conductivity tensor

    Returns:
        (ndarray[..., 3, 3, 3]): R_ijk Hall tensor
    """
    # For reference see: BoltzTraP. A code for calculating band-structure dependent quantities
    # Authors: Georg K.H. Madsena, David J. Singhb

    sigma2_inv = np.linalg.inv(sigma2)
    return np.einsum("...aj,...abk,...ib->...ijk", sigma2_inv, sigma3, sigma2_inv)


def _integration_points(fermi_energy: float, T_min: float, T_max: float, N: int) -> np.ndarray:
    assert N > 0
    if N == 1:
        return np.array([fermi_energy])
    # just use gauss integration points for T_max, as the T_max integral is always the most difficult.
    assert T_min <= T_max
    assert N % 2 == 1
    beta_min = 1 / (_dos.k_B * T_max)
    beta_mid = 1 / (_dos.k_B * (T_max + T_min) / 2)
    mu_j = [fermi_energy, fermi_energy]  # for now just one mu
    beta_j = [beta_min, beta_mid]
    # equations for two points!
    I_f2_c = [2 * scipy.special.zeta(n) * (1 - 2**(1 - n)) if n % 2 == 0 else 0 for n in range(2 * N)]
    I_f3 = lambda n, m: I_f2_c[n + m] / np.float128(math.factorial(2 * max(n, m)) // math.factorial(n + m) * math.factorial(2 * min(m, n)))
    I_nj = lambda n, m, j: sum(scipy.special.binom(n, k) * (mu_j[j] - mu_j[0])**(n - k) * beta_j[j]**(-k - m) * I_f3(m, k) for k in range(n + 1))
    if T_min == T_max:
        mat = np.array([[I_nj(n, m, 0) for n in range(N)] for m in range(N + 1)])
    else:
        # this is sometimes unstable...
        mat = np.array([[I_nj(n, m, 0) for n in range(N // 2 - 1)] + [I_nj(n, m, 1) for n in range(N - (N // 2 - 1))] for m in range(N + 1)])
    q, _ = scipy.linalg.qr(mat)
    poly = [c / math.factorial(2 * (N - i)) for i, c in enumerate(reversed(q[:, -1]))]
    assert len(poly) == N + 1
    return fermi_energy + np.array(sorted([np.real(r) for r in np.roots(poly)]))


def _integration_weights(x: np.ndarray, beta_j: np.ndarray, mu_j: np.ndarray, E_min: float, E_max: float, rtol=1e-4):
    assert len(x) % 2 == 1
    if len(x) == 1:
        return np.array([1.0])
    b = [2 * scipy.special.zeta(n) * (1 - 2**(1 - n)) if n % 2 == 0 else 0 for n in range(len(x))]
    I_nj = lambda n, j: beta_j[j]**(-n) * b[n]
    # sum_i X_ijn w_ij = I_jn
    # M = #j linear equations of the form
    # X_in w_i = I_n
    w = []
    assert len(beta_j) == len(mu_j)
    for j, (mu, beta) in enumerate(zip(mu_j, beta_j)):
        if beta > 1e16:
            w_j = np.zeros(len(x))
            x_i = np.argmin(np.abs(x - mu))
            assert abs(x[x_i] - mu) < 1e-10
            w_j[x_i] = 1
        else:
            X = [[(x_ - mu)**n / math.factorial(n) for x_ in x] for n in range(len(x))]
            I = [I_nj(n, j) for n in range(len(x))]
            w_j = np.linalg.solve(X, I)
        w.append(w_j)
    w = np.array(w)
    if np.sum(w < -0.005) > 0:
        warnings.warn("There are negative weights, which indicates that the integation will be unstable.", RuntimeWarning)
    # filter weights, which are irrelevant due to the integrated function being bounded
    # Note, leaving them out from the beginning leads to worse results.
    select = (np.max(w / np.max(w, axis=1, keepdims=True), axis=0) > rtol * len(x)) & (x > E_min) & (x < E_max)
    return np.array(x[select]), np.array(w[:, select])


class KIntegral:
    """
    This class bundles the data for bandstructure integrals (k-space).
    This is useful as a class, as it can reuse the same calculations for multiple integrals.

    The integrals are of different forms, so look for the `integrate..` functions for details.
    """

    def __init__(self, dos_model: _dos.DensityOfStates, electrons: float, T: Sequence, N=None, rtol=1e-4):
        """Initialize an Integral over the bandstructure (k-space)

        Args:
            dos_model (DensityOfStates): Density of states (DoS) model which contains all important information.
            electrons (int): Number of electrons/filled bands in the model.
            T (list): Temperatures at which the integral is computed. Best at the highest T and at T=0, but interpolated physically inbetween.
            N (int, optional): The number of Fermi-surfaces to be used in the integration. Defaults to 5 if only one temperature is used, otherwise 9.
        """
        assert all(T_ >= 0 for T_ in T), f"Only non negative temperatures are allowed, but got {T}"
        self.bandgap = dos_model.bandgap(electrons)
        self.metal = self.bandgap == 0.0
        self.model = dos_model.model
        self.A = dos_model.A
        self.mu = dos_model.chemical_potential(electrons, T, N=50)  # compute mu here with good enough precision
        self.fermi_energy = dos_model.fermi_energy(electrons)
        if N is None:
            N = (1 if T[0] == 0 else 13) if len(T) == 1 else 11
        assert N % 2 == 1, f"only odd N are allowed, but got {N}"
        self.beta = 1 / (_dos.k_B * np.array(T) + 1e-20)  # in 1/eV
        # prepared data for the integration
        self.k_smpl = []
        self.band_indices = []
        self.weights = []
        # function to collect the fermi surface data
        # TODO allow the use of the dos grid instead of the exact fermi surface
        # -> faster, less precise
        improved_points = False  # True makes it much more robust -> extrapolation possible, however it doesn't increase the convergence order
        non_zero_T = [T_ for T_ in T if T_ > 0]
        if non_zero_T:
            if self.metal:
                self.e_smpl = _integration_points(self.fermi_energy, min(non_zero_T), max(T), N)
                self.e_smpl, self.integration_weights = _integration_weights(self.e_smpl, self.beta, self.mu, *dos_model.energy_range(), rtol=rtol)
            else:
                # do Gauss-Laguerre integrations at the upper and lower bandgap boundary
                # TODO
                raise NotImplementedError("integrals for isolators (semiconductors) are currently not implemented")
        else:
            if self.metal:
                self.e_smpl = np.array([self.fermi_energy])
                self.integration_weights = np.array([[1]])
            else:
                # no e_smpl, all integrals are empty for T=0
                pass
        self.e_smpl = np.array(self.e_smpl)
        for e in self.e_smpl:
            k_smpl, band_indices, weights, _ = dos_model.fermi_surface_samples(e, improved_points=improved_points, improved_weights=False, normalize=None)
            self.k_smpl.append(k_smpl)
            self.band_indices.append(band_indices)
            self.weights.append(weights)
        self.v: list = None  # type: ignore # group velocities
        self.h: list = None  # type: ignore # hessians
        self.w: list = None  # type: ignore # weights divided by absolute group velocities

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
                v = np.take_along_axis(v, band_indices.reshape(-1, 1, 1), axis=-1)[..., 0]
                self.v.append(v)
                self.h.append(np.take_along_axis(h, band_indices.reshape(-1, 1, 1, 1), axis=-1)[..., 0])
                self.w.append(weights / (1e-8 + np.linalg.norm(v, axis=-1)))
        else:
            self.v = []
            self.w = []
            for k_smpl, band_indices, weights in zip(self.k_smpl, self.band_indices, self.weights):
                _, v = self.model.bands_grad(k_smpl)
                v = np.take_along_axis(v, band_indices.reshape(-1, 1, 1), axis=-1)[..., 0]
                self.v.append(v)
                self.w.append(weights / (1e-8 + np.linalg.norm(v, axis=-1)))

    def integrate_df(self, g: Callable, hessians=False, moments=0):
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
            moments (int, optional): How many moments, i.e. integrals with (e - mu)^a with `0 <= a <= moments` are computed. These come almost for free.

        Returns:
            ndarray[N_T]: The value of the integral at each temperature.
        """
        assert moments >= 0 or not isinstance(moments, int), "moments needs to be non negative integer"
        if self.v is None or (hessians and self.h is None):
            self.precompute(hessians)

        res = []
        for index, e in enumerate(self.e_smpl):
            # use data from the precomputation
            if hessians:
                f_res = g(e, self.v[index], self.h[index], self.k_smpl[index])
                res.append(np.sum(f_res * self.w[index].reshape((-1,) + (1,) * len(np.shape(f_res)[1:])), axis=0))
            else:
                f_res = g(e, self.v[index], self.k_smpl[index])
                res.append(np.sum(f_res * self.w[index].reshape((-1,) + (1,) * len(np.shape(f_res)[1:])), axis=0))

        if self.metal:
            I_moments = []
            for a in range(moments + 1):
                I = []
                for mu, w_j in zip(self.mu, self.integration_weights):
                    I.append(sum(r * w * (e - mu)**a for r, e, w in zip(res, self.e_smpl, w_j)))
                I_moments.append(I)
            return np.array(I_moments)
        else:
            if len(self.beta) == 1 and self.beta[0] > 1e16:
                # no fermi surface!
                # to get the output shape right, evaluate g at one point
                if hessians:
                    I = 0.0 * np.asanyarray(g(self.mu, np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3))))
                else:
                    I = 0.0 * np.asanyarray(g(self.mu, np.zeros((1, 3)), np.zeros((1, 3))))
                return np.array([[I]])
            # TODO see comment in __init__
            raise NotImplementedError("integrals for isolators (semiconductors) are currently not implemented")

    def integrate_df_A(self, A, g: Callable, hessians=False, moments=0):
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
            moments (int, optional): How many moments, i.e. integrals with (e - mu)^a with `0 <= a <= moments` are computed. These come almost for free.

        Returns:
            ndarray[N_T]: The value of the integral at each temperature.
        """
        # consider DoS space matrix as an already applied A matrix
        A = A @ np.linalg.inv(self.A)
        # reciprocal space vectors
        B = 2 * np.pi * np.linalg.inv(A.T)
        # transform the derivative, which live in the dual vector space -> dual transformation
        # this transformation also includes the unit conversion using Ångström and hbar in SI.
        # The unit of energy needs to be converted to Joule down below as well.
        grad_transform = (1 / (2 * np.pi) * 1e-10 / hbar) * A
        if hessians:
            def g2(e, v, h, k):
                k = np.einsum("ji,ni->nj", B, k)
                v = np.einsum("ji,ni->nj", grad_transform * eV, v)
                h = np.einsum("ki,lj,nij->nkl", grad_transform * eV, grad_transform, h)
                res = g(e, v, h, k)
                assert len(res) == len(k)
                return res
            return self.integrate_df(g2, hessians=True, moments=moments)
        else:
            def g3(e, v, k):
                k = np.einsum("ji,ni->nj", B, k)
                v = np.einsum("ji,ni->nj", grad_transform * eV, v)
                res = g(e, v, k)
                assert len(res) == len(k)
                return res
            return self.integrate_df(g3, hessians=False, moments=moments)

    def transport_coefficients(self, A: np.ndarray, spin_factor=2, tau=None, max_a=2):
        """Transport coefficients `L^(n)`. To get usual experimental values,
        use the functions `conductivity`, `peltier`, `seebeck` and `electronic_thermal_conductivity`.

        ```raw
        L^(a) = q^(2-a) integral dk tau(k) v o v (E - mu)^a (-df/dE)
        j_e = L^(0) E + L^(1) (-grad T)/T
        j_Q = L^(1) E + L^(2) (-grad T)/T
        ```

        Args:
            A (arraylike[N_d, N_d]): The lattice vectors in units of Angstrom.
            spin_factor (int, optional): The number of electrons per band. Defaults to 2.
            tau (Callable[[arraylike[N_d]], float], optional): The scattering time as a function of k. Defaults to None.
            max_a (int, optional): The maximal L^(a) to be computed. Defaults to 2.

        Returns:
            (ndarray[max_a+1, N_T, N_d, N_d]): (L^(0), L^(1), L^(2)). Results L^(0) in 1/(Ohm*m)/s, L^(1) in J/C/(Ohm*m)/s, L^(2) in (J/C)^2/(Ohm*m)/s.
        """
        V_EZ = np.linalg.det(A * 1e-10)
        if isinstance(tau, float):
            tau_v = tau
            tau = lambda _: tau_v
        tau_1 = (lambda v, _: v) if tau is None else (lambda v, k: v * np.reshape(tau(k), (-1, 1, 1)))
        I = self.integrate_df_A(A, lambda _e, v, k: tau_1(v[:, None, :] * v[:, :, None], k), hessians=False, moments=max_a)
        a = np.arange(max_a + 1)[:, None, None, None]
        D = spin_factor * elementary_charge**(2 - a) * I / eV**(1 - a) / V_EZ
        return D

    # conductivity divided by the electron scattering time tau. Assuming constant tau. Result in 1/(Ohm*m*s)
    def drude_factor(self, A: np.ndarray, spin_factor=2):
        V_EZ = np.linalg.det(A * 1e-10)
        I = self.integrate_df_A(A, lambda _e, v, _k: v[:, None, :] * v[:, :, None], hessians=False)[0]
        D = spin_factor * elementary_charge**2 * I / eV / V_EZ
        return D  # result is in 1/(Ohm*m)/s

    # electric part of the volumetric heat capacity c_V in J/m^3
    # (This is better computed by the DoS itself)
    def heat_capacity(self, spin_factor=2):
        warnings.warn("The function heat_capacity is known to misbehave.", DeprecationWarning)
        I = self.integrate_df(lambda e, _v, _k: e, hessians=False, moments=1)[1]
        cV_T = spin_factor * I * eV
        T = 1 / (self.beta * _dos.k_B)
        return cV_T / T

    def hall_coefficient(self, A: np.ndarray, spin_factor=2):
        warnings.warn("This function is only left in for reference. Use conductivity_hall_tensor or hall_coefficient_metal_cubic.", DeprecationWarning)
        # TODO remove! This is bad!
        # PhysRevB.82.035103 Hall effect formula with constant relaxation time for all bands over all k.
        # In reality the relaxation times are different over k-space or even just for different spins in the same band. (PhysRev.97.647)
        # Somehow their units don't match the expected result unit, so I removed the division by the speed of light to make it work.
        # I canceled one e from sigma_xx with e^2 from sigma_xyz
        V_EZ = np.linalg.det(A * 1e-10)
        # sigma_xyz = elementary_charge/c/eV * self.integrate_df_A(A, lambda _e, v, h, _k: (v[:,0]**2*h[:,1,1] - v[:,0]*v[:,1]*h[:,1,0]), hessians=True)
        I = self.integrate_df_A(A, lambda _e, v, h, _k: (v[:, 0]**2 * h[:, 1, 1] - v[:, 0] * v[:, 1] * h[:, 1, 0]), hessians=True)[0]
        I2 = self.integrate_df_A(A, lambda _e, v, _k: v[:, 0]**2, hessians=False)[0]
        sigma_xyz = -elementary_charge / eV * spin_factor * I
        sigma_xx = elementary_charge / eV * spin_factor * I2
        R_H = sigma_xyz / sigma_xx**2 * V_EZ
        # R_H_error = R_H * ((error/I)**2 + (2*error2/I2)**2)**.5
        return R_H  # in m^3/C = Ohm m/T

    def hall_coefficient_metal_cubic(self, A: np.ndarray, spin_factor=2):
        # PhysRevB.45.10886 Hall effect formula with constant relaxation time over k.
        # In reality the relaxation times are different over k-space or even just for different spins in the same band. (PhysRev.97.647)
        # I canceled one e from sigma_0 with e^2 from sigma_H
        V_EZ = np.linalg.det(A * 1e-10)
        I = self.integrate_df_A(A, lambda _e, v, h, _k: np.einsum("ni,nij,nj->n", v, h, v) - np.einsum("nj,nj,nii->n", v, v, h), hessians=True)[0]
        I2 = self.integrate_df_A(A, lambda _e, v, _k: (v**2).sum(-1), hessians=False)[0]
        sigma_H = elementary_charge / eV * spin_factor / 6 * I
        sigma_0 = elementary_charge / eV * spin_factor / 3 * I2
        R_H = sigma_H / sigma_0**2 * V_EZ
        # R_H_error = R_H * ((error/I)**2 + (2*error2/I2)**2)**.5
        return R_H  # in m^3/C = Ohm m/T

    def conductivity_hall_tensor(self, A: np.ndarray, spin_factor=2, tau=None):
        """Compute the conductivity tensors of rank 2 and rank 3,
        which are required for the Drude weight and the hall coefficient.
        The formulas are:
        ```
        sigma2_ij = tau(k) v_i v_j
        sigma3_ijk = tau(k) epsilon_kab v_i M^-1_ja v_b
        ```

        Args:
            A (arraylike[3, 3]): The lattice vectors in units of Angstrom.
            spin_factor (int, optional): The number of electrons per band. Defaults to 2.
            tau (Callable[[arraylike[N_d]], float], optional): The scattering time as a function of k. Defaults to None.

        Returns:
            ((arraylike[3, 3], arraylike[3, 3]), (arraylike[3, 3, 3], arraylike[3, 3, 3])):
                ((rank 2 conductivity tensor, the error of the tensor), (rank 3 conductivity tensor, the error of the tensor))
        """
        # From the book "C. Hurd, The Hall Coeffcient of Metals and Alloys (Plenum, New York, 1972)"
        # Assuming constant relaxation time for now -> independent of it
        V_EZ = np.linalg.det(A * 1e-10)
        if isinstance(tau, float):
            tau_v = tau
            tau = lambda _: tau_v
        tau_1 = (lambda v, _: v) if tau is None else (lambda v, k: v * tau(k))
        tau_2 = (lambda v, _: v) if tau is None else (lambda v, k: v * tau(k)**2)
        I = self.integrate_df_A(A, lambda _e, v, h, k: tau_2(np.einsum("yds,na,ns,nbd->naby", antisym_tensor, v, v, h), k), hessians=True)[0]
        I2 = self.integrate_df_A(A, lambda _e, v, k: tau_1(v[:, None, :] * v[:, :, None], k), hessians=False)[0]
        sigma3 = 1 / V_EZ * elementary_charge**3 / eV * spin_factor * I
        sigma2 = 1 / V_EZ * elementary_charge**2 / eV * spin_factor * I2
        return sigma2, sigma3


class FreeElectrons:
    """
    Free electron model in eV for testing, reciprocal lattice k in 2pi/Å
    and non duplicated bands for spin, so spin has to be considered when using it.
    """

    def __init__(self, k_neighbors=((0, 0, 0),), limit_count: int | None = None):
        eV_unit = 3.80998211 * (2 * np.pi)**2  # hbar^2/m_e/2 * (2pi/1Å)^2 / 1eV
        self.fac = eV_unit
        self.k_neighbors = np.asarray(k_neighbors).T
        self.limit_count = limit_count

    def __call__(self, k_smpl: Iterable):
        k_smpl = np.asarray(k_smpl)[:, :, None] - self.k_neighbors[None, :, :]
        bands = self.fac * np.linalg.norm(k_smpl, axis=-2)**2
        bands = np.sort(bands, axis=-1)
        if self.limit_count:
            bands = bands[:, :self.limit_count]
        return bands

    def bands_grad(self, k_smpl: Iterable):
        k_smpl = np.asarray(k_smpl)[:, :, None] - self.k_neighbors[None, :, :]
        bands = np.linalg.norm(k_smpl, axis=-2)**2
        grad = 2 * k_smpl
        order = np.argsort(bands, axis=-1)
        if self.limit_count:
            order = order[:, :self.limit_count]
        bands = np.take_along_axis(bands, order, axis=-1)
        grad = np.take_along_axis(grad, order[:, None, :], axis=-1)
        return self.fac * bands, self.fac * grad

    def bands_grad_hess(self, k_smpl: Iterable):
        k_smpl = np.asarray(k_smpl)[:, :, None] - self.k_neighbors[None, :, :]
        bands = np.linalg.norm(k_smpl, axis=-2)**2
        grad = 2 * k_smpl
        hess = np.array([np.eye(3) * 2] * len(self.k_neighbors[0])).T
        order = np.argsort(bands, axis=-1)
        if self.limit_count:
            order = order[:, :self.limit_count]
        bands = np.take_along_axis(bands, order, axis=-1)
        grad = np.take_along_axis(grad, order[:, None, :], axis=-1)
        return self.fac * bands, self.fac * grad, self.fac * hess[None, ...]
