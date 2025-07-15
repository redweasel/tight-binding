# all sorts of functions related to the density of states
# and the fermi energy at finite temperatures

import numpy as np
import scipy
from .smearing import *
from collections.abc import Callable, Iterable
from typing import Any

k_B = 8.61733326214518e-5  # eV/K


def gauss_5_df(f, mu, beta):
    """gauss integration using the derivative of the fermi function as weight function (integration over const 1 is always 1)"""
    x = np.array([-8.211650879369324585, -3.054894371595123559, 0.0, 3.054894371595123559, 8.211650879369324585])
    w = np.array([0.0018831678927720540279, 0.16265409664449248517, 0.6709254709254709216, 0.16265409664449248517, 0.0018831678927720540279])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0)


def gauss_7_df(f, mu, beta):
    """gauss integration using the derivative of the fermi function as weight function (integration over const 1 is always 1)"""
    x = np.array([-13.208619179276130495, -6.822258565704534483, -2.74015863439980345, 0.0, 2.74015863439980345, 6.822258565704534483, 13.208619179276130495])
    w = np.array([1.5006783863250833195e-05, 0.0054715468031045289346, 0.18481163647966422636, 0.61940361986673598773, 0.18481163647966422636, 0.0054715468031045289346, 1.5006783863250833195e-05])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0)


def gauss_4_f(f, mu, beta):
    """gauss integration using the half-fermi function 1/(1+e^abs(x)) as weight function"""
    x = np.array([-5.7755299052817408167, -1.3141165029468411252, 1.3141165029468411252, 5.7755299052817408167])
    w = np.array([0.013822390808990555472, 0.48617760919100944106, 0.48617760919100944106, 0.013822390808990555472])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0) / beta * 2 * np.log(2)


def gauss_6_f(f, mu, beta):
    """gauss integration using the half-fermi function 1/(1+e^abs(x)) as weight function"""
    x = np.array([-10.612971636582431145, -4.7544317516063152596, -1.1799810705877835648, 1.1799810705877835648, 4.7544317516063152596, 10.612971636582431145])
    w = np.array([0.00013527426019966680491, 0.02778699826589691238, 0.47207772747390341905, 0.47207772747390341905, 0.02778699826589691238, 0.00013527426019966680491])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0) / beta * 2 * np.log(2)


def convolve_df(x, energy_smpl: np.ndarray, states: np.ndarray, beta: float, extrapolation: str | None = 'flat', extrapolation_point=None) -> np.ndarray:
    """explicitly compute the value of the convolution of the states
    function with the negative derivative of the Fermi-function.
    takes a piecewise linear representation of the states function
    and returns the exact convolution.

    Args:
        x (float): The energy value to evaluate the convolution at.
        energy_smpl (arraylike[N_e]): The energy points of the piecewise linear approximation.
        states (arraylike[..., N_e]): The values of the piecewise linear approximation. Usually the number of states.
        beta (float): the inverse temperature 1/(k_B*T)
        extrapolation (str, optional): How to extrapolate beyond the piecewise linear part. Either "flat" for a constant piece on both sides, or None. Defaults to 'flat'.
        extrapolation_point (tuple[float, float], optional): If given, (highest_energy, band_count) of the right extrapolation point. The left extrapolation point is assumed to be at (0, 0). Defaults to None.
    """
    def f(e):
        return scipy.special.expit(-beta * e)  # overflow free variant of 1 / (1 + exp(beta*e)), just to silence the warnings

    def F(e):
        # return -np.log1p(np.exp(-beta*e)) / beta
        return scipy.special.log_expit(beta * e) / beta  # overflow free variant

    def F_diff(e0, e1):
        return F(e1) - F(e0)
        # return np.log(scipy.special.expit(beta*e1) + np.exp(-beta*e0)/(1 + np.exp(-beta*e1))) / beta

    def segment(x, x0, x1, y0, y1):
        return y1 * f(x - x1) - y0 * f(x - x0) + (y1 - y0) / (x1 - x0) * F_diff(x - x0, x - x1)
    s = np.sum(segment(x, energy_smpl[..., :-1], energy_smpl[..., 1:], states[..., :-1], states[..., 1:]), axis=-1)
    # add segments/other functions as extrapolation on both side to get correct high temperature behavior
    if extrapolation == 'flat':
        if extrapolation_point is None:
            s += f(x - energy_smpl[0]) * states[..., 0]
            s += f(energy_smpl[-1] - x) * states[..., -1]
        else:
            # assume N(mu) starts at 0 and ends at N(extrapolation_point[0]) = extrapolation_point[1]
            s += f(extrapolation_point[0] - x) * extrapolation_point[1]
    return s


def convolve_ddf(x, energy_smpl: np.ndarray, states: np.ndarray, beta: float, extrapolation='flat', extrapolation_point=None) -> np.ndarray:
    """The analytic derivative of `convolve_df`

    Args:
        x (float): The energy value to evaluate the convolution at.
        energy_smpl (arraylike[N_e]): The energy points of the piecewise linear approximation.
        states (arraylike[..., N_e]): The values of the piecewise linear approximation. Usually the number of states.
        beta (float): the inverse temperature 1/(k_B*T)
        extrapolation (str, optional): How to extrapolate beyond the piecewise linear part. Either "flat" for a constant piece on both sides, or None. Defaults to 'flat'.
        extrapolation_point (tuple[float, float], optional): If given, (highest_energy, band_count) of the right extrapolation point. The left extrapolation point is assumed to be at (0, 0). Defaults to None.
    """
    def df(e):
        return -0.25 * beta / np.cosh(0.5 * beta * e)**2  # ignore warning here. (Could use scipy.stats.hypsecant here to avoid the warnings)

    def f(e):
        return scipy.special.expit(-beta * e)  # overflow free variant of 1 / (1 + exp(beta*e)), just to silence the warnings

    def f_diff(e0, e1):
        return f(e1) - f(e0)

    def dsegment(x, x0, x1, y0, y1):
        return y1 * df(x - x1) - y0 * df(x - x0) + (y1 - y0) / (x1 - x0) * f_diff(x - x0, x - x1)
    s = np.sum(dsegment(x, energy_smpl[..., :-1], energy_smpl[..., 1:], states[..., :-1], states[..., 1:]), axis=-1)
    # add segments/other functions as extrapolation on both side to get correct high temperature behavior
    if extrapolation == 'flat':
        if extrapolation_point is None:
            s += df(x - energy_smpl[0]) * states[..., 0]
            s += df(energy_smpl[-1] - x) * states[..., -1]
        else:
            # assume N(mu) starts at 0 and ends at N(extrapolation_point[0]) = extrapolation_point[1]
            s += df(extrapolation_point[0] - x) * extrapolation_point[1]
    return s


def secant(y, f, a, b, tol=1e-16, max_i=100):
    """
    finds the inverse of strictly monotonic functions on the full range of real numbers
    is unstable at points where the derivative of f is very small (-> strictly monotonic)
    converges faster than secant_bisect
    use the more precise start estimate for b, because it is kept in the first step
    """
    fa = f(a) - y
    fb = f(b) - y
    for i in range(max_i):
        x = (a * fb - b * fa) / (fb - fa)
        if a == x or b == x:
            return x
        val = f(x) - y
        a = b
        fa = fb
        b = x
        fb = val
        if abs(val) < tol or fa == fb:
            return x
    # didn't finish but here is the best available solution
    return (a * fb - b * fa) / (fb - fa)


def _Li2(x):
    return scipy.special.spence(1 - x)


def _int_df(x):
    # 0 at x=0
    return 0.5 * np.tanh(x / 2)


def _int_xdf(x):
    # 0 at x=±inf
    x = -np.abs(x)
    ex = np.exp(x)  # small
    return (x * ex) / (1 + ex) - np.log1p(ex)


def _int_xxdf(x):
    # 0 at x=0
    # use symmetry to avoid cancellation
    sign = np.sign(x)
    x = -np.abs(x)
    ex = np.exp(x)  # small
    f = (x * x * ex) / (1 + ex) - 2 * x * np.log1p(ex) - (2 * _Li2(-ex) + np.pi**2 / 6)
    return -sign * f


def int_poly(x0, x1, a, b, c):
    """
    Integrate (ax^2+bx+c)(-df/de) from x0 to x1 analytically with f(x)=1/(1+e^x) so df/de = -1/(2*cosh(x/2))^2.
    Well tested.
    """
    # TODO combine the calculations into one function, as there is lots of overlap
    return a * (_int_xxdf(x1) - _int_xxdf(x0)) + b * (_int_xdf(x1) - _int_xdf(x0)) + c * (_int_df(x1) - _int_df(x0))


def naive_fermi_energy(bands, electrons):
    i = round(np.prod(np.shape(bands)[:-1]) * electrons)
    return np.mean(np.sort(np.ravel(bands))[i:i + 2])


def naive_energy(bands, mu, T, spin_factor=2):
    e = np.ravel(bands)
    if T > 0:
        return np.mean(e * scipy.special.expit(-(e - mu) / (k_B * T))) * np.shape(bands)[-1] * spin_factor
    else:
        return np.sum(e[e <= mu]) * (np.shape(bands)[-1]/len(e)) * spin_factor


def _samples(e0, e1, mu, beta, N):
    # sample a range e0 to e1 in such a way, that an integral over -df/de has equal weights for each point. (importance sampling)
    # for numerical stability, it is required to limit e0 and e1 here
    e0 = np.maximum(mu - 30 / beta, e0)
    e1 = np.minimum(mu + 30 / beta, e1)
    #return np.log(1 / (1e-16 + np.linspace(scipy.special.expit(-beta * (e0 - mu)), scipy.special.expit(-beta * (e1 - mu)), N)) - 1 + 1e-16) / beta + mu
    return -scipy.special.logit(np.concatenate([np.linspace(scipy.special.expit(-beta * (e0 - mu)), 0.5, N//2, endpoint=False), np.linspace(0.5, scipy.special.expit(-beta * (e1 - mu)), (N+1)//2)])) / beta + mu

class DensityOfStates:
    """This class represents the density of states for a given 3D bandstructure model.
    Using this class, one can compute the Fermi-energy or the chemical potential at finite T.
    It is also the basis for `KIntegral` computations.

    The bandstructure model, used in this class needs to accept crystal coordinates.
    The cubic cell [-0.5, 0.5]^3 should equal the whole reciprocal crystal cell.
    """

    def __init__(self, model: Callable, N=24, A=None, ranges=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)), wrap=True, smearing="cubes", use_gradients=False, midpoint=False, check=True):
        """Initialize a density of states model from a bandstructure model.

        Args:
            model (Callable[[arraylike[N_k, 3]], bands]): The bandstructure model.
                This class allows any function to be used as a bandstructure model as long as it returns sorted bands.
                However there are special features, that use `model.bands_grad` and `model.bands_grad_hess` to improve the results if it is available.
                The input k for the model should be in reciprocal space specified by A, not in crystal space.
            N (int, optional): Number of k-points per direction. Defaults to 24.
            A (arraylike[3, 3]): Matrix with lattice vectors in its columns. Used for computing the k-grid for sampling the model. Defaults to the unit matrix.
            ranges (arraylike[3, 2], optional): The ranges for each axis in crystal space. This should only be changed to consider symmetries, as the whole cell will still be considered to have volume 1. Defaults to ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)).
            wrap (bool, optional): _description_. Defaults to True.
            smearing (str, optional): One of the options {"cubes", "tetras", "spheres"}. For details look at the class `SmearingMethod`. Defaults to "cubes".
            use_gradients (bool, optional): If the smearing supports it, use analytic gradients instead of interpolated gradients for the smearing calculations. Defaults to False.
            midpoint (bool, optional): Shift the k-points by half a cell, such that the borders are not included. Defaults to False.
            check (bool or float, optional): If not False, check the model for correct periodicity. If a float is given instead of a bool, it is interpreted as the tolerance for the check. Defaults to True.
        """
        assert len(ranges) == 3, "This class can only be used for 3D models. Therefore 3 ranges need to be specified."
        if wrap:
            xyz = [np.linspace(*r, N, endpoint=False) + (1 / 2 / N * (r[1] - r[0]) if midpoint else 0.0) for r in ranges]  # type: ignore
        else:
            if midpoint:
                # careful with this! only useful when use_gradients is True!
                xyz = [np.linspace(*r, N, endpoint=False) + 1 / 2 / N * (r[1] - r[0]) for r in ranges]  # type: ignore
            else:
                xyz = [np.linspace(*r, N + 1) for r in ranges]  # type: ignore
        self.crystal_volume = np.prod([r[1] - r[0] for r in ranges])
        # TODO redefine the grid, such that all stepsizes are equal! This is very important for the fermi surface samples!
        self.step_sizes = np.array([x[1] - x[0] for x in xyz])
        if A is None:
            A = np.eye(3)
        self.A = np.asarray(A)
        # reciprocal space k_smpl using A
        self.k_smpl = np.stack(np.meshgrid(*xyz, indexing='ij'), axis=-1) @ np.linalg.inv(self.A)
        self.wrap = wrap
        self.model = model
        # make sure the model has the correct periodicity
        if not (check is False):
            if isinstance(check, float):
                self.check(tolerance=check)
            else:
                self.check()
        shape = np.shape(self.k_smpl)
        can_have_gradients = smearing in ["spheres", "cubes"]
        if can_have_gradients and use_gradients:
            if "bands_grad" not in dir(model):
                raise ValueError(f"The model must implement a method `bands_grad` for computing the bands and gradients for {smearing} smearing.")
            bands, grads = model.bands_grad(np.reshape(self.k_smpl, (-1, 3)))
            grads = np.reshape(grads, shape + (-1,))
        else:
            bands = model(np.reshape(self.k_smpl, (-1, 3)))
        self.bands = np.reshape(bands, shape[:-1] + (-1,))
        if "bands_grad_hess" in dir(model):
            # compute better band ranges using a Newton step to find the actual extrema in k
            # -> use self.model.bands_grads_hess(...), and do the step if the hessian is positive semi-definite
            # -> only do that if model.bands_grads_hess exists to keep supporting simpler models
            bands_range_indices = [(np.argmin(self.bands[..., i]), np.argmax(self.bands[..., i])) for i in range(self.bands.shape[-1])]
            self.bands_range = []
            for i, band_range_indices in enumerate(bands_range_indices):
                # find minimum
                min_k = np.array(self.k_smpl.reshape(-1, 3)[band_range_indices[0]])
                for _ in range(2):
                    _, grad, hess = model.bands_grad_hess(np.array([min_k]))
                    eigvals = np.linalg.eigvalsh(hess[0, :, :, i])
                    if np.any(eigvals < 0):
                        break
                    step = np.linalg.pinv(hess[0, :, :, i]) @ grad[0, :, i]
                    if np.max(np.abs(step / self.step_sizes)) > 0.55:
                        break  # don't follow too big steps (e.g. at band crossings)
                    min_k -= step
                # find maximum
                max_k = np.array(self.k_smpl.reshape(-1, 3)[band_range_indices[1]])
                for _ in range(2):
                    _, grad, hess = model.bands_grad_hess(np.array([max_k]))
                    eigvals = np.linalg.eigvalsh(hess[0, :, :, i])
                    if np.any(eigvals > 0):
                        break
                    step = np.linalg.pinv(hess[0, :, :, i]) @ grad[0, :, i]
                    if np.max(np.abs(step / self.step_sizes)) > 0.55:
                        break  # don't follow too big steps (e.g. at band crossings)
                    max_k -= step
                # compute min/max values and store them
                self.bands_range.append(tuple(model(np.array([min_k, max_k]))[:, i]))
        else:
            self.bands_range = [(np.min(self.bands[..., i]), np.max(self.bands[..., i])) for i in range(self.bands.shape[-1])]
        # TODO add bands_range_k_points, because that is useful information in some contexts
        B = np.linalg.inv(self.A).T
        if smearing == "tetras":
            if use_gradients:
                raise NotImplementedError("use_gradients is not implemented for smearing tetras")
            self.smearing = [TetraSmearing(values=self.bands[..., i], wrap=wrap, B=B) for i in range(len(self.bands_range))]
        elif smearing == "cubes":
            if use_gradients:
                self.smearing = [CubesSmearing(self.k_smpl, A=A, values=self.bands[..., i], grads=grads[..., i], wrap=wrap) for i in range(len(self.bands_range))]
            else:
                self.smearing = [CubesSmearing(self.k_smpl, A=A, values=self.bands[..., i], wrap=wrap) for i in range(len(self.bands_range))]
        elif smearing == "spheres":
            if use_gradients:
                # with gradients (a lot less regular, ig the averaged gradient are really needed)
                self.smearing = [SphereSmearing(self.k_smpl, values=self.bands[..., i], grads=grads[..., i], B=B, wrap=wrap) for i in range(len(self.bands_range))]
            else:
                # with interpolation (VERY close to "cubes", but no fermi surface sampling)
                self.smearing = [SphereSmearing(self.k_smpl, values=self.bands[..., i], B=B, wrap=wrap) for i in range(len(self.bands_range))]
        else:
            raise ValueError(f"{smearing} is not a valid smearing option")

    def model_bandcount(self):
        return len(self.bands_range)

    def check(self, tolerance=1e-10):
        """
        Check if the tight binding model is periodic wrt the given A matrix.
        If not, an assertion error is raised.

        Args:
            tolerance (float, optional): The tolerance when comparing the bandstructure of equivalent cells.
                This is limited by rounding errors in the model and by rounding of the lattice parameters (self.A). Defaults to 1e-10.
        """
        k_smpl = np.random.random((20, 3)) @ np.linalg.inv(self.A)
        bands_a = self.model(k_smpl @ np.linalg.inv(self.A))
        for i in range(3):
            bands_b = self.model((k_smpl + np.identity(3)[None, i]) @ np.linalg.inv(self.A))
            error = np.linalg.norm(bands_a - bands_b)
            assert error < tolerance, f"not periodic with respect to self.A[{i}] (error = {error:.2e})"

    def save(self, filename: str):
        # TODO save without self.model
        raise NotImplementedError()

    @staticmethod
    def load(filename: str, model: Callable | None = None):
        # TODO load without self.model
        raise NotImplementedError()
        return DensityOfStates(...)
    
    def states_below_band(self, energy: float, i: int):
        if self.bands_range[i][0] < energy < self.bands_range[i][1]:
            return np.mean(self.smearing[i].volume(energy))
        elif self.bands_range[i][1] <= energy:
            return 1.0  # completely full
        else:
            return 0.0

    def states_below(self, energy: float):
        states = 0.0
        for i in range(len(self.bands_range)):
            states += self.states_below_band(energy, i)
        return states

    def states_density(self, energy: float):
        """returns states_below, density"""
        states = 0.0
        density = 0.0
        for i, band_range in enumerate(self.bands_range):
            if band_range[0] < energy < band_range[1]:
                volume, dvolume = self.smearing[i].volume_dvolume(energy)
                states += np.mean(volume)
                density += np.mean(dvolume)
            elif band_range[1] <= energy:
                states += 1.0  # completely full
        return states, density

    def density(self, energy: float):
        """returns density of states"""
        density = 0.0
        for i in range(len(self.bands_range)):
            density += self.density_band(energy, i)
        return density

    def density_band(self, energy: float, i: int):
        """returns density for a specific band"""
        if self.bands_range[i][0] < energy < self.bands_range[i][1]:
            return np.mean(self.smearing[i].dvolume(energy))
        return 0.0

    def density_bands(self, energy: float):
        """returns density but split into the band contributions"""
        return [self.density_band(energy, i) for i in range(self.model_bandcount())]

    def cut_band_indices(self, energy: float):
        """get the indices of the bands, that cut the given energy"""
        indices = []
        for i, band_range in enumerate(self.bands_range):
            if band_range[0] < energy < band_range[1]:
                indices.append(i)
        return indices

    def full_curve(self, N=10, T=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the full density of states curve for a given electronic temperature.

        Args:
            N (int, optional): Number of points per band. Defaults to 10.
            T (float, optional): Temperature for the calculation. Defaults to 0.

        Returns:
            tuple[ndarray(N_e), ndarray(N_e), ndarray(N_e)]: energy samples, states, density
        """
        # compute energy samples based on where the bands start and end
        energy_smpl = []
        for (min_e, max_e) in self.bands_range:
            energy_smpl.extend(np.linspace(min_e, max_e, N))
        energy_smpl = np.unique(energy_smpl)
        states = []
        density = []
        for energy in energy_smpl:
            states_, density_ = self.states_density(energy)
            states.append(states_)
            density.append(density_)
        if T != 0:
            # smooth the curve using the Fermi-distribution
            beta = 1 / (k_B * T)  # 1/eV
            energy_smpl_T0, states_T0, density_T0 = energy_smpl, np.asarray(states), np.asarray(density)
            delta = 5.0 / beta
            energy_smpl = np.concatenate([np.linspace(energy_smpl[0] - delta, energy_smpl[0], N, endpoint=False), energy_smpl, np.flip(np.linspace(energy_smpl[-1] + delta, energy_smpl[-1], N, endpoint=False))], axis=0)
            states = [convolve_df(x, energy_smpl_T0, states_T0, beta) for x in energy_smpl]
            density = [convolve_df(x, energy_smpl_T0, density_T0, beta, extrapolation=None) for x in energy_smpl]
            # density = [convolve_ddf(x, energy_smpl_T0, states_T0, beta) for x in energy_smpl]
        return np.array(energy_smpl), np.array(states), np.array(density)

    def k_resolved_density(self, mu: float, T: float, N=10):
        """Compute the k-resolved density of states, that would be
        visible with ARPES (angle-resolved photoemission spectroscopy).

        Args:
            mu (float): The chemical potential/energy.
            T (float): temperature.
            N (int, optional): _description_. Defaults to 10.

        Returns:
            ndarray[N_kx, N_ky, N_kz]: The density of states, matching `self.k_smpl`.
        """
        assert T >= 0, "Temperature must be non-negative"
        if T == 0:
            return sum([s.dvolume(mu) for s in self.smearing])
        e0, e1 = self.energy_range()
        beta = 1 / (k_B * T)  # 1/eV
        energy_smpl = _samples(e0, e1, mu, beta, N)
        states = np.stack([sum([s.volume(e) for s in self.smearing]) for e in energy_smpl], axis=-1)
        return convolve_ddf(mu, energy_smpl, states, beta)

    def fermi_energy(self, electrons: float, tol=1e-8, maxsteps=30) -> float:
        """Compute the Fermi-energy (chemical potential at T=0) for a given number of electrons per cell.
        The number of electrons can be fractional to correctly handle doping or spin.

        Args:
            electrons (float): The number of valence electrons per cell.
            tol (float, optional): The precision of the root-finding procedure. Defaults to 1e-8.
            maxsteps (int, optional): The maximal number of steps for the root-finding procedure. Defaults to 30.

        Raises:
            ValueError: If the root search failed.

        Returns:
            float: The Fermi-energy.
        """
        assert electrons >= 0 and electrons <= len(self.bands_range), f'"electrons" (got {electrons:.2f}) must be between 0 and the number of bands ({len(self.bands_range)})'
        # check if the material is an isolator
        e_int = round(electrons)
        if e_int == electrons:
            if e_int > 0 and e_int < len(self.bands_range):
                max_below = self.bands_range[e_int - 1][1]
                min_above = self.bands_range[e_int][0]
                fermi_energy = (max_below + min_above) / 2
                if max_below < min_above:
                    return fermi_energy  # isolator (can only happen at integer electrons)
            elif e_int == 0:
                return self.bands_range[0][0]
            else:
                assert e_int == len(self.bands_range)  # already checked above, it's here just as a reminder
                return self.bands_range[-1][1]
        # first approximation from a very rough states curve
        e_smpl, states_smpl, _ = self.full_curve(N=2)
        e_index = list(states_smpl > electrons).index(True) - 1
        fermi_energy = (electrons - states_smpl[e_index]) / (states_smpl[e_index + 1] - states_smpl[e_index]) * (e_smpl[e_index + 1] - e_smpl[e_index]) + e_smpl[e_index]
        # now do a couple newton steps to find the exact value
        for _ in range(maxsteps):
            states, density = self.states_density(fermi_energy)
            # TODO handle density = 0 in metals
            fermi_energy -= (states - electrons) / density
            if abs((states - electrons) / density) <= tol:
                return fermi_energy
        raise ValueError(f"root search didn't converge in {maxsteps} steps. {abs((states - electrons) / density)} > {tol}.")

    def bandgap(self, electrons: float) -> float:
        """Return the (already precomputed) bandgap.

        Args:
            electrons (float): The number of valence electrons per cell. If this is not an integer, there is no gap, so this function returns 0.

        Returns:
            float: The size of the bandgap.
        """
        # first approximation from integer number of electrons
        assert electrons >= 0 and electrons <= len(self.bands_range)
        e_int = round(electrons)
        if e_int != electrons:
            return 0.0  # only integers can make isolators
        assert e_int > 0 and e_int < len(self.bands_range)
        max_below = self.bands_range[e_int - 1][1]
        min_above = self.bands_range[e_int][0]
        if max_below < min_above:
            return min_above - max_below  # isolator
        return 0.0  # metal

    def energy_range(self) -> tuple[float, float]:
        """Get the minimal and maximal energy in the used band structure.

        Returns:
            tuple[float, float]: minimal energy, maximal energy
        """
        return self.bands_range[0][0], self.bands_range[-1][1]

    def chemical_potential(self, electrons: float, T_smpl, N=30, tol=1e-8, maxsteps=30) -> np.ndarray[Any, np.dtypes.Float64DType]:
        """Compute the chemical potential for a given number of electrons per cell at multiple temperatures.
        The number of electrons can be fractional to correctly handle doping or spin.

        Note: when batching the temperatures in one call, this method is faster,
          however the approximations involved will slightly differ
          (with an error in the same order of magnitude).

        Args:
            electrons (float): The number of valence electrons per cell (= filled bands).
            T_smpl (arraylike[N_T]): The (sorted!) temperatures for batched computation of chemical potentials.
            N (int, optional): The number of points used to approximate the smeared states. Defaults to 30.
            tol (float, optional): The precision of the root-finding procedure. Defaults to 1e-8.
            maxsteps (int, optional): The maximal number of steps for the root-finding procedure. Defaults to 30.

        Returns:
            ndarray[N_T]: The Fermi-energy.
        """
        fermi_energy = self.fermi_energy(electrons, tol=tol, maxsteps=maxsteps)
        # use a good distribution for energy_smpl,
        # such that this works for small T!
        # the distribution needs to distribute the error terms
        # to be equal on all summands in convolve_df
        # To optimize the performance, all T > ... can be performed with the same distribution linear distribution
        e0, e1 = self.energy_range()
        # any temperature bigger than this can be sampled linear without precision loss
        T_lin = 0.25 * 0.25 * 2 / k_B * max(abs(e0 - fermi_energy), abs(e1 - fermi_energy))
        energy_smpl_lin = np.linspace(e0, e1, N)
        T_smpl = np.asarray(T_smpl).reshape(-1)
        if np.max(T_smpl) >= T_lin:
            states_lin = np.array([self.states_below(energy) for energy in energy_smpl_lin])
        beta = -10000.0  # invalid value
        res = []
        for T in T_smpl:
            if T <= 0:
                res.append(fermi_energy)
            elif T >= T_lin:
                beta = 1 / (k_B * T)  # 1/eV
                res.append(secant(electrons, lambda x: convolve_df(x, energy_smpl_lin, states_lin, beta, extrapolation='flat'), fermi_energy, fermi_energy + 0.1, tol, maxsteps))
            else:
                new_beta = 1 / (k_B * T)  # 1/eV
                if abs(beta - new_beta) / new_beta > 5e-2:
                    # recalculate distribution for precision
                    beta = 0.25 / (k_B * T)  # 1/eV
                    energy_smpl = _samples(e0, e1, fermi_energy, beta, N)
                    beta = new_beta
                    states = np.array([self.states_below(energy) for energy in energy_smpl])
                # keep distribution if it doesn't cause too big errors (performance)
                beta = new_beta
                res.append(secant(electrons, lambda x: convolve_df(x, energy_smpl, states, beta, extrapolation='flat', extrapolation_point=(e1, self.model_bandcount())), fermi_energy, fermi_energy + 1 / beta, tol, maxsteps))
        return np.array(res)

    def energy(self, mu: float, T=0.0, N=256, spin_factor=2) -> float:
        """Compute the total energy of the system defined as: integral e * density(e) * f(e) de.
        This is done in the most numerically accurate way.

        Args:
            mu (float): The chemical potential.
            T (float, optional): Temperature to compute the energy at. Defaults to 0.0K.
            N (int, optional): Number of samples of the DoS to use. Defaults to 32.
            spin_factor (int, optional): The number of electrons per band. Defaults to 2.

        Returns:
            float: The total energy of the system.
        """
        # energy is integral e f(e) rho(e) de
        # = [...] - integral (f + e df/de) N(e) de
        # The constant from partial integration is 0 if the full integral from -inf to inf is computed like here.

        # if the temperature is zero, then just integrate the states curve using linear segments.
        # The integral above simplifies to
        # integral -(f + e df/de) N(e) de = mu N(mu) - integral_{-oo}^mu N(e) de
        # TODO it would be good to compute the total energy of each band beforehand and reuse it here
        e0, e1 = self.energy_range()
        mu = float(mu)
        # instead of equally distributed samples, use samples, which match band corners, such that the flat parts of N(e) are correctly integrated as flat.
        energy = 0
        for i, (min_e, max_e) in enumerate(self.bands_range):
            if min_e < mu:
                e_smpl = np.linspace(min_e, min(max_e, mu), N//2*2+1)
                states = np.array([self.states_below_band(e, i) for e in e_smpl])
                # trapezoidal rule (for arbitrary sorted lists of e_smpl)
                int1 = np.sum((states[:-1] + states[1:])/2 * (e_smpl[1:] - e_smpl[:-1]))
                # Romberg extrapolation with just one more step -> ~ Simpson rule.
                # this assumes that the density is continuous, which is the case for 3D bandstructures.
                int2 = np.sum((states[:-2:2] + states[2::2])/2 * (e_smpl[2::2] - e_smpl[:-2:2]))
                energy += mu * states[-1] - (int1 + (int1 - int2)/(4 - 1))
        # TODO allow more than one T to be used!
        if T <= 0.0:
            # this part is pretty well convergent
            return energy * spin_factor
        
        # for high temperatures, one can use the naive formula (see `naive_energy`)
        # however a different approach is taken here, which works better for small
        # temperatures, but worse for large temperatures.

        # now compute the correction due to temperature (using f0 as the Fermi-distribution
        # for T=0, but still centered on mu, as that was used above).
        # like usual assume N(e) to be piecewise linear and do the remaining integral
        # = -integral (f-f0(e-mu) + e (df/de + delta(e-mu))) (a e + b) de
        # = -integral (a e + b) (f-f0) + (a e^2 + b e) df/de de {- (a mu^2 + b mu) if e0<mu<=e1}  | next: integration by parts
        # = -[(a e^2/2 + b e) (f-f0)] - integral -(a e^2/2 + b e) (df/de + delta(e-mu)) + (a e^2 + b e) df/de de {- (a mu^2 + b mu) if e0<mu<=e1}
        # = -[(a e^2/2 + b e) (f-f0)] + integral a e^2/2 (-df/de) de {- a mu^2/2 if e0<mu<=e1}
        # the additional term with mu appears only once in the entire integration, so it is added at the end using the exact a=rho(mu).
        # A substitution is necessary, as the integrals are defined for f at beta=1 and mu=0, so substitute e->x/beta+mu, x=(e-mu)*beta
        # = -[(a e^2/2 + b e) (f-f0)] + integral a/2 (x/beta + mu)^2 (-df/dx(x) * beta) dx/beta {- a mu^2/2 if e0<mu<=e1}

        beta = 1 / (k_B * T)  # 1/eV
        def accumulate(e_smpl, states):
            def segment(e0, e1, y0, y1):
                dx = e1 - e0
                dy = y1 - y0
                a = dy / dx
                b = y0 - e0 * a
                def f(e):
                    return (a * e / 2 + b) * e * (scipy.special.expit(-(e - mu) * beta) - np.where(e < mu, 1, 0))
                simple_term = f(e1) - f(e0)
                return -simple_term + a / 2 * int_poly((e0 - mu) * beta, (e1 - mu) * beta, 1/beta**2, 2*mu/beta, mu*mu)
            s = np.sum(segment(e_smpl[:-1], e_smpl[1:], states[:-1], states[1:]))
            # add segments/other functions as extrapolation on both side here to get correct high temperature behavior
            s += segment(e_smpl[-1], e_smpl[-1] + 8. / beta, self.model_bandcount(), self.model_bandcount()) # flat N(e) above the energy range
            return s
        # calculate custom distribution for precision/cost balance
        e_smpl = _samples(e0, e1, mu, beta, N)
        states = np.array([self.states_below(energy) for energy in e_smpl])
        return (energy + accumulate(e_smpl, states) - self.density(mu) * mu*mu/2) * spin_factor

    def seebeck(self, electrons: float, T_smpl: Iterable[float], N=30, h=1e-4) -> np.ndarray:
        """The derivative of the chemical potential w.r.t. temperature.
        It can be interpreted as part of the Seebeck-coefficient.
        (See https://doi.org/10.1103/PhysRevB.98.224101)

        Args:
            electrons (float): Number of filled bands, just like for the chemical potential calculation.
            T_smpl (arraylike): Temperatures for the computation
            N (int, optional): The number of energy samples to use for the integration. Defaults to 30.

        Returns:
            float: The derivative of the chemical potential
        """
        # for T=0 the seebeck coefficient is 0
        T_smpl = np.asarray(T_smpl)
        subset = np.arange(len(T_smpl))[T_smpl > 0]
        T_ = T_smpl[subset]
        inv_order = np.zeros(len(T_smpl), dtype=int)
        inv_order[subset] = np.arange(len(T_))
        assert h < np.min(T_), "h needs to be bigger than the smallest positive temperature"
        T_h = [T + s * h for T in T_ for s in [-1, 1]]
        # numerical derivative of the chemical potential
        mu = self.chemical_potential(electrons, T_h, N=N)
        # Note: the result would be multiplied by 1eV/e = 1V, so this depends on the unit eV!
        return np.array([(mu[inv_order[i] * 2 + 1] - mu[inv_order[i] * 2]) / (2 * h) if T_smpl[i] > 0 else 0 for i in range(len(T_smpl))])

    # heat capacity at constant volume in eV/K
    def heat_capacity(self, electrons: float, T: float, N=256, spin_factor=2, h=1e-4) -> float:
        # The heat capacity is du/dT where u is the energy per unit cell
        # u = integral e f(e) rho(e) de
        # = [...] - integral (f + e df/de) N(e) de
        # the temperature derivative can be broken up into the part from the change in chemical potential
        # and the part from the change f(e)

        # du/dT = integral rho(e)*e*(e - mu(T) + mu'(T)*k_B*T)/T*(-df/de(e)) de
        # = integral rho(e)*e*(e - mu2)/T*(-df/de(e)) de
        # for 3D bandstructures, rho(e) is continuous,
        # so this can solved using a piecewise linear interpolaton of rho(e).
        # du/dT = 1/T * integral rho(e)*e*(e - mu(T) + mu'(T)*k_B*T) * w de
        # the resulting integral has terms up to e^2, for which the analytical
        # solution is implemented. To use it, the substitution e = x/beta + mu must be made.
        # -> e*(e-mu2) -> (x/beta + mu)*(x/beta - mu'*k_B*T) = x^2/beta^2 + (mu-mu'*k_B*T)*x/beta + mu*mu'*k_B*T
        # Note that rho should be approximated linear as rho(e) = a*(e-e0)+b, which makes the integral cubic in x.
        # Then, one has to Taylor to get a quadratic approximation centered in the relevant interval.

        # TODO allow more than one T to be used!
        if T <= 0.0:
            return 0.0
        
        e0, e1 = self.energy_range()
        # numerical derivative of the chemical potential
        dT = h * T
        mu = self.chemical_potential(electrons, [T - dT, T, T + dT], N=N)
        seebeck = (mu[2] - mu[0]) / (2 * dT) * k_B * T
        mu = mu[1]
        mu2 = mu - seebeck
        
        beta = 1 / (k_B * T)  # 1/eV
        def accumulate(e_smpl, rho):
            def segment(e0, e1, y0, y1):
                dx = e1 - e0
                dy = y1 - y0
                em = (e0 + e1) / 2
                a = dy / dx
                b = (y0 + y1) / 2
                # rho(e) = a*(e-em)+b
                a0 = (a*(2*em - mu + seebeck) + b)/beta**2
                b0 = (a*(-3*em**2 - em*seebeck - 2*mu**2 + mu*(5*em + 2*seebeck)) + b*(mu + seebeck))/beta
                c0 = a*(em**3 - mu**3 + mu**2*(3*em + seebeck) + mu*(-3*em**2 - em*seebeck)) + b*mu*seebeck
                return int_poly((e0 - mu) * beta, (e1 - mu) * beta, a0, b0, c0)
            # above and below energy range, this is just zero, so no extrapolation needed.
            return np.sum(segment(e_smpl[:-1], e_smpl[1:], rho[:-1], rho[1:]))
        # calculate custom distribution for precision/cost balance
        e_smpl = _samples(e0, e1, mu, beta, N)
        rho = np.array([self.density(energy) for energy in e_smpl])
        return accumulate(e_smpl, rho) / T * spin_factor

    def conductivity(self, mu):
        # TODO conductivity from the naive sum
        return None

    def fermi_surface_samples(self,
                              energy: float,
                              improved_points=True,
                              improved_weights=False,
                              weight_by_gradient=False,
                              normalize=None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute points on the (fermi) surface at the given (fermi) energy.
        If the improved keyword argument is True, the results will be much more precise
        at the cost of an integration over the fermi surface.

        Args:
            energy (float): The chemical potential energy at which the Fermi-surface is computed.
            improved_points (bool, optional): If True, a Newton step is used to improve the precision of the Fermi-surface. Defaults to True.
            improved_weights (bool, optional): If True, the weights will be recomputed to make integrals over the Fermi-surface more precise. Defaults to False.
            weight_by_gradient (bool, optional): If True, the weights will include a reciprocal gradient term as in area/norm(grad). This is useful for integration. Defaults to False.
            normalize (str, optional): Either of [None, "band", "total"].
                    None means the weights match the k-space area associated with each point.
                    "band" means that the weights in a band will sum up to 1.
                    "total" means that all bands will sum up to 1.
                    Defaults to None.

        Returns:
            tuple[ndarray(N, 3), ndarray(N), ndarray(N), float]: points, band_indices, weights, total area
        """
        assert self.model is not None, "This function needs an underlying model"
        assert normalize in [None, "band", "total"], "normalize needs to be None, 'band' or 'total'"
        if not isinstance(self.smearing[0], CubesSmearing):
            raise NotImplementedError("This function is not yet implemented for anything but cubes")
        # use cube_cut_area_com() to get points, then
        # use the approximate gradients to do a newton step
        # to get the points even closer to the fermi surface.
        weights = []
        band_indices = []
        points = []
        total_w_sum = 0
        for i in self.cut_band_indices(energy):
            x, grads, w = self.smearing[i].samples(energy)
            # HACK: scale up the weights to fill the entire briloun zone.
            # The assumption here is, that if the k_range is anything but [-0.5,0.5]^3,
            # then that is, because the other parts are equal and can be reproduced by symmetry
            w *= np.linalg.det(self.A) / self.crystal_volume
            if improved_points:
                if "bands_grad" in dir(self.model):
                    # if available, evaluate the model with bands and exact gradients (way more precise!)
                    bands, grads = self.model.bands_grad(x)
                    bands = bands[:, i:i + 1] - energy
                    grads = grads[..., i]
                else:
                    # evaluate model, but take the gradients from the known approximations
                    bands = self.model(x)[:, i:i + 1] - energy
                x -= grads * (bands / (1e-20 + np.linalg.norm(grads, axis=-1, keepdims=True)**2))
            # don't use gradients when computing total area
            w_sum = np.sum(w)
            total_w_sum += w_sum
            if weight_by_gradient and not improved_weights:
                # NOTE: the gradients could have changed here if improved_points=True
                # This division is in reciprocal space.
                w /= np.linalg.norm(grads, axis=-1) + 1e-20
            if normalize == 'band':
                w /= w_sum
            points.extend(x)
            weights.extend(w)
            band_indices.extend([i] * len(w))
        assert len(points) == len(weights)
        total_area = total_w_sum
        points = np.reshape(points, (-1, 3))
        band_indices = np.array(band_indices, dtype=np.int64)
        if improved_weights:
            # The optimal solution here is triangulation, however the whole point of the cutting cubes was to avoid that!

            # It's not possible to know the results for plane waves...
            # Some results however are possible to know based on symmetries, if they would be supplied.

            # approximate delta-functions (gaussians) have a known approximate result.
            # -> maybe that could be enough to improve the weights slightly. (use scipy.sparse)
            #   -> problem: it becomes a volume integral instead of a surface integral... This is bad for half metals.

            # N = (len(points) + 7) // 8 * 8 # use at least as many test functions as points to make the linear function invertible!
            # def func(x):
            #    return x # TODO
            # w = conjugate_gradient_solve(func, np.ones(), np.array(weights))

            # The cutting cube weight is computes in a fixed grid above. The points and their derivatives are known.
            # What if one would average over multiple shifted cube grids to get the best weights??? I think that could work well!
            # -> use the best available gradients for that!
            # -> I don't think it's a problem if two points share a cube in this context.
            # -> call this anti-aliasing and do 8 samples (no evaluation of the model = relatively cheap)
            # -> TODO think about shifting the points as well using this method. Move points to mean COM, Newton step.
            # -> think about handling duplicate surfaces correctly. I.e. (I'm assumming rare) cases where two cubes produce the same fermi surface points.
            if "bands_grad" in dir(self.model):
                # if available, evaluate the model with bands and exact gradients (way more precise!)
                bands, grads = self.model.bands_grad(points)
                bands = np.take_along_axis(bands, band_indices[:, None], axis=-1)[..., 0]
                grads = np.take_along_axis(grads, band_indices[:, None, None], axis=-1)[..., 0]
                # do another newton step here for free! (Assume the gradient is constant)
                points -= grads * (bands / (1e-20 + np.linalg.norm(grads, axis=-1)**2))[..., None]
            else:
                raise NotImplementedError("No implementation to get gradients from the lattice.")
            # TODO WRONG for A != 1I
            ax, ay, az = tuple(grads.T * self.step_sizes[:, None])  # transformed for cubes
            w = np.zeros(len(weights))
            n = 2
            offsets = np.stack(np.meshgrid(*3 * [np.linspace(-0.5, 0.5, n, endpoint=False)])).reshape(-1, 3)
            for offset in offsets:
                # compute cube indices of the shifted points
                cube_indices = np.round(offset + points / self.step_sizes) - offset
                # only use one point per cube, or multiple if the gradients sufficiently disagree.
                select = np.zeros(len(cube_indices), dtype=bool)
                select[np.unique(cube_indices, axis=0, return_index=True)[1]] = True
                # compute a0 (band value at the center of the cube) assuming it's 0 at the known points
                # TODO figure out the sign here (for weight_by_gradient=True ???)
                a0 = np.sum(grads * self.step_sizes * (cube_indices * self.step_sizes - points), -1)[select]
                # TODO figure out how to compute the area for cuboids instead of cubes
                if weight_by_gradient:
                    w[select] += cube_cut_dvolume(a0, ax[select], ay[select], az[select])
                else:
                    # TODO COM calculation is not needed
                    w[select] += cube_cut_area_com(a0, ax[select], ay[select], az[select])[0]
            w /= len(offsets)  # mean
            total_w_sum = np.sum(w)
            if normalize == 'band':
                # TODO apply band normalisation
                raise NotImplementedError("band normalisation with improved_weights is currently not implemented")
        else:
            w = np.array(weights)
        # apply normalisation
        if normalize == 'total':
            w /= total_w_sum
        return points, band_indices, w, total_area
