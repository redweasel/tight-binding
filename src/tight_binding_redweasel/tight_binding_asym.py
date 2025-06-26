import numpy as np
from typing import Self
from collections.abc import Sequence, Iterable
from .symmetry import *
from . import logger
from . import json_tb_format
from . import wannier90_tb_format as tb_fmt
from .linalg import *
from .loss import *
import time


class HermitianFourierSeries:
    """
    This class represents a fourier series with a hermitian output for each input position k.

    For this it needs the neighbors `r`, such that `H_k = sum_r H_r exp(2πi k·r)`.
    To make sure that the result is hermitian, it is processed as `H'_k = H_k + H_k^+ + H_0`
    where `H_0` is referring to the matrix at `r=0`, which is skipped in the sum above.
    """

    def __init__(self, neighbors: Sequence, H_r: Sequence):
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        assert len(neighbors) == len(H_r)
        assert np.shape(H_r)[1] == np.shape(H_r)[2]
        assert len(np.shape(neighbors)) == 2
        assert len(np.shape(H_r)) == 3
        self.neighbors = np.asarray(neighbors, dtype=float)
        self.H_r = np.asarray(H_r)

    def dim(self) -> int:
        """
        Returns:
            int: Dimension of the neighbor positions. Usually 1, 2 or 3.
        """
        return len(self.neighbors[0])

    def f_i(self, k: Iterable, i: int) -> np.ndarray:
        """Exponential function that appears as coefficient in the fourier series."""
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        return np.exp(2j*np.pi*(k @ r))

    def df_i(self, k: Iterable, i: int) -> np.ndarray:
        """Derivative of f_i"""
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        return 2j*np.pi * np.exp(2j*np.pi*(k @ r))[..., None] * np.asarray(r)

    def ddf_i(self, k: Iterable, i: int) -> np.ndarray:
        """Second derivative of f_i"""
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        r_sqr = np.asarray(r)[:, None] * np.asarray(r)[None, :]
        return (2j*np.pi)**2 * np.exp(2j*np.pi*(k @ r))[..., None, None] * r_sqr

    def f(self, k: Sequence) -> np.ndarray:
        """Compute the hermitian (N, N) matrix"""
        mat = np.zeros(np.shape(k)[:-1] + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = (1,)*len(np.shape(k)[:-1]) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += self.f_i(k, i)[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        # expect H_r[0] to always be hermitian! (this is very important for symmetrization)
        mat += self.f_i(k, 0)[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat

    def df(self, k: Sequence) -> np.ndarray:
        """Compute the derivative of f wrt k in the direction dk, outputshape (dim(k), N, N)"""
        mat = np.zeros(np.shape(k) + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = (1,)*len(np.shape(k)) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += self.df_i(k, i)[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += self.df_i(k, 0)[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat

    def ddf(self, k: Sequence) -> np.ndarray:
        """Compute the second derivative of f wrt k in the direction dk, outputshape (dim(k), dim(k), N, N)"""
        mat = np.zeros(np.shape(k) + (np.shape(k)
                       [-1],) + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = tuple([1]*(1+len(np.shape(k)))) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += self.ddf_i(k, i)[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += self.ddf_i(k, 0)[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat

    def copy(self) -> 'HermitianFourierSeries':
        return HermitianFourierSeries(self.neighbors.copy(), self.H_r.copy())

    @staticmethod
    def unit_matrix(neighbors: Sequence, n: int) -> 'HermitianFourierSeries':
        """Initialize the fourier series, such that it equals the identity for all k.

        Args:
            neighbors (arraylike): Positions for the fourier transform. (see class documentation)
            n (int): Matrix dimension.

        Returns:
            Self: identity fourier series
        """
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        H_r = np.zeros((len(neighbors), n, n))
        H_r[0] = np.eye(n)
        return HermitianFourierSeries(neighbors, H_r)

    def direct_sum(self, other: 'HermitianFourierSeries') -> 'HermitianFourierSeries':
        if np.any(self.neighbors != other.neighbors):
            raise NotImplementedError("mixing neighbor sets in the direct sum is currently not implemented.")
        return HermitianFourierSeries(self.neighbors, direct_sum(self.H_r, other.H_r))

    def add_neighbors(self, neighbors: Sequence):
        """Add more neighbors with zero coefficients.

        Args:
            neighbors (arraylike(N_R, dim)): The neighbors to be added.
        """
        for n in neighbors:
            if np.min(np.linalg.norm(self.neighbors - n, axis=-1)) < 1e-5:
                if len(neighbors) == 1:
                    print(f"Warning: ignore duplicate neighbor {n}")
                    return
                else:
                    raise ValueError("redundant/duplicate neighbors are not allowed")
        self.neighbors = np.concatenate([self.neighbors, neighbors], axis=0)
        self.H_r = np.concatenate([self.H_r, np.zeros((len(neighbors),) + self.H_r.shape[1:])], axis=0)

    def limit_neighbors(self, max_length: float) -> int:
        """Remove all neighbors that have a vector length more than a given threshold length.

        Args:
            max_length (float): Maximal length/distance for the neighbors. Everything else gets cut off. Exact values like 2 are safe, as there is a buildin epsilon.

        Returns:
            int: number of removed neighbors
        """
        assert max_length >= 0.0, "can only limit with non negative maximal length, as the 0 neighbor needs to be kept"
        keep = np.linalg.norm(self.neighbors, axis=-1) <= max_length + 1e-8
        old_len = len(self.neighbors)
        self.neighbors = np.array(self.neighbors[keep])
        self.H_r = np.array(self.H_r[keep])
        return old_len - len(self.neighbors)

    def limit_neighbor_count(self, max_count: int) -> int:
        """Remove all neighbors that exceed the targeted number of neighbors.

        Args:
            max_count (int): Maximal number of neighbors to be kept.

        Returns:
            int: number of removed neighbors
        """
        # sort neighbors by length first and trim the index list
        sort = np.argsort(np.linalg.norm(self.neighbors, axis=-1))[:max_count]
        old_len = len(self.neighbors)
        self.neighbors = np.array(self.neighbors[sort])
        self.H_r = np.array(self.H_r[sort])
        return old_len - len(self.neighbors)

    def cleanup_neighbors(self, min_norm: float) -> int:
        """Remove all neighbors that have a coefficient matrix with a norm smaller than a given threshold.

        Args:
            min_norm (float): Minimal norm for the neighbor coefficients. Everything else gets cut off.

        Returns:
            int: number of removed neighbors
        """
        keep = np.linalg.norm(self.H_r, axis=(-1, -2)) >= min_norm
        old_len = len(self.neighbors)
        keep[0] = True  # always keep the 0
        self.neighbors = np.array(self.neighbors[keep])
        self.H_r = np.array(self.H_r[keep])
        return old_len - len(self.neighbors)

    def __add__(self, other: 'HermitianFourierSeries') -> 'HermitianFourierSeries':
        # do the direct sum of the two series.
        # 1. bring them to the same number of neigbors
        # 2. do the direct sum for the coefficient matrices
        assert np.all(self.neighbors == other.neighbors), "currently only supported for equal neighbors"
        return HermitianFourierSeries(self.neighbors, direct_sum(self.H_r, other.H_r))

class AsymTightBindingModel:
    """
    This class represents a tight binding model without any symmetrisation options.
    Instead it has the option to add an orbital overlap matrix S.
    """

    def __init__(self, H: HermitianFourierSeries, S=None):
        assert type(H) == HermitianFourierSeries
        assert S is None or type(S) == HermitianFourierSeries
        self.H = H
        self.S = HermitianFourierSeries.unit_matrix([(0,)*H.dim()], np.shape(H.H_r)[1]) if S is None else S

    def new(neighbors: np.ndarray, band_count: int) -> Self:
        """Create a new empty model.

        Args:
            neighbors (arraylike(N_R, dim_k)): Positions of the used matrices
            band_count (int): Number of bands in the model.

        Returns:
            Self: An empty model with bandstructure = 0 for all bands.
        """
        return AsymTightBindingModel(HermitianFourierSeries(neighbors, np.zeros((len(neighbors), band_count, band_count), dtype=complex)))

    def band_count(self):
        """
        Returns:
            int: Number of bands of the model. Often referred to as N_b.
        """
        return len(self.H.H_r[0])

    def dim(self):
        """
        Returns:
            int: Dimension of the neighbor positions. Usually 1, 2 or 3.
        """
        return self.H.dim()

    def copy(self):
        return AsymTightBindingModel(self.H.copy(), self.S.copy())

    def init_from_ref(neighbors, k_smpl, ref_bands, use_S=False) -> Self:
        """Initialize a tight binding model from reference data."""
        assert len(k_smpl[0]) == len(neighbors[0])
        dim = len(neighbors[0])
        param_count = len(neighbors)
        # first matrix is always constant term
        assert np.linalg.norm(neighbors[0]) == 0
        # make the model correct for k=0
        k0_index = np.argmin(np.linalg.norm(k_smpl, axis=-1))
        assert np.linalg.norm(k_smpl[k0_index]) == 0
        k0_bands = ref_bands[k0_index]
        model_bands = k0_bands
        # add random matrices for the k dependence to break the gradient descent subspace
        # TODO make this per band as otherwise it will become really large for far apart bands
        scale = (np.max(model_bands) - np.min(model_bands)) * 0.001
        H_r = [np.diag(model_bands)] + [random_hermitian(len(model_bands)) * scale for _ in range(param_count-1)]
        if use_S:
            tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r), HermitianFourierSeries.unit_matrix(neighbors, len(H_r[0])))
        else:
            tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        tb.normalize()
        return tb

    def save(self, filename: str, format=None):
        if format is None:
            if filename.endswith(".json"):
                format = "json"
            else:
                raise ValueError("unrecognised format for file " + filename)
        if format not in {"json"}:
            raise ValueError('only supported format is "json"')
        opt = np.get_printoptions()
        np.set_printoptions(precision=None, suppress=False, floatmode="unique", threshold=100000, legacy='1.21')
        if format == "json":
            json_tb_format.save(filename, self.H.neighbors, self.H.H_r)
            # TODO save the S matrix part as well, maybe in a separate file? Or extend the format...

        np.set_printoptions(**opt)  # reset printoptions

    @staticmethod
    def load(filename: str, format=None) -> Self:
        if format is None:
            if filename.endswith(".repr"):
                format = "python"
            elif filename.endswith(".json"):
                format = "json"
            elif filename.endswith("tb.dat"):
                format = "wannier90tb"
            elif filename.endswith("hr.dat"):
                format = "wannier90hr"
            else:
                raise ValueError("unrecognised format for file " + filename)
        if format not in {"python", "json", "wannier90hr", "wannier90tb"}:
            raise ValueError('supported formats are "python", "json", "wannier90hr" and "wannier90tb", but was ' + str(format))
        if format == "python":
            with open(filename, "r") as file:
                H_r_repr = " ".join(file.readlines())
                H_r, neighbors, S, inversion = eval(H_r_repr.replace("array", "np.array"))
                # convert to this asymmetric format!
                sym = Symmetry(S, inversion=inversion)
                if len(sym) > 1:
                    # multiply the parameters with correct weights
                    weights = Symmetry(S, inversion=False).r_class_size(neighbors)
                    H_r[1:] /= weights[1:, None, None]
                    # complete neighbors and H_r using sym
                    complete_neighbors, order = sym.complete_neighbors(neighbors, return_order=True)
                    model = AsymTightBindingModel(HermitianFourierSeries(complete_neighbors, H_r[order]))
                    raise ValueError("input with coefficient symmetry is currently not supported as there is some inconsistency somewhere")
                else:
                    model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        elif format == "wannier90hr":
            neighbors, H_r = tb_fmt.load_hr(filename)
            model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        elif format == "wannier90tb":
            neighbors, H_r, w_r_params, degeneracies, A = tb_fmt.load_tb(filename)
            model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        elif format == "json":
            neighbors, H_r = json_tb_format.load(filename)
            model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        return model

    def randomize(self, sigma, keep_zeros=False, randomize_S=False):
        """
        Randomize parameters with normal distributed numbers with a standard deviation sigma.

        **This can break symmetry in bad ways -> call `self.normalize()` manually after this!**
        """
        dparams = sigma * np.random.standard_normal(self.H.H_r.shape) + sigma * 1j*np.random.standard_normal(self.H.H_r.shape)
        if keep_zeros:
            dparams *= np.where(np.abs(self.H.H_r) < 1e-8, 0, 1)
        self.H.H_r += dparams
        if randomize_S:
            dparams = sigma * np.random.standard_normal(self.S.H_r.shape) + sigma * 1j*np.random.standard_normal(self.S.H_r.shape)
            if keep_zeros:
                dparams *= np.where(np.abs(self.S.H_r) < 1e-8, 0, 1)
            self.S.H_r += dparams

    def error(self, k_smpl, ref_bands, band_weights, band_offset):
        """
        Returns:
            (float, ndarray(N_b)): the weighted loss (standard deviation) and the maximal error per band
        """
        return model_error(self(k_smpl), ref_bands, band_weights, band_offset)

    def loss(self, k_smpl, ref_bands, band_weights, band_offset):
        """
        Returns:
            float: the weighted loss (standard deviation)
        """
        return model_loss(self(k_smpl), ref_bands, band_weights, band_offset)

    def windowed_loss(self, k_smpl, ref_bands, min_energy, max_energy, allow_skipped_bands=False):
        """
        Returns:
            float: the windowed loss (standard deviation)
        """
        return model_windowed_loss(self(k_smpl), ref_bands, min_energy, max_energy, allow_skipped_bands=allow_skipped_bands)

    def print_error(self, k_smpl, ref_bands, band_weights, band_offset, prefix="", log=None):
        """Print the loss and the maximal error per band"""
        band_weights = np.broadcast_to(np.ravel(band_weights / np.mean(band_weights)), (len(ref_bands[0]),))
        l, err = self.error(k_smpl, ref_bands, band_weights, band_offset)
        if log is not None:
            log.add_message(f"{prefix}loss: {l:.2e} (max band-error {err})")
        else:
            print(f"{prefix}loss: {l:.2e} (max band-error {err})")

    # TODO remove _ which was there for k_smpl weights before, those can and should be computed in this function
    def optimize(self, k_smpl, _, ref_bands, band_weights, band_offset: int, iterations: int, batch_div=1, learning_rate=1.0, train_S=False, use_pinv=True, use_lstsq_stepsize=False, max_accel_global=None, regularization=1.0, keep_zeros=False, convergence_threshold=1e-3, loss_threshold=1e-16, log=True):
        N = np.shape(self.H.H_r)[1]
        assert band_offset >= 0 and band_offset <= N - len(ref_bands[0]), f"band_offset={band_offset} must be in [0, {N-len(ref_bands[0])}]"
        if log == False:
            # logger that doesn't print
            log = logger.OptimisationLogger(print_loss=False, verbose=False)
        elif log == True:
            # logger that prints
            log = logger.OptimisationLogger(print_loss=True, update_line=True, verbose=True)
        # reshape normalized band_weights
        band_weights_norm = np.max(band_weights)**2
        band_weights = np.broadcast_to(np.reshape([band_weights], (1, -1)), (1, len(ref_bands[0])))
        # mask for keep_zeros
        if keep_zeros:
            H_r_mask = np.where(np.abs(self.H.H_r) < 1e-14, 0.0, 1.0)
            if train_S:
                S_r_mask = np.where(np.abs(self.S.H_r) < 1e-14, 0.0, 1.0)
        if batch_div == 1 and use_pinv:  # improved optimization
            if max_accel_global is None:
                max_accel_global = 1.0
            log.add_message(f"maximal acceleration {max_accel_global}")
        # memoize self.H.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.H.H_r)), dtype=np.complex128)
        for i in range(len(self.H.H_r)):
            f_i[:, i] = self.H.f_i(k_smpl, i)
        f_i[:, 0] /= 2  # divide by 2 because it is added without the symmetrization
        # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
        # modified gradient descent with conjugated derivatives
        c_i = np.conj(f_i)
        if batch_div == 1 and use_pinv:  # improved optimization
            log.add_message("preparing pseudoinverse for H")
            # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
            c_i = np.linalg.pinv(f_i.T)
            # counteract the usual treatment:
            c_i[:, 0] *= 2
            norm = len(f_i[0]) - 1 + 0.5**3
            c_i *= norm
            c_i *= len(k_smpl)
        if train_S:
            # memoize self.S.f_i here using a rectangular matrix
            s_f_i = np.zeros((len(k_smpl), len(self.S.H_r)),
                             dtype=np.complex128)
            for i in range(len(self.S.H_r)):
                s_f_i[:, i] = self.S.f_i(k_smpl, i)
            # divide by 2 because it is added without the symmetrization
            s_f_i[:, 0] /= 2
            # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
            # modified gradient descent with conjugated derivatives
            s_c_i = np.conj(s_f_i)
            if batch_div == 1 and use_pinv:  # improved optimization
                log.add_message("preparing pseudoinverse for S")
                # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
                s_c_i = np.linalg.pinv(s_f_i.T)
                # counteract the usual treatment:
                s_c_i[:, 0] *= 2
                norm = len(s_f_i[0]) - 1 + 0.5**3
                s_c_i *= norm
                s_c_i *= len(k_smpl)

        if max_accel_global is None:
            max_accel_global = len(k_smpl)
        self.normalize()

        if not train_S:
            # precompute S for every k_smpl
            S = self.S.f(k_smpl)
        # start stochastic gradient descent
        last_add = np.zeros_like(self.H.H_r)
        last_add_s = np.zeros_like(self.S.H_r)
        last_loss = float("inf")
        try:
            for iteration in range(iterations):
                batch = k_smpl[iteration % batch_div::batch_div]
                batch_ref = ref_bands[iteration % batch_div::batch_div]
                batch_f_i = f_i[iteration % batch_div::batch_div]
                batch_c_i = c_i[iteration % batch_div::batch_div]
                if train_S:
                    batch_s_f_i = s_f_i[iteration % batch_div::batch_div]
                    batch_s_c_i = s_c_i[iteration % batch_div::batch_div]
                max_accel = min(len(batch), max_accel_global)
                # compute c_i if pinv is requested, even though this is kinda slow...
                if batch_div != 1 and use_pinv:
                    batch_c_i = np.linalg.pinv(batch_f_i.T)
                    # counteract the usual treatment:
                    batch_c_i[:, 0] *= 2
                    norm = len(batch_f_i[0]) - 1 + 0.5**3
                    batch_c_i *= norm
                    batch_c_i *= len(batch)

                # regularization
                if regularization != 1.0:
                    self.H.H_r[1:] *= regularization

                # faster H and S using cached values
                H = np.einsum("nkl,in->ikl", self.H.H_r, batch_f_i)
                H += np.conj(np.swapaxes(H, -1, -2))
                if train_S:
                    H = np.einsum("nkl,in->ikl", self.S.H_r, batch_s_f_i)
                    S += np.conj(np.swapaxes(S, -1, -2))
                eigvals, eigvecs = geigh(H, S)
                eigvals = eigvals[:, band_offset:][:, :len(batch_ref[0])]
                eigvecs = eigvecs[:, :, band_offset:][:, :, :len(batch_ref[0])]
                eigvecs_c = np.conj(eigvecs)

                # the following "norm" makes sure that if only one k_smpl is used, then the convergence happens in one step.
                #norm = (np.abs(batch_f_i)**2).sum(1) # use f_i as weights, but normalize the whole step
                # faster calculation of the above due to the content of batch_f_i
                norm = np.array([len(batch_f_i[0])]) - 1 + 0.5**3
                if train_S:
                    #norm_s = (np.abs(batch_s_f_i)**2).sum(1) # use s_f_i as weights, but normalize the whole step
                    norm_s = norm
                diff = batch_ref - eigvals
                if batch_div == 1:
                    max_err = np.max(np.abs(diff), axis=0)

                # implementation of the following einsum
                #H_r_add = np.einsum("bik,bjk,bk,b,bn,k->nij", eigvecs, eigvecs_c, diff, 2 / s, batch_f_i, weights, optimize="greedy") / len(batch)
                # but faster: (I don't know why it's faster...)
                diff *= band_weights
                if batch_div == 1:
                    assert len(batch) == len(k_smpl)
                    # this is how the loss is computed in self.error
                    loss = np.linalg.norm(diff) / len(batch)**.5

                # check if the iteration is already converged
                if iteration % 100 == 0 or batch_div == 1:
                    if batch_div != 1:
                        loss, max_err = self.error(k_smpl, ref_bands, band_weights[0], band_offset)
                    if abs(last_loss / loss - 1) < convergence_threshold or loss < loss_threshold:
                        log.add_message("converged")
                        break
                    last_loss = loss
                    log.add_data(iteration, loss, max_err)

                # band_weights with new stepsize estimation, needs these squared
                diff *= band_weights
                H_diff = diff / norm[:, None]

                H_diff = eigvecs @ (H_diff[...,None] * np.swapaxes(eigvecs_c, 1, 2))
                H_r_add = np.einsum("nij,nk->kij", H_diff, batch_c_i)
                if keep_zeros:
                    H_r_add *= H_r_mask
                # preconditioning the cg step for 0 by dividing by 2
                H_r_add[0] = (H_r_add[0] + H_r_add[0].T.conj()) / 4

                # compute stepsize using my new formula based on least squares fitting
                if not train_S and use_lstsq_stepsize:
                    # this angle should always be ~90°
                    #log.add_message(f"angle {180/np.pi*np.arccos(min(1, max(-1, np.real(np.sum(last_add * H_r_add.conj())) / (np.linalg.norm(H_r_add) * np.linalg.norm(last_add)))))}")

                    H_r_add[0] *= 2
                    rr = np.linalg.norm(H_r_add)**2 # this needs to be computed without the preconditioner
                    H_r_add[0] /= 2
                    #r_AA_r = np.einsum("kn,nid,njd,d,ln,nad,nbd,lab,kij", batch_c_i, eigvecs, eigvecs_c, band_weights[0]**2, batch_f_i, eigvecs_c, eigvecs, H_r_add, H_r_add.conj(), optimize="optimal")
                    ev_r = np.einsum("nad,nbd,lab,d->nld", eigvecs_c, eigvecs, H_r_add, band_weights[0], optimize="greedy")
                    r_AA_r = np.einsum("nk,nl,nld,nkd", batch_c_i, batch_f_i, ev_r, ev_r.conj(), optimize="greedy")
                    # factor 2 because of how the symmetrisation works above!
                    alpha = norm * 0.5 * rr / np.real(r_AA_r)
                    #log.add_message(f"{alpha}, {0.5 / len(batch)}")
                else:
                    alpha = 0.5 / len(batch) / band_weights_norm

                H_r_add *= learning_rate * alpha
                if train_S:
                    S_diff = eigvecs @ (((diff / -norm_s[:,None]) * eigvals)[...,None] * np.swapaxes(eigvecs_c, 1, 2))
                    S_r_add = np.einsum("bij,bn->nij", S_diff, batch_s_c_i)
                    if keep_zeros:
                        S_r_add *= S_r_mask
                    S_r_add[0] = (S_r_add[0] + S_r_add[0].T.conj()) / 4
                    S_r_add *= learning_rate * alpha
                # impulse acceleration (factor given by the problem)
                H_r_add += last_add * (1 - 1 / max_accel)
                if train_S:
                    S_r_add += last_add_s * (1 - 1 / max_accel) # don't accelerate S???
                # change parameters
                self.H.H_r = self.H.H_r + H_r_add
                last_add = H_r_add
                if train_S:
                    self.S.H_r = self.S.H_r + S_r_add
                    last_add_s = S_r_add
            else:
                iteration = iterations
        except KeyboardInterrupt:
            log.abort()
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights[0], band_offset)
        log.add_data(iteration, l, err)
        return log

    def optimize_cg(self, k_smpl, ref_bands, band_weights, band_offset: int, iterations: int, train_S=False, keep_zeros=False, precond=True, convergence_threshold=1e-3, loss_threshold=1e-16, max_cg_iterations=5, log=True):
        N = np.shape(self.H.H_r)[1]
        N_B = len(ref_bands[0])
        assert band_offset >= 0 and band_offset <= N - N_B, f"band_offset={band_offset} must be in [0, {N-N_B}]"
        if log == False:
            # logger that doesn't print
            log = logger.OptimisationLogger(print_loss=False, verbose=False)
        elif log == True:
            # logger that prints
            log = logger.OptimisationLogger(print_loss=True, update_line=True, verbose=True)
        # reshape band_weights (no normalisation!)
        band_weights = np.broadcast_to(np.reshape([band_weights], (1, -1)), (1, N_B))
        band_weights_sqr = (band_weights * band_weights)[0]
        weights = band_weights_sqr
        # mask for keep_zeros
        if keep_zeros:
            H_r_mask = np.where(np.abs(self.H.H_r) < 1e-14, 0.0, 1.0)
            if train_S:
                S_r_mask = np.where(np.abs(self.S.H_r) < 1e-14, 0.0, 1.0)
        # memoize self.H.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.H.H_r)), dtype=np.complex128)
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.H.H_r)):
                f_i[ki, i] = self.H.f_i(k, i)
        # divide by 2 because it is added without the symmetrization
        f_i[:, 0] /= 2
        c_i = np.conj(f_i)
        fc_i = np.array([f_i, c_i])

        if train_S:
            # memoize self.S.f_i here using a rectangular matrix
            s_f_i = np.zeros((len(k_smpl), len(self.S.H_r)),
                             dtype=np.complex128)
            for ki, k in enumerate(k_smpl):
                for i in range(len(self.S.H_r)):
                    s_f_i[ki, i] = self.S.f_i(k, i)
            # divide by 2 because it is added without the symmetrization
            s_f_i[:, 0] /= 2
            s_c_i = np.conj(s_f_i)

        # unsure about these k_weights...
        #c2_i = np.linalg.pinv(f_i.T)
        #k_weights = np.real(np.einsum("nm->n", np.einsum("nk,mk->nm", c2_i, c2_i.conj()))[:,None])
        weights = weights[None,:]# * k_weights
        
        if precond:
            # preconditioned CG (here) is:
            # E^+ A^+ A E y = E^+ A^+ b, x = E y, E invertible
            # where all preconditioning matrices E are expressed as
            # functions, which take ownership of their argument and mutate it!
            # E = E^+
            # TODO add k_weights in here as well!
            # TODO it seems that preconditioning like this does not always result in an improvement... why?
            #E_mat = np.linalg.pinv(np.linalg.cholesky(np.einsum("nk,nl->kl", c_i, f_i))) * len(k_smpl)**.5
            E_mat = np.linalg.cholesky(np.linalg.pinv(np.einsum("nk,nl->kl", c_i, f_i))) * len(k_smpl)**.5
            E_mat_c = np.conj(E_mat)
            # precompute a contraction that comes up often
            c_i_E_mat_c = np.einsum("nk,kl->nl", c_i, E_mat_c)
            #print(E_mat)

            def precond(x):
                np.einsum("lk,kij->lij", E_mat, x, out=x)
                return x
        else:
            E_mat = np.eye(len(f_i[0]))
            E_mat_c = np.eye(len(f_i[0]))
            c_i_E_mat_c = c_i

            def precond(x):
                return x

        self.normalize()

        if not train_S:
            # precompute S for every k_smpl
            S = self.S.f(k_smpl)

        # loss = self.loss(k_smpl, ref_bands, band_, band_offset)
        loss = float("inf")
        try:
            mat_t_path = combined_path = None
            for iteration in range(iterations):
                # faster H and S using cached values
                H = np.einsum("nkl,in->ikl", self.H.H_r, f_i)
                H += np.conj(np.swapaxes(H, -1, -2))
                if train_S:
                    S = np.einsum("nkl,in->ikl", self.S.H_r, s_f_i)
                    S += np.conj(np.swapaxes(S, -1, -2))
                eigvals, eigvecs = geigh(H, S)
                eigvals = eigvals[:, band_offset:][:, :N_B]
                eigvecs = eigvecs[:, :, band_offset:][:, :, :N_B]
                eigvecs_c = np.conj(eigvecs)

                diff = eigvals - ref_bands
                max_err = np.max(np.abs(diff), axis=0)

                # this is how the loss is computed in self.error
                err = np.linalg.norm(diff * band_weights)
                new_loss = err / len(k_smpl)**.5
                if abs(loss / new_loss - 1) < convergence_threshold or new_loss < loss_threshold:
                    log.add_message("converged")
                    break
                loss = new_loss

                if mat_t_path is None:
                    #mat_t_path, info = np.einsum_path("nk,nid,njd,nd->kij", c_i_E_mat_c, eigvecs, eigvecs_c, diff, optimize="optimal")
                    #print(mat_t_path)
                    #print(info)
                    # HACK: sometimes einsum_path completely fails! This is a path that works well for my most common case:
                    mat_t_path = ['einsum_path', (1, 3), (1, 2), (0, 1)]
                diff *= weights
                # TODO test contracting eigvecs and eigvecs_c beforehand, because that combination is used everywhere
                b = np.einsum("nk,nid,njd,nd->kij", c_i_E_mat_c, eigvecs, eigvecs_c, diff, optimize=mat_t_path)
                if combined_path is None:
                    #mat_path = np.einsum_path("nid,njd,nij->nd", eigvecs_c, eigvecs, np.einsum("nk,kij->nij", f_i, b), optimize="optimal")[0]
                    #combined_path = np.einsum_path("nk,nid,njd,nad,nbd,nab,kl->lij", c_i, eigvecs, eigvecs_c, eigvecs_c, eigvecs, np.einsum("nk,kij->nij", f_i, b), E_mat_c, optimize="optimal")[0]
                    #combined_path, info = np.einsum_path("nk,nid,njd,nd,nad,nbd,onp,opab->kij", c_i_E_mat_c, eigvecs, eigvecs_c, weights, eigvecs_c, eigvecs, [f_i, f_i], [b, b], optimize="optimal")
                    #print(combined_path)
                    #print(info)
                    # HACK: sometimes einsum_path completely fails! This is a path that works well for my most common case:
                    combined_path = ['einsum_path', (6, 7), (4, 6), (4, 5), (3, 4), (1, 3), (1, 2), (0, 1)]
                def A(x):
                    x = precond(x / len(k_smpl))
                    #fx = np.einsum("nk,kij->nij", f_i, x)
                    #fx += np.swapaxes(fx, -1, -2).conj()
                    #return np.einsum("nk,nid,njd,nad,nbd,nab,kl->lij", c_i, eigvecs, eigvecs_c, eigvecs_c, eigvecs, fx, E_mat_c, optimize=combined_path)
                    return np.einsum("nk,nid,njd,nd,nad,nbd,onp,opab->kij", c_i_E_mat_c, eigvecs, eigvecs_c, weights, eigvecs_c, eigvecs, fc_i, [x, np.swapaxes(x, -1, -2).conj()], optimize=combined_path)
                # A(x) is close to a projection matrix
                # tested a different start value to move through saddle points in the optimisation, but it didn't work...
                x0 = None# (np.random.standard_normal(b.shape) + np.random.standard_normal(b.shape)*1j) * np.linalg.norm(b) * 1e-1
                step = precond(conjugate_gradient_solve(A, b, x0=x0, err=np.linalg.norm(b) * 1e-3, max_i=max_cg_iterations))
                step *= 1 / len(k_smpl)

                self.H.H_r -= step
                if keep_zeros:
                    self.H.H_r *= H_r_mask
                # keep the 0 entry hermitian!
                self.H.H_r[0] += np.conj(self.H.H_r[0].T)
                self.H.H_r[0] /= 2
                log.add_data(iteration, loss, max_err)
            else:
                iteration = iterations
        except KeyboardInterrupt:
            log.abort()
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights[0], band_offset)
        log.add_data(iteration, l, err)
        return log

    def normalize(self):
        """
        Apply as many transformations as possible to normalize the parameters.
        Sadly this is not enough to guarantee that two models with the same coefficient
        functions (neighbors) and bandstructure will have the same hamiltonian.
        """
        self.H.H_r[0] = (self.H.H_r[0] + self.H.H_r[0].T.conj()) / 2
        self.S.H_r[0] = (self.S.H_r[0] + self.S.H_r[0].T.conj()) / 2
        # first normalize S such that S(0)=1
        # NOTE: I think using cholestky instead of matrix sqrt can break symmetries...
        # L = np.conj(np.linalg.inv(np.linalg.cholesky(self.S.f(((0,)*self.dim(),))[0])).T)
        la, ev = np.linalg.eigh(self.S.f(((0,)*self.dim(),))[0])
        L = ev @ np.diag(la**-.5) @ np.conj(ev.T)
        self.H.H_r = np.einsum("ji,njk,kl->nil", np.conj(L), self.H.H_r, L)
        self.S.H_r = np.einsum("ji,njk,kl->nil", np.conj(L), self.S.H_r, L)
        # normalize H
        _, ev = np.linalg.eigh(self.H.f(((0,)*self.dim(),))) # this keeps the symmetry intact
        ev = ev[0]
        # stable sort ev such that the 0 structure of H_r[0] is kept (important for symmetry)
        sorting = np.argsort(np.argmin(np.abs(ev) < 1e-7, axis=0))
        ev = ev.T[sorting].T
        self.H.H_r = np.einsum("ji,njk,kl->nil", np.conj(ev), self.H.H_r, ev)
        self.S.H_r = np.einsum("ji,njk,kl->nil", np.conj(ev), self.S.H_r, ev)
        # TODO test the following further normalisation
        if False:
            # normalize a little more using complex reflections on the second matrix
            if len(self.H.H_r) > 1:
                for i in range(1, len(self.H.H_r[1])):
                    x = self.H.H_r[1][i-1, i]
                    a = np.abs(x)
                    if a != 0:
                        sign = np.conj(x) / a
                        self.H.H_r[1:, :, i] *= sign
                        self.H.H_r[1:, i, :] *= np.conj(sign)
                        self.S.H_r[1:, :, i] *= sign
                        self.S.H_r[1:, i, :] *= np.conj(sign)
                    else:
                        pass  # TODO switch to a different cell to normalize
        # normalize continuous DoF (TODO)

    def permute(self, order):
        """apply a permutation to the basis of the hamiltonian"""
        for i in range(len(self.H.H_r)):
            self.H.H_r[i] = self.H.H_r[i][order]
            self.H.H_r[i] = self.H.H_r[i][:, order]
        for i in range(len(self.S.H_r)):
            self.S.H_r[i] = self.S.H_r[i][order]
            self.S.H_r[i] = self.S.H_r[i][:, order]

    def transform(self, A):
        """Apply a transformation on the neighbors in the fourier series.
        This is useful to go from a crystal space fit to the reciprocal space fit.
        For that, put the lattice matrix A with the lattice vectors in the argument.

        Args:
            A (arraylike(dim, dim)): Transformation matrix for the neighbors. E.g. matrix with the real lattice vectors in the columns.
        """
        self.H.neighbors = np.einsum("ji,ni->nj", A, self.H.neighbors)
        self.S.neighbors = np.einsum("ji,ni->nj", A, self.S.neighbors)

    def params_complex(self):
        """
        Get the complex fourier coefficients $H_r$.

        Returns:
            ndarray: The coefficients of the HermitianFourierSeries `self.H`
        """
        return self.H.H_r

    def set_from_complex(self, H_r):
        """Set `self.H` from the values of the complex fourier coefficients $H_r$.

        Args:
            H_r (arraylike): the complex matrices, that appear in the fourier series, matching the neighbor list of this model.
        """
        self.H.H_r = np.asarray(H_r).astype(np.complex128)

    def bands(self, k_smpl):
        """Compute the bandstructure for a given set of k samples.

        Args:
            k_smpl (arraylike): k-space samples in shape (N_k, dim)

        Returns:
            arraylike: the bandstructure in shape (N_k, N_b)
        """
        return geigvalsh(self.H.f(k_smpl), self.S.f(k_smpl))

    def __call__(self, k_smpl):
        return self.bands(k_smpl)

    def bands_grad(self, k_smpl):
        """Computes the gradients of the bands (group velocities).
        Because that requires computing the bandstructure, the bandstructure is also returned.

        Args:
            k_smpl (arraylike): k-space samples in shape (N_k, dim)

        Returns:
            (arraylike(N_k, N_b), arraylike(N_k, dim, N_b)): (bands, grads)
        """
        return tuple(geigh_grad(self.H.f(k_smpl), self.S.f(k_smpl), self.H.df(k_smpl), self.S.df(k_smpl))[:2])

    def bands_grad_hess(self, k_smpl):
        """Computes the hessians of the bands (effective inverse masses).
        Because that requires computing the bandstructure and its gradients, they are also returned.

        Args:
            k_smpl (arraylike): k-space samples in shape (N_k, dim)

        Returns:
            (arraylike(N_k, N_b), arraylike(N_k, dim, N_b), arraylike(N_k, dim, dim, N_b)): (bands, grads, hessians)
        """
        # TODO add dS and ddS!!
        bands, ev = geigh(self.H.f(k_smpl), self.S.f(k_smpl))
        df = self.H.df(k_smpl)
        ddf = self.H.ddf(k_smpl)
        # first order perturbation theory terms
        df_ev = np.einsum("mji, mnjk, mkl -> mnil", np.conj(ev), df, ev)
        grads = np.real(np.diagonal(df_ev, axis1=2, axis2=3))
        hess1 = np.real(np.einsum("mji, mpqjk, mki -> mpqi", np.conj(ev), ddf, ev))
        # second order perturbation theory terms
        no_diag = np.array(df_ev)  # copy before modification (grads is a view)
        for i in range(len(grads[0, 0])):
            no_diag[:, :, i, i] = 0  # zero out diagonal terms
        # dev = ev[:,None,:,:] @ (no_diag / (bands[:,None,:,None] - bands[:,None,None,:] + 1e-40))
        # hess2 = np.real(np.einsum("mji, mpjk, mqki -> mpqi", 2*np.conj(ev), df, dev))
        db = no_diag / (bands[:, None, :, None] -
                        bands[:, None, None, :] + 1e-40)
        hess2 = np.real(np.einsum("mpik, mqki -> mpqi", df_ev, db))
        return bands, grads, hess1 - 2*hess2

    def supercell(self, A_original, A_new) -> 'AsymTightBindingModel':
        """
        Generate a tight binding model (with self.neighbors set) for a supercell defined as A' = A Λ,
        where Λ is a non singular integer valued matrix.

        Args:
            A_original (arraylike(dim, dim)): The basis vectors of the real lattice **before** the supercell transformation.
            A_new (arraylike(dim, dim)): The basis vectors of the real lattice **after** the supercell transformation.
            cos_reduced (bool, optional): Same as in `init_tight_binding`. Defaults to False.
            exp (bool, optional): Same as in `init_tight_binding`. Defaults to True.

        Returns:
            Self: BandStructureModel that describes the same solid, but using a bigger cell.
        """
        A_original = np.asarray(A_original)
        A_new = np.asarray(A_new)
        dim = len(self.H.neighbors[0])
        assert dim == len(A_original) and dim == len(A_new), "A matrix doesn't match the dimension of the model"
        matrix = np.linalg.inv(A_original) @ A_new
        assert np.all(np.abs(np.round(matrix) - matrix) < 1e-7), "The supercell matrix must be integer valued"
        matrix = np.round(matrix).astype(np.int64)
        det = round(np.linalg.det(matrix))
        new_neighbors = self.H.neighbors @ matrix.T
        n = len(self.H.H_r[0])
        new_band_count = n * det
        H_r = self.params_complex()
        # now get all integer positions in the cell defined by matrix
        # for that, compute the (half open) bounding box of matrix * [0,1[^3
        box = np.stack(np.meshgrid(*[[0, 1]]*dim), axis=-1).reshape(-1, dim)
        box = box @ matrix.T
        bounding_box = np.min(box, axis=0), np.max(box, axis=0)
        assert np.array(bounding_box).dtype == np.int64
        # now create a meshgrid inside the bounding box and select the points with inv(matrix) in [0, 1[
        box = np.stack(np.meshgrid(*[np.arange(bounding_box[0][d], bounding_box[1][d]) for d in range(dim)]), axis=-1).reshape(-1, dim)
        p_box = box @ np.linalg.inv(matrix).T
        # internal positions + origin (0)
        internal_positions = list(p_box[np.all((p_box >= 0-1e-7) & (p_box < 1-1e-7), axis=1)] @ A_new.T)
        assert len(internal_positions) == det
        # now build the new hamiltonian
        H_r2 = np.zeros((len(H_r), new_band_count, new_band_count), dtype=np.complex128)
        neighbor_func = try_neighbor_function(self.H.neighbors)
        for k, nk in enumerate(new_neighbors):
            for i, pi in enumerate(internal_positions):
                for j, pj in enumerate(internal_positions):
                    m, mirror = neighbor_func(nk + pj - pi)
                    if m is not None:
                        H_r2[k, i*n:(i+1)*n, j*n:(j+1)*n] = H_r[m] if not mirror else np.conj(H_r[m].T)
        model = AsymTightBindingModel(HermitianFourierSeries(new_neighbors, H_r2))
        return model

    def __add__(self, other) -> 'AsymTightBindingModel':
        if type(other) == type(self):
            # direct sum of the models -> combine the bandstructures
            return AsymTightBindingModel(self.H + other.H, S=self.S + other.S)
        else:
            # shift the whole bandstructure
            # TODO figure out how this actually works wih S(k)
            H = self.H.copy()
            H.H_r[0] += self.S.H_r[0] * float(other)
            return AsymTightBindingModel(H, S=self.S.copy())

    def __mul__(self, fac: float) -> 'AsymTightBindingModel':
        return AsymTightBindingModel(HermitianFourierSeries(self.H.neighbors, self.H.H_r * fac), S=self.S.copy())


# free electron model (scaled to have hessian 1I) for testing
# TODO add basis to allow for fcc and bcc for testing.
def free_electron_model_orthogonal(a: float, b: float, c: float, neighbor_count: int, bands_per_direction: int) -> tuple[AsymTightBindingModel, float]:
    """Recreate a free electron model using a tight-binding model.
    The model represents H(k) = k^2, so it is completely unitless.
    The second return value is the energy unit in eV, assuming a, b, c are in Ångström.

    Args:
        a (float): lattice constant a (x direction)
        b (float): lattice constant b (y direction)
        c (float): lattice constant c (z direction)
        neighbor_count (int): number of neighbor terms used in each direction
        bands_per_direction (int): the final band count will be `bands_per_direction**3`

    Returns:
        (AsymTightBindingModel, float): a tight-binding model describing the free electrons, energy unit in eV (multiply by this to get the model in eV units)
    """
    neighbors = [(0,0,0)] + [v for n in range(1,(neighbor_count+1)*bands_per_direction) for v in [(a*n,0,0),(0,b*n,0),(0,0,c*n)]]
    H_r = np.zeros((len(neighbors), 1, 1))
    for i, n in enumerate(neighbors):
        if n[0] != 0 and n[1] == 0 and n[2] == 0:
            H_r[i] += 2 * (-1)**round(n[0]/a) / n[0]**2
        if n[0] == 0 and n[1] != 0 and n[2] == 0:
            H_r[i] += 2 * (-1)**round(n[1]/b) / n[1]**2
        if n[0] == 0 and n[1] == 0 and n[2] != 0:
            H_r[i] += 2 * (-1)**round(n[2]/c) / n[2]**2
    H_r[0] = np.pi**2 * (1/a**2 + 1/b**2 + 1/c**2)/3
    model = AsymTightBindingModel.new(neighbors, 1)
    model.set_from_complex(H_r)
    model *= 1/(2*np.pi)**2

    model2 = model.supercell(np.diag([a, b, c]), np.diag([a, b, c])*bands_per_direction)
    model2.H.neighbors /= bands_per_direction
    model2 *= bands_per_direction**2
    model2.H.limit_neighbor_count(neighbor_count * 3 + 1)

    eV_unit = 3.80998211 * (2*np.pi)**2 # hbar^2/m_e/2 / (1Å)^2 / 1eV * (2pi)^2
    return model2, eV_unit


def autofit_asym(name, neighbors_src, k_smpl, ref_bands, band_weights, sym: Symmetry, start_neighbors_count=2, add_bands_below=0, add_bands_above=0, randomize=True, restarts=4):
    """
    This function creates a fit for a given dataset and
    saves it at various checkpoints as json file `asym_{name}_{neighbors_count}.json`.

    see `/examples/fit_copper.py` for an example how to use this function.
    """
    assert add_bands_below >= 0 and add_bands_above >= 0 and start_neighbors_count >= 0
    start = time.time()

    # the fitting requires (k_smpl_sym, ref_bands, band_weights, neighbors)
    # the best fitting protocol requires sorting k by distance to 0.
    reorder = np.argsort(np.linalg.norm(k_smpl, axis=-1))
    k_smpl = k_smpl[reorder]
    ref_bands = ref_bands[reorder]

    neighbors_count = start_neighbors_count + 1
    neighbors = sym.complete_neighbors(neighbors_src[:neighbors_count])

    best_tb_error = float("inf")
    best_tb = None # type: AsymTightBindingModel
    for _ in range(restarts):
        print("restart with new model")
        tb = AsymTightBindingModel.new(neighbors, add_bands_below + np.shape(ref_bands)[-1] + add_bands_above)
        tb.randomize(0.01)
        tb.normalize()
        tb.H.H_r[0] = np.diag(ref_bands[0])
        # now fit in 10 steps extending from the gamma point (0-point)
        log = logger.OptimisationLogger(update_line=False)
        for j in range(1, 11):
            n = (j * len(k_smpl) + 9) // 10
            tb.optimize_cg(k_smpl[:n], ref_bands[:n], band_weights, add_bands_below, 1, max_cg_iterations=10, log=log)
        if log.last_loss() < best_tb_error:
            best_tb = tb
            best_tb_error = log.last_loss()
    assert best_tb is not None
    tb = best_tb
    # save checkpoint
    tb.save(f"asym_{name}_start.json")
    ellapsed = time.time() - start
    start = time.time()
    print(f"saved asym_{name}_start.json (After {ellapsed:.3f}s)")

    # now fit with increasing amount of neighbors
    log = logger.OptimisationLogger(update_line=False)
    l, err = tb.error(k_smpl, ref_bands, band_weights, add_bands_below)
    log.add_data(0, l, err)
    first = True
    while neighbors_count < len(neighbors_src):
        if not first:
            new_neighbors = sym.complete_neighbors([neighbors_src[neighbors_count]])
            tb.H.add_neighbors(new_neighbors)
            neighbors_count += 1
        first = False
        print("fit with neighbors", neighbors_src[:neighbors_count])
        for _ in range(10 if randomize else 1):
            # small randomisation seems to help convergence...?!?
            tb.randomize(log.last_loss() * (0.1 if randomize else 0.01))
            tb.normalize()
            tb.optimize_cg(k_smpl, ref_bands, band_weights, add_bands_below, 100, convergence_threshold=1e-3, max_cg_iterations=5, log=log)
            # save checkpoint
            tb.save(f"asym_{name}_{neighbors_count}.json")
            print(f"saved asym_{name}_{neighbors_count}.json")
        ellapsed = time.time() - start
        start = time.time()
        print(f"fit for this neighbor set completed in {ellapsed:.3f}s")
    return tb

