import numpy as np
import scipy
from typing import Self
from .symmetry import *
from . import logger
from . import json_tb_format
from . import wannier90_tb_format as tb_fmt

def random_hermitian(n):
    h = (np.random.random((n, n)) * 2 - 1) + 1j * (2 * np.random.random((n, n)) - 1)
    return h + np.conj(h.T)

def geigh(H: np.ndarray, S: np.ndarray):
    if len(H.shape) == 3:
        if np.linalg.norm(S - np.eye(S.shape[-1])) < 1e-8:
            # fast path
            return np.linalg.eigh(H)
        else:
            res_la = np.zeros(H.shape[:2], dtype=H.dtype)
            res_ev = np.zeros(H.shape, dtype=H.dtype)
            for i in range(len(H)):
                la, ev = scipy.linalg.eigh(H[i], S[i])
                res_la.append(la)
                res_ev.append(ev)
            return np.array(res_la), np.array(res_ev)
    else:
        return scipy.linalg.eigh(H, S)

def geigvalsh(H, S):
    if len(np.shape(H)) == 3:
        res_la = []
        for i in range(len(H)):
            la = scipy.linalg.eigvalsh(H[i], S[i])
            res_la.append(la)
        return np.array(res_la)
    else:
        return scipy.linalg.eigvalsh(H, S)

# A is a linear symmetric python function
# b is of the same dimension as A(x0)
# apart from solving a normal linear eq, it can also be used
# to calculate the pseudo inverse of A efficiently like this:
# A_pinv = conjugate_gradient_solve(lambda x: A @ x, np.identity(4), np.diag(1/(np.diag(A) + 1e-12)))
def conjugate_gradient_solve(A, b, err=1e-9, max_i=None):
    x = np.zeros_like(b)
    r = -b
    d = r.copy()
    r_sqr = np.sum(r.conj() * r)
    if max_i == None:
        max_i = int(np.prod(np.shape(x)) * 2.5) + 1
    i = 0
    while True:
        A_d = A(d)
        d_sqr = np.real(np.sum(d.conj() * A_d)) # real because A is positive definite!
        if d_sqr == 0:
            #print("cg", i, "d")
            return x
        alpha = r_sqr / d_sqr
        x -= alpha * d
        if r_sqr <= err**2:
            #print("cg", i)
            return x
        if i > max_i:
            #print(f"conjugate gradient didn't converge ({r_sqr**.5:.2e} > {err:.2e})")
            return x
        
        i += 1
        A_d *= alpha
        r -= A_d
        r_sqr_last = r_sqr
        r_sqr = np.sum(r.conj() * r)
        d *= r_sqr / r_sqr_last
        d += r

class HermitianFourierSeries:
    """
    This class represents a fourier series with a hermitian output for each input position k.

    For this it needs the neighbors `r`, such that `H_k = sum_r H_r exp(2πi k·r)`.
    To make sure that the result is hermitian, it is processed as `H'_k = H_k + H_k^+ + H_0`
    where `H_0` is referring to the matrix at `r=0`, which is skipped in the sum above.
    """

    def __init__(self, neighbors, H_r):
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        assert len(neighbors) == len(H_r)
        assert np.shape(H_r)[1] == np.shape(H_r)[2]
        assert len(np.shape(neighbors)) == 2
        assert len(np.shape(H_r)) == 3
        self.neighbors = np.asarray(neighbors)
        self.H_r = np.asarray(H_r)

    def dim(self):
        """
        Returns:
            int: Dimension of the neighbor positions. Usually 1, 2 or 3.
        """
        return len(self.neighbors[0])

    def f_i(self, k, i):
        """Exponential function that appears as coefficient in the fourier series."""
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        return np.exp(2j*np.pi*(k @ r))

    def df_i(self, k, i):
        """Derivative of f_i"""
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        return 2j*np.pi * np.exp(2j*np.pi*(k @ r))[...,None] * np.asarray(r)
    
    def ddf_i(self, k, i):
        """Second derivative of f_i"""
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        r_sqr = np.asarray(r)[:,None] * np.asarray(r)[None,:]
        return (2j*np.pi)**2 * np.exp(2j*np.pi*(k @ r))[...,None,None] * r_sqr

    def f(self, k):
        """Compute the hermitian (N, N) matrix"""
        mat = np.zeros(np.shape(k)[:-1] + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = (1,)*len(np.shape(k)[:-1]) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += np.asarray(self.f_i(k, i))[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        # expect H_r[0] to always be hermitian! (this is very important for symmetrization)
        mat += np.asarray(self.f_i(k, 0))[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat
    
    def df(self, k):
        """Compute the derivative of f wrt k in the direction dk, outputshape (dim(k), N, N)"""
        mat = np.zeros(np.shape(k) + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = (1,)*len(np.shape(k)) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += np.asarray(self.df_i(k, i))[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.df_i(k, 0))[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat
    
    def ddf(self, k):
        """Compute the second derivative of f wrt k in the direction dk, outputshape (dim(k), dim(k), N, N)"""
        mat = np.zeros(np.shape(k) + (np.shape(k)[-1],) + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = tuple([1]*(1+len(np.shape(k)))) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += np.asarray(self.ddf_i(k, i))[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.ddf_i(k, 0))[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat
    
    def copy(self):
        return HermitianFourierSeries(self.neighbors.copy(), self.H_r.copy())
    
    def unit_matrix(neighbors, n) -> Self:
        """Initialize the fourier series, such that it equals the identity for all k.

        Args:
            neighbors (arraylike): Positions for the fourier transform. (see class documentation)
            n (int): Matrix dimension.

        Returns:
            Self: identity fourier series
        """
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        S_r = np.zeros((len(neighbors), n, n))
        S_r[0] = np.eye(n)
        return HermitianFourierSeries(neighbors, S_r)

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
    
    def new(neighbors, band_count) -> Self:
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
        scale = (np.max(model_bands) - np.min(model_bands)) * 0.001 # TODO make this per band as otherwise it will become really large for far apart bands
        H_r = [np.diag(model_bands)] + [random_hermitian(len(model_bands)) * scale for _ in range(param_count-1)]
        if use_S:
            tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r), HermitianFourierSeries.unit_matrix(neighbors, len(H_r[0])))
        else:
            tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        tb.normalize()
        return tb

    def save(self, filename, format=None):
        if format is None:
            if filename.endswith(".json"):
                format = "json"
            else:
                raise ValueError("unrecognised format for file " + filename)
        if format not in {"json"}:
            raise ValueError('only supported format is "json"')
        opt = np.get_printoptions()
        np.set_printoptions(precision=16, suppress=False, threshold=100000, legacy='1.25')
        if format == "json":
            json_tb_format.save(filename, self.H.neighbors, self.H.H_r)
            # TODO save the S matrix part as well, maybe in a separate file? Or extend the format...
    
        np.set_printoptions(**opt) # reset printoptions

    def load(filename, format=None) -> Self:
        if format is None:
            if filename.endswith(".repr"):
                format = "python"
            elif filename.endswith(".json"):
                format = "json"
            elif filename.endswith(".dat"):
                format = "wannier90"
            else:
                raise ValueError("unrecognised format for file " + filename)
        if format not in {"python", "json", "wannier90"}:
            raise ValueError('supported formats are "python", "json" and "wannier90", but was ' + str(format))
        if format == "python":
            with open(filename, "r") as file:
                H_r_repr = " ".join(file.readlines())
                H_r, neighbors, S, inversion = eval(H_r_repr.replace("array", "np.array"))
                # convert to this asymmetric format!
                sym = Symmetry(S, inversion=inversion)
                if len(sym) > 1:
                    # multiply the parameters with correct weights
                    weights = Symmetry(S, inversion=False).r_class_size(neighbors)
                    H_r[1:] /= weights[1:,None,None]
                    # complete neighbors and H_r using sym
                    complete_neighbors, order = sym.complete_neighbors(neighbors, return_order=True)
                    model = AsymTightBindingModel(HermitianFourierSeries(complete_neighbors, H_r[order]))
                    raise ValueError("input with coefficient symmetry is currently not supported as there is some inconsistency somewhere")
                else:
                    model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        elif format == "wannier90":
            neighbors, H_r, w_r_H_r = tb_fmt.load(filename)
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
        bands = self.bands(k_smpl)[:,band_offset:][:,:len(ref_bands[0])]
        err = bands - ref_bands
        max_err = np.max(np.abs(err), axis=0)
        err *= np.reshape(band_weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5, max_err
    
    def loss(self, k_smpl, ref_bands, band_weights, band_offset):
        """
        Returns:
            float: the weighted loss (standard deviation)
        """
        bands = self.bands(k_smpl)[:,band_offset:][:,:len(ref_bands[0])]
        err = (bands - ref_bands) * np.reshape(band_weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5
    
    def print_error(self, k_smpl, ref_bands, band_weights, band_offset, prefix="", log=None):
        """Print the loss and the maximal error per band"""
        band_weights = np.broadcast_to(np.reshape([band_weights / np.mean(band_weights)], (1, -1)), (1, len(ref_bands[0])))
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
        # TODO normalisation changes the loss!
        band_weights_norm = np.max(band_weights)**2
        band_weights = np.broadcast_to(np.reshape([band_weights], (1, -1)), (1, len(ref_bands[0])))
        # mask for keep_zeros
        if keep_zeros:
            H_r_mask = np.where(np.abs(self.H.H_r) < 1e-14, 0.0, 1.0)
            if train_S:
                S_r_mask = np.where(np.abs(self.S.H_r) < 1e-14, 0.0, 1.0)
        if batch_div == 1 and use_pinv: # improved optimization
            if max_accel_global is None:
                max_accel_global = 1.0
            log.add_message(f"maximal acceleration {max_accel_global}")
        # memoize self.H.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.H.H_r)), dtype=np.complex128)
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.H.H_r)):
                f_i[ki, i] = self.H.f_i(k, i)
        f_i[:, 0] /= 2 # divide by 2 because it is added without the symmetrization
        # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
        c_i = np.conj(f_i) # modified gradient descent with conjugated derivatives
        if batch_div == 1 and use_pinv: # improved optimization
            log.add_message("preparing pseudoinverse for H")
            # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
            c_i = np.linalg.pinv(f_i.T)
            # counteract the usual treatment:
            c_i[:,0] *= 2
            norm = len(f_i[0]) - 1 + 0.5**3
            c_i *= norm
            c_i *= len(k_smpl)
        if train_S:
            # memoize self.S.f_i here using a rectangular matrix
            s_f_i = np.zeros((len(k_smpl), len(self.S.H_r)), dtype=np.complex128)
            for ki, k in enumerate(k_smpl):
                for i in range(len(self.S.H_r)):
                    s_f_i[ki, i] = self.S.f_i(k, i)
            s_f_i[:, 0] /= 2 # divide by 2 because it is added without the symmetrization
            # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
            s_c_i = np.conj(s_f_i) # modified gradient descent with conjugated derivatives
            if batch_div == 1 and use_pinv: # improved optimization
                log.add_message("preparing pseudoinverse for S")
                # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
                s_c_i = np.linalg.pinv(s_f_i.T)
                # counteract the usual treatment:
                s_c_i[:,0] *= 2
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
                    batch_c_i[:,0] *= 2
                    norm = len(batch_f_i[0]) - 1 + 0.5**3
                    batch_c_i *= norm
                    batch_c_i *= len(batch)

                # regularization
                if regularization != 1.0:
                    self.H.H_r[1:] *= regularization

                # faster H and S using cached values
                H = (self.H.H_r[None,...] * batch_f_i[...,None,None]).sum(1)
                H += np.conj(np.swapaxes(H, -1, -2))
                if train_S:
                    S = (self.S.H_r[None,...] * batch_s_f_i[...,None,None]).sum(1)
                    S += np.conj(np.swapaxes(S, -1, -2))
                eigvals, eigvecs = geigh(H, S)
                eigvals = eigvals[:,band_offset:][:,:len(batch_ref[0])]
                eigvecs = eigvecs[:,:,band_offset:][:,:,:len(batch_ref[0])]
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
                    loss = np.linalg.norm(diff) / len(batch)**.5 # this is how the loss is computed in self.error

                # check if the iteration is already converged
                if iteration % 100 == 0 or batch_div == 1:
                    if batch_div != 1:
                        loss, max_err = self.error(k_smpl, ref_bands, band_weights, band_offset)
                    if abs(last_loss / loss - 1) < convergence_threshold or loss < loss_threshold:
                        log.add_message("converged")
                        break
                    last_loss = loss
                    log.add_data(iteration, loss, max_err)
                
                # band_weights with new stepsize estimation, needs these squared
                diff *= band_weights
                H_diff = diff / norm[:,None]

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
            log.add_message("aborted")
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights, band_offset)
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
        f_i[:, 0] /= 2 # divide by 2 because it is added without the symmetrization
        c_i = np.conj(f_i)
        
        if train_S:
            # memoize self.S.f_i here using a rectangular matrix
            s_f_i = np.zeros((len(k_smpl), len(self.S.H_r)),
                             dtype=np.complex128)
            for ki, k in enumerate(k_smpl):
                for i in range(len(self.S.H_r)):
                    s_f_i[ki, i] = self.S.f_i(k, i)
            s_f_i[:, 0] /= 2 # divide by 2 because it is added without the symmetrization
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
                # TODO preconditioning based on band_weights!
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
            mat_t_path = mat_path = None
            for iteration in range(iterations):
                # faster H and S using cached values
                H = (self.H.H_r[None,...] * f_i[...,None,None]).sum(1)
                H += np.conj(np.swapaxes(H, -1, -2))
                if train_S:
                    S = (self.S.H_r[None,...] * s_f_i[...,None,None]).sum(1)
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
                if mat_path is None:
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
                    return np.einsum("nk,nid,njd,nd,nad,nbd,onp,opab->kij", c_i_E_mat_c, eigvecs, eigvecs_c, weights, eigvecs_c, eigvecs, [f_i, f_i.conj()], [x, np.swapaxes(x, -1, -2).conj()], optimize=combined_path)
                # A(x) is close to a projection matrix
                step = precond(conjugate_gradient_solve(A, b, err=np.linalg.norm(b) * 1e-3, max_i=max_cg_iterations))
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
            log.add_message("aborted")
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights, band_offset)
        log.add_data(iteration, l, err)
        return log

    def normalize(self):
        """
        Apply as many transformations as possible to normalize the parameters.
        Sadly this is not enough to guarantee that two models with the same coefficient
        functions (neighbors) and bandstructure will have the same hamiltonian.

        NOTE: CURRENTLY NOT IMPLEMENTED!
        """
        self.H.H_r[0] = (self.H.H_r[0] + self.H.H_r[0].T.conj()) / 2
        self.S.H_r[0] = (self.S.H_r[0] + self.S.H_r[0].T.conj()) / 2
        if True:
            # TODO add cholesky decomposition to normalize S to 1 as well
            return
        # normalize
        _, ev = np.linalg.eigh(self.H.f(((0,)*self.dim(),))) # this keeps the symmetry intact
        ev = ev[0]
        #_, ev = np.linalg.eigh(self.H_r[0])
        # stable sort ev such that the 0 structure of H_r[0] is kept (important for symmetry)
        sorting = np.argsort(np.argmin(np.abs(ev) < 1e-7, axis=0))
        ev = ev.T[sorting].T
        for i in range(len(self.H.H_r)):
            self.H.H_r[i] = np.conj(ev.T) @ self.H.H_r[i] @ ev
        # TODO test the following
        # normalize a little more using complex reflections on the second matrix
        if len(self.H.H_r) > 1:
            for i in range(1, len(self.H.H_r[1])):
                x = self.H.H_r[1][i-1, i]
                a = np.abs(x)
                if a != 0:
                    sign = np.conj(x) / a
                    self.H.H_r[:, :, i] *= sign
                    self.H.H_r[:, i, :] *= np.conj(sign)
                else:
                    pass # TODO switch to a different cell to normalize
        # normalize continuous DoF (TODO)

    def permute(self, order):
        """apply a permutation to the basis of the hamiltonian"""
        for i in range(len(self.H.H_r)):
            self.H.H_r[i] = self.H.H_r[i][order]
            self.H.H_r[i] = self.H.H_r[i][:,order]
        for i in range(len(self.S.S_r)):
            self.S.S_r[i] = self.S.S_r[i][order]
            self.S.S_r[i] = self.S.S_r[i][:,order]
    
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
        bands, ev = geigh(self.H.f(k_smpl), self.S.f(k_smpl))
        dH = self.H.df(k_smpl)
        dS = self.S.df(k_smpl)
        grads = np.real(np.einsum("mji, mnjk, mki -> mni", np.conj(ev), dH, ev))
        # TODO check!
        grads += np.real(np.einsum("mji, mnjk, mki -> mni", np.conj(ev), dS, ev)) * -bands[:,:,None]
        return bands, grads
    
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
        no_diag = np.array(df_ev) # copy before modification (grads is a view)
        for i in range(len(grads[0,0])):
            no_diag[:,:,i,i] = 0 # zero out diagonal terms
        #dev = ev[:,None,:,:] @ (no_diag / (bands[:,None,:,None] - bands[:,None,None,:] + 1e-40))
        #hess2 = np.real(np.einsum("mji, mpjk, mqki -> mpqi", 2*np.conj(ev), df, dev))
        db = no_diag / (bands[:,None,:,None] - bands[:,None,None,:] + 1e-40)
        hess2 = np.real(np.einsum("mpik, mqki -> mpqi", df_ev, db))
        return bands, grads, hess1 - 2*hess2
    
    def supercell(self, A_original, A_new):
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
        matrix = np.round(matrix)
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
        H_r = np.zeros((len(H_r), new_band_count, new_band_count), dtype=np.complex128)
        neighbor_func = try_neighbor_function(self.H.neighbors)
        for k, nk in enumerate(new_neighbors):
            for i, pi in enumerate(internal_positions):
                for j, pj in enumerate(internal_positions):
                    m, mirror = neighbor_func(nk + pj - pi)
                    if m is not None:
                        H_r[k, i*n:(i+1)*n, j*n:(j+1)*n] = H_r[m] if not mirror else np.conj(H_r[m].T)
        model = AsymTightBindingModel(HermitianFourierSeries(new_neighbors, H_r))
        return model
