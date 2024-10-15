import numpy as np
from matplotlib import pyplot as plt
from typing import Self
from .symmetry import *
from .linalg import *
from . import logger
from . import json_tb_format
from . import wannier90_tb_format as tb_fmt
import sys


def plot_bands_generic(k_smpl, bands, *args, **kwargs):
    """Simplest possible bandstructure plotting function.
    This is just here for reference and to understand the basic principle.

    Args:
        k_smpl (arraylike): k-space samples in shape (N_k, dim)
        bands (arraylike): bandstructure in shape (N_k, N_b)
    """
    plt.gca().set_prop_cycle(None)
    for i in range(len(bands[0])):
        if len(np.ravel(k_smpl)) == len(k_smpl):
            plt.plot(k_smpl, [bands[j][i]
                     for j in range(len(k_smpl))], *args, **kwargs)
        else:
            plt.plot([bands[j][i]
                     for j in range(len(k_smpl))], *args, **kwargs)


class BandStructureModel:
    """
    This class is a highly customisable bandstructure model.  
    The usual initialisation without fit data is
    ```
    neighbors = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    band_count = 5
    tb = BandStructureModel.init_tight_binding(Symmetry.none(), neighbors, band_count)
    ```
    If fit data is available, then the initialisation should look like this
    ```
    k_smpl, ref_bands, band_offset = ...
    neighbors = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    add_bands = 0
    tb = BandStructureModel.init_tight_binding_from_ref(Symmetry.none(), neighbors, k_smpl, ref_bands, band_offset, add_bands)
    ```

    The flexibility of the class comes in form of custom weight functions (see `__init__` for an example).
    A couple presets can be used with init_tight_binding, namely (cos_reduced, exp),
    which effect how the data is stored in self.params.

    For the default tight_binding functions, there is 
    """

    def __init__(self, f_i, df_i, params, ddf_i=None):
        """Initialize the class from coefficient functions and params.  

        The hamiltonian is computed using `H_k = sum_i(params[i] * f_i(k, i))`.

        Args:
            f_i (function(k: arraylike(N_k, dim), i: int) -> arraylike(N_k,)): coefficient function
            df_i (function(k: arraylike(N_k, dim), i: int) -> arraylike(N_k, dim)): gradient of the coefficient function with respect to k
            params (arraylike(N, N_b, N_b)): parameters that get multiplied and summed with the coefficients.
            ddf_i (function(k: arraylike(N_k, dim), i: int) -> arraylike(N_k, dim, dim), optional): hessian of the coefficient function with respect to k. Defaults to None.
        """
        self.f_i = f_i
        self.df_i = df_i
        self.ddf_i = ddf_i
        self.symmetrizer = None
        self.params = np.asarray(params)
        self.sym = None
        self.cos_reduced = None
        self.exp = None
        self.neighbors = None
        pshape = np.shape(self.params)
        assert len(pshape) == 3
        assert pshape[1] == pshape[2]
        assert pshape[0] > 0

    def copy(self) -> Self:
        model = BandStructureModel(self.f_i, self.df_i, self.params.copy())
        model.symmetrizer = self.symmetrizer
        model.sym = self.sym
        model.neighbors = self.neighbors
        model.cos_reduced = self.cos_reduced
        model.exp = self.exp
        return model

    def band_count(self):
        """
        Returns:
            int: Number of bands of the model. Often referred to as N_b
        """
        return len(self.params[0])

    def init_tight_binding(symmetry: Symmetry, neighbors, band_count, cos_reduced=False, exp=False) -> Self:
        neighbors = np.array(neighbors, dtype=np.float64)
        # symmetry.check_neighbors(neighbors)
        f_i_tb, df_i_tb, ddf_i_tb, term_count, neighbors = _get_tight_binding_coeff_funcs(
            symmetry, neighbors, cos_reduced=cos_reduced, exp=exp)
        model = BandStructureModel(f_i_tb, df_i_tb, np.zeros(
            (term_count, band_count, band_count), dtype=complex), ddf_i=ddf_i_tb)
        model.sym = symmetry
        model.neighbors = neighbors
        model.cos_reduced = cos_reduced
        model.exp = exp
        return model

    def init_tight_binding_from_ref(symmetry: Symmetry, neighbors, k_smpl, ref_bands, band_offset=0, add_bands=0, cos_reduced=False, exp=False) -> Self:
        neighbors = np.array(neighbors, dtype=np.float64)
        f_i_tb, df_i_tb, ddf_i_tb, term_count, neighbors = _get_tight_binding_coeff_funcs(
            symmetry, neighbors, cos_reduced=cos_reduced, exp=exp)
        model = BandStructureModel.init_from_ref(
            f_i_tb, df_i_tb, ddf_i_tb, term_count, k_smpl, ref_bands, band_offset, add_bands)
        # TODO handle the case where cos_reduced=False, because there the params need to be processed here
        model.sym = symmetry
        model.neighbors = neighbors
        model.cos_reduced = cos_reduced
        model.exp = exp
        return model

    def init_from_ref(f_i, df_i, ddf_i, param_count, k_smpl, ref_bands, band_offset=0, add_bands=0) -> Self:
        # assuming the first matrix is a constant term
        k0_index = np.argmin(np.linalg.norm(k_smpl, axis=-1))
        assert np.linalg.norm(k_smpl[k0_index]) == 0
        k0_bands = ref_bands[k0_index]
        left_pad = band_offset
        right_pad = add_bands - band_offset
        k0_bands = np.concatenate(
            [[k0_bands[0]]*left_pad, k0_bands, [k0_bands[-1]]*right_pad])
        scale = 0.001
        params = [np.diag(k0_bands)] + [random_hermitian(len(k0_bands))
                                        * scale for _ in range(param_count-1)]
        return BandStructureModel(f_i, df_i, params, ddf_i)

    def randomize(self, sigma, keep_zeros=False):
        """
        Randomize parameters with normal distributed numbers with a standard deviation sigma.

        **This can break symmetry in bad ways -> call `self.normalize()` manually after this!**
        """
        dparams = sigma * np.random.standard_normal(
            self.params.shape) + sigma * 1j*np.random.standard_normal(self.params.shape)
        if keep_zeros:
            dparams *= np.where(np.abs(self.params) < 1e-8, 0, 1)
        self.params += dparams

    def f(self, k):
        """Compute the hamiltonian

        Returns:
            ndarray: The hamiltonian for each k, shape: (N_k, N_b, N_b)
        """
        mat = np.zeros(np.shape(k)[:-1] +
                       self.params[0].shape, dtype=np.complex128)
        params_shape = (1,)*len(np.shape(k)[:-1]) + self.params[0].shape
        for i in range(1, len(self.params)):
            mat += np.asarray(self.f_i(k, i))[..., None,
                                              None] * self.params[i].reshape(params_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        # expect params[0] to always be hermitian! (this is very important for symmetrization)
        mat += np.asarray(self.f_i(k, 0))[..., None,
                                          None] * self.params[0].reshape(params_shape)
        return mat

    def df(self, k):
        """Compute the gradient of the hamiltonian with respect to k

        Returns:
            ndarray: The hamiltonian gradient for each k, shape: (N_k, dim, N_b, N_b)
        """
        mat = np.zeros(np.shape(k) + self.params[0].shape, dtype=np.complex128)
        params_shape = (1,)*len(np.shape(k)) + self.params[0].shape
        for i in range(1, len(self.params)):
            mat += np.asarray(self.df_i(k, i))[..., None,
                                               None] * self.params[i].reshape(params_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.df_i(k, 0))[..., None,
                                           None] * self.params[0].reshape(params_shape)
        return mat

    def ddf(self, k):
        """Compute the hessian of the hamiltonian with respect to k

        Returns:
            ndarray: The hamiltonian gradient for each k, shape: (N_k, dim, dim, N_b, N_b)
        """
        mat = np.zeros(np.shape(k) + (np.shape(k)
                       [-1],) + self.params[0].shape, dtype=np.complex128)
        params_shape = (1,) + (1,)*len(np.shape(k)) + self.params[0].shape
        for i in range(1, len(self.params)):
            mat += np.asarray(self.ddf_i(k, i)
                              )[..., None, None] * self.params[i].reshape(params_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.ddf_i(k, 0)
                          )[..., None, None] * self.params[0].reshape(params_shape)
        return mat

    def error(self, k_smpl, ref_bands, band_weights, band_offset):
        """
        Returns:
            (float, ndarray(N_b)): the weighted loss (standard deviation) and the maximal error per band
        """
        assert len(k_smpl) == len(ref_bands)
        assert len(band_weights) == len(ref_bands[0])
        bands = self.bands(k_smpl)
        err = (bands[:, band_offset:][:, :len(ref_bands[0])] - ref_bands)
        max_err = np.max(np.abs(err), axis=0)
        err *= np.reshape(band_weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5, max_err

    def loss(self, k_smpl, ref_bands, band_weights, band_offset):
        """
        Returns:
            float: the weighted loss (standard deviation)
        """
        assert len(k_smpl) == len(ref_bands)
        assert len(band_weights) == len(ref_bands[0])
        bands = self.bands(k_smpl)
        err = (bands[:, band_offset:][:, :len(ref_bands[0])] -
               ref_bands) * np.reshape(band_weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5
        # return np.max(np.abs(bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands))

    def print_error(self, k_smpl, ref_bands, band_weights, band_offset, prefix="", log=None):
        """Print the loss and the maximal error per band"""
        band_weights = np.broadcast_to(np.ravel(band_weights / np.mean(band_weights)), (len(ref_bands[0]),))
        l, err = self.error(k_smpl, ref_bands, band_weights, band_offset)
        if log is not None:
            log.add_message(f"{prefix}loss: {l:.2e} (max band-error {err})")
        else:
            print(f"{prefix}loss: {l:.2e} (max band-error {err})")

    def print_model_norms(self):
        """Print the norm of the matrices in params (along with their corresponding neighbor if available)"""
        if self.neighbors is not None:
            for n, p in zip(self.neighbors, self.params):
                print(f"{tuple(n)}: {np.linalg.norm(p)}")
        else:
            for i, p in enumerate(self.params):
                print(f"{i:2}: {np.linalg.norm(p)}")

    def optimize(self, k_smpl, k_smpl_weights, ref_bands, band_weights, band_offset: int, iterations: int, batch_div=1, train_k0=True, regularization=1.0, learning_rate=1.0, log=True, max_accel_global=None, use_pinv=True, keep_zeros=False, convergence_threshold=1e-3, loss_threshold=1e-16):
        """
        This function optimizes the parameters of the model to fit the given data.
        This works for any set of custom functions.  
        The algorithm used is a modified version of gradient descent.  
        If a symmetrizer is specified in the model, it will be used on each step.  
        The final result is normalized.

        Args:
            k_smpl (arraylike): k-space samples with shape (N_k, dim) that go into `self.f(k)`
            k_smpl_weights (arraylike | float): Weights for each k_smpl. Usually 1.0.
            ref_bands (arraylike): Bandstructure to fit the model to with shape (N_k, N_b)
            band_weights (arraylike | float): Weights for each band index. Usually 1.0.
            band_offset (int): If the model has more bands than the reference, this specifies the number of bands, that are below the reference bands. Usually 0.
            iterations (int): Number of optimizer iterations.
            batch_div (int, optional): Divide k_smpl into "batch_div" batches. Defaults to 1.
            train_k0 (bool, optional): Allow/Disallow the value of `self.f(0)` to change. Defaults to True.
            regularization (float, optional): Scaling of params in each step to keep params small. Defaults to 1.0.
            learning_rate (float, optional): Scaling of stepsize. Defaults to 1.0.
            log (bool | OptimisationLogger, optional): The logger to report the fit progress to. This can be set to True or False to initialize a default logger with printing enabled/disabled. Defaults to True.
            max_accel_global (float, optional): How much the stepsize can grow if the stepdirection doesn't change. Defaults to `len(k_smpl)` or 1.0 if use_pinv=True.
            use_pinv (bool, optional): Use a pseudoinverse to compute better coefficient for each step. Defaults to True.
            keep_zeros (bool, optional): If True, any zeros in `self.params` will be kept. This is useful for some simple weak Symmetrisation. Defaults to False.
            convergence_threshold (float, optional): If two consecutive loss evaluations have a relative difference smaller than this, stop. Defaults to 1e-3.
            loss_threshold (float, optional): If the loss is below this threshold, stop. Defaults to 1e-16.

        Returns:
            OptimisationLogger: the logger with the log data. This can be used to plot the loss over iterations.
        """
        N = np.shape(self.params)[1]
        assert band_offset >= 0 and band_offset <= N - len(ref_bands[0])
        if log == False:
            # logger that doesn't print
            log = logger.OptimisationLogger(print_loss=False, verbose=False)
        elif log == True:
            # logger that prints
            log = logger.OptimisationLogger(
                print_loss=True, update_line=True, verbose=True)
        # recast k_smpl and ref_bands to make sure they are arrays
        k_smpl = np.asarray(k_smpl)
        ref_bands = np.asarray(ref_bands)
        # reshape k_smpl_weights
        k_smpl_weights = np.broadcast_to(np.reshape(
            [k_smpl_weights], (-1, 1)), (len(k_smpl), 1))
        # reshape band_weights
        band_weights = np.broadcast_to(np.reshape(
            [band_weights], (1, -1)), (1, len(ref_bands[0])))
        # normalize band_weights for the stepsize
        # NOTE: This is completely wrong for use_pinv=True. Use bandweights with care...
        # also it's wrong for use_pinv=False, as technically no band_weights_normalized should be larger than 1.
        band_weights_normalized = band_weights / np.mean(band_weights)
        # memoize self.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.params)), dtype=np.complex128)
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.params)):
                f_i[ki, i] = self.f_i(k, i)
        f_i[:, 0] /= 2  # divide by 2 because it is added without the symmetrization
        # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
        c_i = np.conj(f_i)  # classic gradient descent
        if batch_div == 1 and use_pinv:  # improved optimization
            log.add_message("preparing pseudoinverse")
            # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
            c_i = np.linalg.pinv(f_i.T)
            # print(c_i)
            # print(c_i @ f_i.T) # weird matrix
            # print(f_i.T @ c_i) # identity
            if max_accel_global is None:
                max_accel_global = 1.0
            # normalize (does also work without this very slow step)
            # if verbose:
            #    print("normalizing pseudoinverse")
            # for i in range(len(self.params)):
            #    div = max(np.abs((c_i * f_i[i]).sum()), 1.0)
            #    c_i /= div
            #    max_accel_global *= div
            # for i in range(len(k_smpl)):
            #    c_i[i,:] /= max(np.abs((c_i[i] * f_i).sum()), 1.0)
            log.add_message(f"maximal acceleration {max_accel_global}")
            # counteract the usual treatment:
            c_i *= (np.abs(f_i)**2).sum(1, keepdims=True)
            c_i *= len(k_smpl)
        if max_accel_global is None:
            max_accel_global = len(k_smpl)
        # start stochastic gradient descent
        last_add = np.zeros_like(self.params)
        self.normalize()
        last_loss = float("inf")
        try:
            for iteration in range(iterations):
                batch = k_smpl[iteration % batch_div::batch_div]
                batch_weights = k_smpl_weights[iteration %
                                               batch_div::batch_div]
                batch_ref = ref_bands[iteration % batch_div::batch_div]
                batch_f_i = f_i[iteration % batch_div::batch_div]
                batch_c_i = c_i[iteration % batch_div::batch_div]
                batch_weights = batch_weights / np.mean(batch_weights)
                max_accel = min(len(batch), max_accel_global)

                # regularization
                if regularization != 1.0:
                    self.params[1:] *= regularization

                # faster f using cached values
                f = (self.params[None, ...] *
                     batch_f_i[..., None, None]).sum(1)
                f += np.conj(np.swapaxes(f, -1, -2))
                eigvals, eigvecs = np.linalg.eigh(f)
                eigvals = eigvals[:, band_offset:][:, :len(batch_ref[0])]
                eigvecs = eigvecs[:, :, band_offset:][:, :, :len(batch_ref[0])]

                # use f_i as weights, but normalize the whole step
                s = (np.abs(batch_f_i)**2).sum(1)
                diff = batch_ref - eigvals
                if batch_div == 1:
                    max_err = np.max(np.abs(diff), axis=0)
                    # this is how the loss is computed in self.error
                    loss = np.linalg.norm(diff * band_weights) / len(batch)**.5

                # check if the iteration can end here
                if iteration % 100 == 0 or batch_div == 1:
                    if batch_div != 1:
                        loss, max_err = self.error(
                            k_smpl, ref_bands, band_weights[0], band_offset)
                    if abs(last_loss / loss - 1) < convergence_threshold or loss < loss_threshold:
                        log.add_message("converged")
                        break
                    last_loss = loss
                    log.add_data(iteration, loss, max_err)

                # implementation of the following einsum
                # params_add = np.einsum("bik, bjk, bk, b, bn, k -> nij", eigvecs, np.conj(eigvecs), diff, 2 / s, batch_f_i, band_weights_normalized, optimize="optimal") / len(batch)
                # but faster: (I don't know why it's faster...)
                diff *= band_weights_normalized

                diff *= batch_weights
                diff *= np.reshape((0.5 / len(batch)) / s, (-1, 1))
                diff = eigvecs @ (diff[..., None] *
                                  np.conj(np.swapaxes(eigvecs, 1, 2)))
                params_add = np.einsum("bij,bn->nij", diff, batch_c_i)
                if self.symmetrizer is not None:
                    params_add = self.symmetrizer(params_add)
                if not train_k0:
                    # TODO this only works with cos_reduced...
                    params_add[0] *= 0.0
                if keep_zeros:
                    params_add *= np.where(np.abs(self.params)
                                           < 1e-14, 0.0, 1.0)
                params_add *= learning_rate
                # impulse acceleration (beta given by the problem)
                params_add += last_add * (1 - 1 / max_accel)
                # change parameters (alpha = 1.0 with the chosen way to normalize the gradient)
                self.params = self.params + params_add
                last_add = params_add
            else:
                iteration = iterations
        except KeyboardInterrupt:
            log.add_message("aborted")
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights[0], band_offset)
        log.add_data(iteration, l, err)
        return log

    def optimize_cg(self, k_smpl, k_smpl_weights, ref_bands, band_weights, band_offset: int, iterations: int, log=True, precond=True, keep_zeros=False, convergence_threshold=1e-3, loss_threshold=1e-16, max_cg_iterations=5):
        """
        This function optimizes the parameters of the model to fit the given data.
        This works for any set of custom functions.  
        The algorithm used is based on least squares fit on each linearized step.
        If a symmetrizer is specified in the model, it will be used on each step.  
        The final result is normalized.

        Args:
            k_smpl (arraylike): k-space samples with shape (N_k, dim) that go into `self.f(k)`
            k_smpl_weights (arraylike | float): Weights for each k_smpl. Usually 1.0.
            ref_bands (arraylike): Bandstructure to fit the model to with shape (N_k, N_b)
            band_weights (arraylike | float): Weights for each band index. Usually 1.0.
            band_offset (int): If the model has more bands than the reference, this specifies the number of bands, that are below the reference bands. Usually 0.
            iterations (int): Number of optimizer iterations.
            log (bool | OptimisationLogger, optional): The logger to report the fit progress to. This can be set to True or False to initialize a default logger with printing enabled/disabled. Defaults to True.
            precond (bool, optional): Use a preconditioner for the cg linear equation solver. Defaults to True.
            keep_zeros (bool, optional): If True, any zeros in `self.params` will be kept. This is useful for some simple weak Symmetrisation. Defaults to False.
            convergence_threshold (float, optional): If two consecutive loss evaluations have a relative difference smaller than this, stop. Defaults to 1e-3.
            loss_threshold (float, optional): If the loss is below this threshold, stop. Defaults to 1e-16.
            max_cg_iterations (int, optional): Maximal number of steps for the conjugate gradient in each iteration. Defaults to 5.

        Returns:
            OptimisationLogger: the logger with the log data. This can be used to plot the loss over iterations.
        """
        N = np.shape(self.params)[1]
        N_B = len(ref_bands[0])
        assert band_offset >= 0 and band_offset <= N - \
            N_B, f"band_offset={band_offset} must be in [0, {N-N_B}]"
        if log == False:
            # logger that doesn't print
            log = logger.OptimisationLogger(print_loss=False, verbose=False)
        elif log == True:
            # logger that prints
            log = logger.OptimisationLogger(
                print_loss=True, update_line=True, verbose=True)
        # reshape band_weights (no normalisation!)
        band_weights = np.broadcast_to(
            np.reshape([band_weights], (1, -1)), (1, N_B))
        band_weights_sqr = (band_weights * band_weights)[0]
        weights = band_weights_sqr
        # mask for keep_zeros
        if keep_zeros:
            H_r_mask = np.where(np.abs(self.params) < 1e-14, 0.0, 1.0)
        # memoize self.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.params)), dtype=np.complex128)
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.params)):
                f_i[ki, i] = self.f_i(k, i)
        f_i[:, 0] /= 2  # divide by 2 because it is added without the symmetrization
        c_i = np.conj(f_i)

        # unsure about these k_weights...
        # c2_i = np.linalg.pinv(f_i.T)
        # k_weights = np.real(np.einsum("nm->n", np.einsum("nk,mk->nm", c2_i, c2_i.conj()))[:,None])
        weights = weights[None, :]  # * k_weights

        if precond:
            # preconditioned CG (here) is:
            # E^+ A^+ A E y = E^+ A^+ b, x = E y, E invertible
            # where all preconditioning matrices E are expressed as
            # functions, which take ownership of their argument and mutate it!
            # E = E^+
            # TODO add k_weights in here as well!
            # TODO it seems that preconditioning like this does not always result in an improvement... why?
            # E_mat = np.linalg.pinv(np.linalg.cholesky(np.einsum("nk,nl->kl", c_i, f_i))) * len(k_smpl)**.5
            E_mat = np.linalg.cholesky(np.linalg.pinv(
                np.einsum("nk,nl->kl", c_i, f_i))) * len(k_smpl)**.5
            E_mat_c = np.conj(E_mat)
            # precompute a contraction that comes up often
            c_i_E_mat_c = np.einsum("nk,kl->nl", c_i, E_mat_c)

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
        loss = float("inf")
        try:
            mat_t_path = mat_path = None
            for iteration in range(iterations):
                # faster H and S using cached values
                H = (self.params[None, ...] * f_i[..., None, None]).sum(1)
                H += np.conj(np.swapaxes(H, -1, -2))
                eigvals, eigvecs = np.linalg.eigh(H)
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
                    # mat_t_path, info = np.einsum_path("nk,nid,njd,nd->kij", c_i_E_mat_c, eigvecs, eigvecs_c, diff, optimize="optimal")
                    # print(mat_t_path, info)
                    # HACK: sometimes einsum_path completely fails! This is a path that works well for my most common case:
                    mat_t_path = ['einsum_path', (1, 3), (1, 2), (0, 1)]
                diff *= weights
                # TODO test contracting eigvecs and eigvecs_c beforehand, because that combination is used everywhere
                b = np.einsum("nk,nid,njd,nd->kij", c_i_E_mat_c,
                              eigvecs, eigvecs_c, diff, optimize=mat_t_path)
                if mat_path is None:
                    # combined_path, info = np.einsum_path("nk,nid,njd,nd,nad,nbd,onp,opab->kij", c_i_E_mat_c, eigvecs, eigvecs_c, weights, eigvecs_c, eigvecs, [f_i, f_i], [b, b], optimize="optimal")
                    # print(combined_path, info)
                    # HACK: sometimes einsum_path completely fails! This is a path that works well for my most common case:
                    combined_path = [
                        'einsum_path', (6, 7), (4, 6), (4, 5), (3, 4), (1, 3), (1, 2), (0, 1)]

                def A(x):
                    x = precond(x / len(k_smpl))
                    return np.einsum("nk,nid,njd,nd,nad,nbd,onp,opab->kij", c_i_E_mat_c, eigvecs, eigvecs_c, weights, eigvecs_c, eigvecs, [f_i, f_i.conj()], [x, np.swapaxes(x, -1, -2).conj()], optimize=combined_path)
                # A(x) is close to a projection matrix
                step = precond(conjugate_gradient_solve(
                    A, b, err=np.linalg.norm(b) * 1e-3, max_i=max_cg_iterations))
                step *= 1 / len(k_smpl)

                self.params -= step
                if self.symmetrizer is not None:
                    self.params = self.symmetrizer(self.params)
                if keep_zeros:
                    self.params *= H_r_mask
                # keep the 0 entry hermitian!
                self.params[0] += np.conj(self.params[0].T)
                self.params[0] /= 2
                log.add_data(iteration, loss, max_err)
            else:
                iteration = iterations
        except KeyboardInterrupt:
            log.add_message("aborted")
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights[0], band_offset)
        log.add_data(iteration, l, err)
        return log

    def normalize(self):
        """
        Apply as many transformations as possible to normalize the parameters.
        Sadly this is not enough to guarantee that two models with the same coefficient functions (neighbors) and bandstructure will have the same hamiltonian.
        If a symmetrizer is specified in the model, then only the symmetrizer will be executed, to make sure that nothing breaks the symmetry.
        """
        if self.symmetrizer is not None:
            self.params = self.symmetrizer(self.params)
            # don't use the normalization if a symmetrizer is specified, as it might mess with that.
            return
        if self.exp is None or not self.exp or self.sym.inversion:
            # here the matrices are assumed to be hermitian, so fix them just in case!
            # in case of inversion symmetry, the exponential form is more free than the cos/sin form.
            # -> Reduce that freedom by making the matrices hermitian!
            self.params[1:] += np.conj(np.swapaxes(self.params[1:], -1, -2))
            self.params[1:] /= 2
        # the zero matrix is always hermitian
        self.params[0] += np.conj(self.params[0].T)
        self.params[0] /= 2
        la, ev = np.linalg.eigh(self.params[0])
        # la, ev = np.linalg.eigh(self.f(0))
        for i in range(len(self.params)):
            self.params[i] = np.conj(ev.T) @ self.params[i] @ ev
        # normalize a little more using complex reflections on the second matrix
        if len(self.params) > 1:
            for i in range(1, len(self.params[1])):
                x = self.params[1][i-1, i]
                a = np.abs(x)
                if a > 1e-14:
                    sign = np.conj(x) / a
                    self.params[:, :, i] *= sign
                    self.params[:, i, :] *= np.conj(sign)
                else:
                    # set very small values to 0, because they are likely the result of small symmetry breaks
                    self.params[1][i-1, i] = 0.0

    def permute(self, order):
        """Apply a permutation to the basis of the hamiltonian"""
        for i in range(len(self.params)):
            self.params[i] = self.params[i][order]
            self.params[i] = self.params[i][:, order]

    def params_complex(self):
        """
        Get the complex fourier coefficients $H_r$ for the initially specified neighbors.
        This function specifically works for models created with `init_tight_binding_from_ref`
        and raises an exception otherwise.
        This function also corrects `self.params` for `cos_reduced` and `exp` to get the actual $H_r$.

        Returns:
            ndarray: The complex matrices matching the neighbors
        """
        if self.neighbors is None:
            raise ValueError(
                "This function can only be used on default tight binding models")
        if self.exp:
            H_r = np.array(self.params)
            if self.cos_reduced:
                H_r[0] -= np.sum(H_r[1:] +
                                 np.conj(np.swapaxes(H_r[1:], -1, -2)), axis=0)
            return H_r
        # TODO this probably doesn't need to be divided by 2 anymore...
        # TODO add a test for this function by comparing different models!
        H_r = np.array(self.params)
        if not self.sym.inversion:
            n = (len(self.params)+1)//2
            if self.cos_reduced:
                # H_r[1:n] += H_r[0]
                H_r[0] -= np.sum(H_r[1:n] +
                                 np.conj(np.swapaxes(H_r[1:n], -1, -2)), axis=0)
                pass
            H_r[1:n] += H_r[n:] * -1j
            H_r = H_r[:n]
        elif self.cos_reduced:
            H_r[0] -= np.sum(H_r[1:] +
                             np.conj(np.swapaxes(H_r[1:], -1, -2)), axis=0)
        return H_r

    def set_params_complex(self, H_r):
        """Set `self.params` from the values of the complex fourier coefficients $H_r$.
        This function specifically works for models created with `init_tight_binding_from_ref`
        and raises an exception otherwise.

        Args:
            H_r (arraylike): the complex matrices, that appear in the fourier series, matching the neighbor list of this model.
        """
        if self.neighbors is None:
            raise ValueError(
                "This function can only be used on default tight binding models")
        if len(H_r) != len(self.neighbors):
            raise ValueError(f"The parameters {np.shape(H_r)} must match the neighbor count {len(self.neighbors)}")
        if self.exp:
            self.params = np.array(H_r)
            if self.cos_reduced:
                self.params[0] += np.sum(self.params[1:] +
                                         np.conj(np.swapaxes(self.params[1:], -1, -2)), axis=0)
        else:
            self.params = np.array(H_r)
            n = len(H_r)
            if not self.sym.inversion:
                A = self.params + np.conj(np.swapaxes(self.params, -1, -2))
                B = 1j*(self.params - np.conj(np.swapaxes(self.params, -1, -2)))
                self.params = np.concatenate([A, B[1:]], axis=0)
            else:
                self.params += np.conj(np.swapaxes(self.params, -1, -2))
            self.params /= 2
            if self.cos_reduced:
                self.params[0] += np.sum(self.params[1:n], axis=0)*2

    def save(self, filename, format=None):
        """Save the model to a file format, which can be read with the correspoding `BandStructureModel.load` function.
        This function specifically works for models created with `init_tight_binding_from_ref`
        and raises an exception otherwise.  

        The most flexible file format is "python", which can save params, neighbors and symmetry.  
        The most useful format is "json", as it is safe and easy to read in any language.

        Args:
            filename (str): Name of the destination file.
            format (str, optional): Supported formats are "python", "json" and "wannier90". Defaults to (format for filename extension).
        """
        if self.neighbors is None:
            raise ValueError(
                "This function can only be used on default tight binding models")
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
            raise ValueError(
                'supported formats are "python", "json" and "wannier90"')
        opt = np.get_printoptions()
        np.set_printoptions(precision=16, suppress=False,
                            threshold=100000, legacy='1.25')
        if format == "python":
            with open(filename, "w") as file:
                file.write(repr(self.params_complex()) + ",\\\n")
                file.write(repr(self.neighbors) + ",\\\n")
                file.write(repr(self.sym.S) + ",\\\n")
                file.write(repr(self.sym.inversion) + "\n")
        elif format == "wannier90":
            tb_fmt.save_hr(filename, self.neighbors, self.params_complex())
        elif format == "json":
            json_tb_format.save(filename, self.neighbors,
                                self.params_complex())

        # reset printoptions to what they were before
        np.set_printoptions(**opt)

    def load(filename, format=None, cos_reduced=False, exp=True) -> Self:
        """Import a tight binding model into the given `self.params` specification (cos_reduced, exp).

        Warning: the "python" format executes the file as python code. Don't open unchecked files with it.

        Args:
            filename (str): Name of the loaded file.
            format (str, optional): Supported formats are "python", "json", "wannier90tb" and "wannier90hr". Defaults to (format for filename extension).
            cos_reduced (bool, optional): Same as in `init_tight_binding`. Defaults to False.
            exp (bool, optional): Same as in `init_tight_binding`. Defaults to True.

        Returns:
            Self: BandStructureModel represented by the file.
        """
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
            raise ValueError(
                'supported formats are "python", "json" and "wannier90", but was ' + str(format))
        if format == "python":
            with open(filename, "r") as file:
                H_r_repr = " ".join(file.readlines())
                H_r, neighbors, S, inversion = eval(
                    H_r_repr.replace("array", "np.array"))
                # TODO for some reason this only works with cos_reduced=True, exp=False
                model = BandStructureModel.init_tight_binding(Symmetry(
                    S, inversion=inversion), neighbors, len(H_r[0]), cos_reduced=True, exp=False)
                model.set_params_complex(H_r)
        elif format == "wannier90hr":
            neighbors, H_r = tb_fmt.load_hr(filename)
            model = BandStructureModel.init_tight_binding(
                Symmetry.none(), neighbors, len(H_r[0]), cos_reduced=cos_reduced, exp=exp)
            model.set_params_complex(H_r)
        elif format == "wannier90tb":
            neighbors, H_r, w_r_params, degeneracies, A = tb_fmt.load_tb(
                filename)
            model = BandStructureModel.init_tight_binding(
                Symmetry.none(), neighbors, len(H_r[0]), cos_reduced=cos_reduced, exp=exp)
            model.set_params_complex(H_r)
        elif format == "json":
            neighbors, H_r = json_tb_format.load(filename)
            model = BandStructureModel.init_tight_binding(
                Symmetry.none(), neighbors, len(H_r[0]), cos_reduced=cos_reduced, exp=exp)
            model.set_params_complex(H_r)
        return model

    def bands(self, k_smpl):
        """Compute the bandstructure for a given set of k samples.

        Args:
            k_smpl (arraylike): k-space samples in shape (N_k, dim)

        Returns:
            arraylike: the bandstructure in shape (N_k, N_b)
        """
        return np.linalg.eigvalsh(self.f(k_smpl))

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
        bands, ev = np.linalg.eigh(self.f(k_smpl))
        df = self.df(k_smpl)
        # all of the following 3 are about the same speed for some reason
        # grads = np.real(np.einsum("mij, mnjk, mki -> mni", np.conj(np.swapaxes(ev, 1, 2)), df, ev))
        grads = np.real(
            np.einsum("mji, mnjk, mki -> mni", np.conj(ev), df, ev))
        # grads = np.real(np.diagonal(np.conj(np.swapaxes(ev, 1, 2))[:,None,:,:] @ df @ ev[:,None,:,:], axis1=2, axis2=3))
        return bands, grads

    def bands_grad_hess(self, k_smpl):
        """Computes the hessians of the bands (effective inverse masses).
        Because that requires computing the bandstructure and its gradients, they are also returned.

        Args:
            k_smpl (arraylike): k-space samples in shape (N_k, dim)

        Returns:
            (arraylike(N_k, N_b), arraylike(N_k, dim, N_b), arraylike(N_k, dim, dim, N_b)): (bands, grads, hessians)
        """
        bands, ev = np.linalg.eigh(self.f(k_smpl))
        df = self.df(k_smpl)
        ddf = self.ddf(k_smpl)
        # first order perturbation theory terms
        df_ev = np.einsum("mji, mnjk, mkl -> mnil", np.conj(ev), df, ev)
        grads = np.real(np.diagonal(df_ev, axis1=2, axis2=3))
        hess1 = np.real(
            np.einsum("mji, mpqjk, mki -> mpqi", np.conj(ev), ddf, ev))
        # second order perturbation theory terms
        # TODO degenerate perturbation by computing np.linalg.eigh(self.df(k_smpl)) in each degenerate subspace
        no_diag = np.array(df_ev)  # copy before modification (grads is a view)
        for i in range(len(grads[0, 0])):
            no_diag[:, :, i, i] = 0  # zero out diagonal terms
        # dev = ev[:,None,:,:] @ (no_diag / (bands[:,None,:,None] - bands[:,None,None,:] + 1e-40))
        # hess2 = np.real(np.einsum("mji, mpjk, mqki -> mpqi", 2*np.conj(ev), df, dev))
        db = no_diag / (bands[:, None, :, None] -
                        bands[:, None, None, :] + 1e-40)
        hess2 = np.real(np.einsum("mpik, mqki -> mpqi", df_ev, db))
        return bands, grads, hess1 - 2*hess2

    # compute an approximate electron phonon coupling https://doi.org/10.1103/PhysRevB.19.6130
    # g_nm(k, k') is computed for k = k_smpl[k1_indices] and k' = k_smpl[k2_indices]
    # shape(g_nm) = (len(k1_indices), len(k2_indices), len(k), bands, bands)
    # returns bands, grads, g_nm(k1, k2)
    def electron_phonon_coupling(self, k_smpl, k1_indices, k2_indices):
        bands, ev = np.linalg.eigh(self.f(k_smpl))
        df = self.df(k_smpl)
        df_ev = np.einsum("mji, mnjk, mkl -> mnil", np.conj(ev), df, ev)
        v = np.real(np.diagonal(df_ev, axis1=2, axis2=3))
        ev_c = np.conj(ev)
        g = np.einsum("ani, aki, bkj -> abnij", v[k1_indices], ev_c[k1_indices], ev[k2_indices])\
            - np.einsum("aki, bkj, bnj -> abnij",
                        ev_c[k1_indices], ev[k2_indices], v[k2_indices])
        return bands, v, g

    def transform(self, A):
        """Apply a transformation on the neighbors in the fourier series.
        This is useful to go from a crystal space fit to the reciprocal space fit.
        For that, put the lattice matrix A with the lattice vectors in the argument.

        Args:
            A (arraylike(dim, dim)): Transformation matrix for the neighbors. E.g. matrix with the real lattice vectors in the columns.
        """
        if self.neighbors is None:
            raise ValueError(
                "This function can only be used on default tight binding models")
        # replace the content of neighbors, but not the array itself, as it is linked in the coefficient functions!
        self.neighbors[:] = np.einsum("ji,ni->nj", A, self.neighbors)

    def supercell(self, A_original, A_new, cos_reduced=False, exp=True) -> Self:
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
        if self.neighbors is None:
            raise ValueError(
                "This function can only be used on default tight binding models")
        A_original = np.asarray(A_original)
        A_new = np.asarray(A_new)
        dim = len(self.neighbors[0])
        assert dim == len(A_original) and dim == len(
            A_new), "A matrix doesn't match the dimension of the model"
        matrix = np.linalg.inv(A_original) @ A_new
        assert np.all(np.abs(np.round(matrix) - matrix) <
                      1e-7), "The supercell matrix must be integer valued"
        matrix = np.round(matrix)
        det = round(np.linalg.det(matrix))
        assert self.sym is None or len(
            self.sym) == 1, "No symmetries of the coefficients are allowed for this operation"
        new_neighbors = self.neighbors @ matrix.T
        n = len(self.params[0])
        new_band_count = n * det
        params = self.params_complex()
        # now get all integer positions in the cell defined by matrix
        # for that, compute the (half open) bounding box of matrix * [0,1[^3
        box = np.stack(np.meshgrid(*[[0, 1]]*dim), axis=-1).reshape(-1, dim)
        box = box @ matrix.T
        bounding_box = np.min(box, axis=0), np.max(box, axis=0)
        assert np.array(bounding_box).dtype == np.int64
        # now create a meshgrid inside the bounding box and select the points with inv(matrix) in [0, 1[
        box = np.stack(np.meshgrid(*[np.arange(bounding_box[0][d], bounding_box[1][d])
                       for d in range(dim)]), axis=-1).reshape(-1, dim)
        p_box = box @ np.linalg.inv(matrix).T
        # internal positions + origin (0)
        internal_positions = list(
            p_box[np.all((p_box >= 0-1e-7) & (p_box < 1-1e-7), axis=1)] @ A_new.T)
        assert len(internal_positions) == det
        # now build the new hamiltonian
        H_r = np.zeros((len(params), new_band_count,
                       new_band_count), dtype=np.complex128)
        neighbor_func = try_neighbor_function(self.neighbors)
        for k, nk in enumerate(new_neighbors):
            for i, pi in enumerate(internal_positions):
                for j, pj in enumerate(internal_positions):
                    m, mirror = neighbor_func(nk + pj - pi)
                    if m is not None:
                        H_r[k, i*n:(i+1)*n, j*n:(j+1) *
                            n] = params[m] if not mirror else np.conj(params[m].T)
        model = BandStructureModel.init_tight_binding(
            Symmetry.none(), new_neighbors, new_band_count, cos_reduced=cos_reduced, exp=exp)
        model.set_params_complex(H_r)
        return model

    def __add__(self, other) -> Self:
        if type(other) == type(self):
            # direct sum of the models -> combine the bandstructures
            assert self.cos_reduced == other.cos_reduced, "Bandstructure needs to be of the same type (cos_reduced is different)"
            assert self.exp == other.exp, "Bandstructure needs to be of the same type (exp is different)"
            assert np.linalg.norm(
                self.neighbors - other.neighbors) < 1e-8, "Bandstructure needs to be of the same type (neighbors are different)"
            res = self.copy()
            res.params = direct_sum(self.params, other.params)
            return res
        else:
            # shift the whole bandstructure
            res = self.copy()
            res.params[0] += np.eye(self.band_count()) * float(other)
            return res

    def __mul__(self, fac: float) -> Self:
        res = self.copy()
        res.params *= fac
        return res

    def plot_bands(self, k_smpl, *args, **kwargs):
        """Simple plot function for the bandstructure.
        Only useful if the k_smpl are a path in k-space.
        To create a complicated path, use `kpath.KPath` and then also use its plot function, because it is better.

        Args:
            k_smpl (arraylike(N_k, dim)): k-space samples for the plot.
        """
        plot_bands_generic(k_smpl, self.bands(k_smpl), *args, **kwargs)

# returns f_i_sym, df_i_sym, term_count, neighbors (transformed)
# if cos_reduced == True then the functions will use cos(kr)-1 instead of cos(kr)
# if exp == True then the exponential functions e^ikR will be used directly.


def _get_tight_binding_coeff_funcs(sym, neighbors: np.ndarray, cos_reduced=False, exp=False):
    assert type(neighbors) == np.ndarray, "neighbors must be an ndarray here, so it doesn't need a copy and can be linked to the class internal neighbors."

    cos_func = np.cos if not cos_reduced else lambda x: np.cos(x)-1
    if exp:
        coeff_funcs = [(lambda x: np.exp(1j*x))
                       if not cos_reduced else (lambda x: np.exp(1j*x)-1)]
        coeff_dfuncs = [lambda x: 1j*np.exp(1j*x)]
        coeff_neg_ddfuncs = [lambda x: np.exp(1j*x)]
        term_count = len(neighbors)
    elif sym.inversion:
        coeff_funcs = [cos_func]
        coeff_dfuncs = [lambda x: -np.sin(x)]
        coeff_neg_ddfuncs = [np.cos]
        term_count = len(neighbors)
    else:
        coeff_funcs = [cos_func, np.sin]
        coeff_dfuncs = [lambda x: -np.sin(x), np.cos]
        coeff_neg_ddfuncs = [np.cos, np.sin]
        term_count = len(neighbors) * 2 - 1

    scale = 2*np.pi

    def f_i_sym(k, i):
        assert i >= 0
        k = np.asarray(k)
        if i == 0:
            return np.ones_like(k[..., 0])
        res = 0.0
        part = i // len(neighbors)
        r_orig = neighbors[i % len(neighbors) + part]
        func = coeff_funcs[part]
        for s in sym.S:
            r = s @ r_orig
            res = res + func(scale * (k @ r))
        return res

    # 1. derivative of f_i
    def df_i_sym(k, i):
        assert i >= 0
        k = np.asarray(k)
        res = np.zeros_like(k)
        if i == 0:
            return res
        part = i // len(neighbors)
        r_orig = neighbors[i % len(neighbors) + part]
        func = coeff_dfuncs[part]
        for s in sym.S:
            r = s @ r_orig
            res = res + func(scale * (k @ r))[..., None] * r
        return res * scale

    # 2. derivative of f_i (based on the assumpton ddf = -(scale*r)**2*f)
    def ddf_i_sym(k, i):
        assert i >= 0
        k = np.asarray(k)
        res = np.zeros(k.shape + (k.shape[-1],))
        if i == 0:
            return res
        part = i // len(neighbors)
        r_orig = neighbors[i % len(neighbors) + part]
        func = coeff_neg_ddfuncs[part]
        for s in sym.S:
            r = s @ r_orig
            res = res - func(scale * (k @ r)
                             )[..., None, None] * np.tensordot(r, r, axes=0)
        return res * scale**2

    return f_i_sym, df_i_sym, ddf_i_sym, term_count, neighbors
