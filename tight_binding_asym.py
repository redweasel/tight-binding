import numpy as np
import scipy
from matplotlib import pyplot as plt
from symmetry import *
from unitary_representations import *

def random_hermitian(n):
    h = (np.random.random((n, n)) * 2 - 1) + 1j * (2 * np.random.random((n, n)) - 1)
    return h + np.conj(h.T)

class HermitianFourierSeries:
    def __init__(self, neighbors, H_r):
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        assert len(neighbors) == len(H_r)
        assert np.shape(H_r)[1] == np.shape(H_r)[2]
        assert len(np.shape(neighbors)) == 2
        assert len(np.shape(H_r)) == 3
        self.neighbors = np.asarray(neighbors)
        self.H_r = np.asarray(H_r)

    def dim(self):
        return len(self.neighbors[0])

    def f_i(self, k, i):
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        return np.exp(2j*np.pi*(k @ r))

    # derivative of f_i
    def df_i(self, k, i):
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        return 2j*np.pi * np.exp(2j*np.pi*(k @ r))[...,None] * np.asarray(r)
    
    # second derivative of f_i
    def ddf_i(self, k, i):
        assert i >= 0
        k = np.asarray(k)
        r = self.neighbors[i]
        r_sqr = np.asarray(r)[:,None] * np.asarray(r)[None,:]
        return (2j*np.pi)**2 * np.exp(2j*np.pi*(k @ r))[...,None,None] * r_sqr

    # full function for the band structure
    def f(self, k):
        mat = np.zeros(np.shape(k)[:-1] + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = tuple([1 for _ in range(len(np.shape(k)[:-1]))]) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += np.asarray(self.f_i(k, i))[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        # expect H_r[0] to always be hermitian! (this is very important for symmetrization)
        mat += np.asarray(self.f_i(k, 0))[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat
    
    # derivative of f wrt k in the direction dk, outputshape (dim(k), N, N)
    def df(self, k):
        mat = np.zeros(np.shape(k) + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = tuple([1 for _ in range(len(np.shape(k)))]) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += np.asarray(self.df_i(k, i))[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.df_i(k, 0))[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat
    
    # second derivative of f wrt k in the direction dk, outputshape (dim(k), dim(k), N, N)
    def ddf(self, k):
        mat = np.zeros(np.shape(k) + (np.shape(k)[-1],) + self.H_r[0].shape, dtype=np.complex128)
        H_r_shape = (1,) + tuple([1 for _ in range(len(np.shape(k)))]) + self.H_r[0].shape
        for i in range(1, len(self.H_r)):
            mat += np.asarray(self.ddf_i(k, i))[..., None, None] * self.H_r[i].reshape(H_r_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.ddf_i(k, 0))[..., None, None] * self.H_r[0].reshape(H_r_shape)
        return mat
    
    def copy(self):
        return HermitianFourierSeries(self.neighbors.copy(), self.H_r.copy())
    
    def unit_matrix(neighbors, n):
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        S_r = np.zeros((len(neighbors), n, n))
        S_r[0] = np.eye(n)
        return HermitianFourierSeries(neighbors, S_r)

class AsymTightBindingModel:
    # initialize a tight binding model
    def __init__(self, H: HermitianFourierSeries, S=None):
        assert type(H) == HermitianFourierSeries
        assert S is None or type(S) == HermitianFourierSeries
        self.H = H
        self.S = HermitianFourierSeries.unit_matrix([(0,)*H.dim()], np.shape(H.H_r)[1]) if S is None else S

    # initialize a tight binding model from a reference computation
    def init_from_ref(neighbors, k_smpl, ref_bands):
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
        tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        tb.normalize()
        return tb

    def dim(self):
        return self.H.dim() # dimension from neighbors

    def save(self, filename, format=None):
        if format is None:
            if filename.endswith(".json"):
                format = "json"
            else:
                raise ValueError("unrecognised format for file " + filename)
        if format not in {"json"}:
            raise ValueError('only supported format is "json"')
        opt = np.get_printoptions()
        np.set_printoptions(precision=16, suppress=False, threshold=100000)
        if format == "json":
            import json_tb_format
            json_tb_format.save(filename, self.H.neighbors, self.H.H_r)
            # TODO save the S matrix part as well
    
        np.set_printoptions(**opt) # reset printoptions

    # import a tight binding model
    def load(filename, format=None):
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
                    model = AsymTightBindingModel(neighbors, H_r)
        elif format == "wannier90":
            import wannier90_tb_format as tb_fmt
            neighbors, H_r, w_r_H_r = tb_fmt.load(filename)
            model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        elif format == "json":
            import json_tb_format
            neighbors, H_r = json_tb_format.load(filename)
            model = AsymTightBindingModel(HermitianFourierSeries(neighbors, H_r))
        return model

    # returns the weighted loss (standard deviation) and the maximal error per band
    def error(self, k_smpl, ref_bands, band_weights, band_offset):
        bands = self.bands(k_smpl)
        err = (bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands)
        max_err = np.max(np.abs(err), axis=0)
        err *= np.reshape(band_weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5, max_err
    
    def loss(self, k_smpl, ref_bands, band_weights, band_offset):
        bands = self.bands(k_smpl)
        err = (bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands) * np.reshape(band_weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5
        #return np.max(np.abs(bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands))
    
    def optimize(self, k_smpl, k_smpl_weights, ref_bands, band_weights, band_offset, iterations, batch_div=1, learning_rate=1.0, train_k0=True, use_pinv=True, max_accel_global=None, regularization=1.0, keep_zeros=False, verbose=True):
        N = np.shape(self.H.H_r)[1]
        assert band_offset >= 0 and band_offset <= N - len(ref_bands[0])
        # reshape k_smpl_weights
        k_smpl_weights = np.broadcast_to(np.reshape([k_smpl_weights], (-1, 1)), (len(k_smpl), 1))
        # reshape normalized band_weights
        band_weights = np.broadcast_to(np.reshape([band_weights / np.mean(band_weights)], (1, -1)), (1, len(ref_bands[0])))
        # mask for keep_zeros
        H_r_mask = np.where(np.abs(self.H.H_r) < 1e-14, 0.0, 1.0)
        # memoize self.H.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.H.H_r)), dtype=np.complex128)
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.H.H_r)):
                f_i[ki, i] = self.H.f_i(k, i)
        f_i[:, 0] /= 2 # divide by 2 because it is added without the symmetrization
        # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
        c_i = np.conj(f_i) # modified gradient descent with conjugated derivatives
        if batch_div == 1 and use_pinv: # improved optimization
            if verbose:
                print("preparing pseudoinverse")
            # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
            c_i = np.linalg.pinv(f_i.T)
            if max_accel_global is None:
                max_accel_global = 1.0
            if verbose:
                print(f"maximal acceleration {max_accel_global}")
            # counteract the usual treatment:
            c_i *= (np.abs(f_i)**2).sum(1, keepdims=True)
            c_i *= len(k_smpl)
        if max_accel_global is None:
            max_accel_global = len(k_smpl)
        # start stochastic gradient descent
        last_add = np.zeros_like(self.H.H_r)
        self.normalize()
        #last_loss = self.loss(k_smpl, ref_bands, band_, band_offset)
        try:
            for iteration in range(iterations):
                batch = k_smpl[iteration % batch_div::batch_div]
                batch_weights = k_smpl_weights[iteration % batch_div::batch_div]
                batch_ref = ref_bands[iteration % batch_div::batch_div]
                batch_f_i = f_i[iteration % batch_div::batch_div]
                batch_c_i = c_i[iteration % batch_div::batch_div]
                top_pad = np.shape(self.H.H_r)[1] - len(batch_ref[0]) - band_offset
                batch_weights = batch_weights / np.mean(batch_weights)
                max_accel = min(len(batch), max_accel_global)

                # regularization
                if regularization != 1.0:
                    self.H.H_r[1:] *= regularization

                # faster f using cached values
                f = (self.H.H_r[None,...] * batch_f_i[...,None,None]).sum(1)
                f += np.conj(np.swapaxes(f, -1, -2))
                eigvals, eigvecs = np.linalg.eigh(f)

                s = (np.abs(batch_f_i)**2).sum(1) # use f_i as weights, but normalize the whole step
                diff = batch_ref - eigvals[:,band_offset:][:,:len(batch_ref[0])]
                if batch_div == 1:
                    max_err = np.max(np.abs(diff), axis=0)
                # implementation of the following einsum
                #H_r_add = np.einsum("bik, bjk, bk, b, bn, k -> nij", eigvecs[:,:,band_offset:N-top_pad], np.conj(eigvecs[:,:,band_offset:N-top_pad]), diff, 2 / s, batch_f_i, weights, optimize="greedy") / len(batch)
                # but faster: (I don't know why it's faster...)
                diff *= band_weights

                if batch_div == 1:
                    loss = np.linalg.norm(diff) / len(batch)**.5 # this is how the loss is computed in self.error
                
                diff *= batch_weights
                diff *= np.reshape((0.5 / len(batch)) / s, (-1, 1))
                diff = eigvecs[:,:,band_offset:N-top_pad] @ (diff[...,None] * np.conj(np.swapaxes(eigvecs[:,:,band_offset:N-top_pad], 1, 2)))
                H_r_add = np.einsum("bij,bn->nij", diff, batch_c_i)
                if not train_k0:
                    H_r_add[0] *= 0.0
                if keep_zeros:
                    H_r_add *= H_r_mask
                H_r_add *= learning_rate
                # impulse acceleration (beta given by the problem)
                H_r_add += last_add * (1 - 1 / max_accel)
                # change parameters (alpha = 1.0 with the chosen way to normalize the gradient)
                self.H.H_r = self.H.H_r + H_r_add
                last_add = H_r_add

                if (iteration % 100 == 0 or batch_div == 1) and verbose:
                    #print(f"loss: {new_loss:.2e} alpha: {alpha:.1e}")
                    if batch_div != 1:
                        loss, max_err = self.error(k_smpl, ref_bands, band_weights, band_offset)
                    print(f"\rloss: {loss:.2e} (max band-error {max_err})", end="")
        except KeyboardInterrupt:
            print("\naborted")
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, band_weights, band_offset)
        if verbose:
            print(f"\rfinal loss: {l:.2e} (max band-error {err})")

    def normalize(self):
        self.H.H_r[0] = (self.H.H_r[0] + self.H.H_r[0].T.conj()) / 2
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

    # get the complex fourier coefficients H_r
    def H_r_complex(self):
        return self.H.H_r
    
    # set H_r from the complex fourier coefficients H_r
    def set_from_complex(self, H_r):
        self.H.H_r = np.asarray(H_r).astype(np.complex128)
    
    def bands(self, k_smpl):
        return scipy.linalg.eigvalsh(self.H.f(k_smpl), self.S.f(k_smpl))
    
    def __call__(self, k_smpl):
        return self.bands(k_smpl)
    
    # gradients of the bands (group velocities) in the shape (len(k_smpl), len(k), len(bands))
    # returns (bands, bands_grad)
    def bands_grad(self, k_smpl):
        bands, ev = scipy.linalg.eigh(self.H.f(k_smpl), self.S.f(k_smpl))
        dH = self.H.df(k_smpl)
        dS = self.S.df(k_smpl)
        grads = np.real(np.einsum("mji, mnjk, mki -> mni", np.conj(ev), dH, ev))
        # TODO check!
        grads += np.real(np.einsum("mji, mnjk, mki -> mni", np.conj(ev), dS, ev)) * -bands[:,:,None]
        return bands, grads
    
    # hessian, gradients wrt to k of the bands at k_smpl
    # returns (bands, grads, hessians)
    def bands_grad_hess(self, k_smpl):
        # TODO add dS and ddS!!
        bands, ev = scipy.linalg.eigh(self.H.f(k_smpl), self.S.f(k_smpl))
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

    def copy(self):
        return AsymTightBindingModel(self.H.copy(), self.S.copy())
    
    # Generate a tight binding model (with self.neighbors set)
    # for a supercell defined as A' = A Λ
    # where Λ is a non singular integer valued matrix.
    def supercell(self, A_original, A_new, cos_reduced=False, exp=False):
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
        H_r = self.H_r_complex()
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
        model = AsymTightBindingModel(new_neighbors, H_r)
        return model


def test_asym_tight_binding_optimize():
    neighbors = ((0, 0, 0), (1, 0, 0))#, (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    n = 5
    tb = AsymTightBindingModel(HermitianFourierSeries(neighbors, np.zeros((len(neighbors), n, n))))

    for k_count in [1]:
        # TODO use k_count
        k_smpl = np.array([(0.1, 0.2, 0.3),])
        ref_bands = np.arange(n).astype(np.float64)[None,:]

        # test whether a single optimize step solves the single k case
        for kw in [1, 1.5]:
            for bw in [1, 1.5]:
                tb.set_from_complex(np.random.random(tb.H_r_complex().shape))
                tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, use_pinv=False, verbose=False)
                assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} without pinv was incorrect, kw = {kw}, bw = {bw}"
                tb.set_from_complex(np.random.random(tb.H_r_complex().shape))
                tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, use_pinv=True, verbose=False)
                assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} with pinv was incorrect, kw = {kw}, bw = {bw}"
