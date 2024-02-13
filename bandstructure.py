import numpy as np
from matplotlib import pyplot as plt

def plot_bands_generic(k_smpl, bands, *args, **kwargs):
    plt.gca().set_prop_cycle(None)
    for i in range(len(bands[0])):
        if len(np.ravel(k_smpl)) == len(k_smpl):
            plt.plot(k_smpl, [bands[j][i] for j in range(len(k_smpl))], *args, **kwargs)
        else:
            plt.plot([bands[j][i] for j in range(len(k_smpl))], *args, **kwargs)

def random_hermitian(n):
    h = (np.random.random((n, n)) * 2 - 1) + 1j * (2 * np.random.random((n, n)) - 1)
    return h + np.conj(h.T)

class BandStructureModel:
    def __init__(self, f_i, df_i, params, ddf_i=None):
        self.f_i = f_i
        self.df_i = df_i
        self.ddf_i = ddf_i
        self.params = np.asarray(params)
        self.sym = None
        self.cos_reduced = None
        pshape = np.shape(self.params)
        assert len(pshape) == 3
        assert pshape[1] == pshape[2]
        assert pshape[0] > 0

    def init_tight_binding_from_ref(symmetry, neighbors, k_smpl, ref_bands, band_offset=0, additional_bands=0, cos_reduced=False):
        f_i_tb, df_i_tb, ddf_i_tb, term_count, neighbors = get_tight_binding_coeff_funcs(symmetry, np.eye(3), neighbors, cos_reduced=cos_reduced)
        model = BandStructureModel.init_from_ref(f_i_tb, df_i_tb, ddf_i_tb, term_count, k_smpl, ref_bands, band_offset, additional_bands)
        model.sym = symmetry
        model.cos_reduced = cos_reduced
        return model

    def init_from_ref(f_i, df_i, ddf_i, param_count, k_smpl, ref_bands, band_offset=0, additional_bands=0):
        # assuming the first matrix is a constant term
        k0_index = np.argmin(np.linalg.norm(k_smpl, axis=-1))
        assert np.linalg.norm(k_smpl[k0_index]) == 0
        k0_bands = ref_bands[k0_index]
        left_pad = band_offset
        right_pad = additional_bands - band_offset
        k0_bands = np.concatenate([[k0_bands[0]]*left_pad, k0_bands, [k0_bands[-1]]*right_pad])
        scale = 0.001
        params = [np.diag(k0_bands)] + [random_hermitian(len(k0_bands)) * scale for _ in range(param_count-1)]
        return BandStructureModel(f_i, df_i, params, ddf_i)
    
    # full function for the band structure
    def f(self, k):
        mat = np.zeros(np.shape(k)[:-1] + self.params[0].shape, dtype=self.params.dtype)
        params_shape = tuple([1 for _ in range(len(np.shape(k)[:-1]))]) + self.params[0].shape
        for i in range(len(self.params)):
            mat += np.asarray(self.f_i(k, i))[..., None, None] * self.params[i].reshape(params_shape)
        return mat
    
    # derivative of f wrt k in the direction dk, outputshape (len(k), N, N)
    def df(self, k):
        mat = np.zeros(np.shape(k) + self.params[0].shape, dtype=self.params.dtype)
        params_shape = tuple([1 for _ in range(len(np.shape(k)))]) + self.params[0].shape
        for i in range(len(self.params)):
            mat += np.asarray(self.df_i(k, i))[..., None, None] * self.params[i].reshape(params_shape)
        return mat
    
    # 2. derivative of f wrt k in the direction dk, outputshape (len(k), N, N)
    def ddf(self, k):
        mat = np.zeros(np.shape(k) + (np.shape(k)[-1],) + self.params[0].shape, dtype=self.params.dtype)
        params_shape = (1,) + tuple([1 for _ in range(len(np.shape(k)))]) + self.params[0].shape
        for i in range(len(self.params)):
            mat += np.asarray(self.ddf_i(k, i))[..., None, None] * self.params[i].reshape(params_shape)
        return mat

    # returns the weighted loss (standard deviation) and the maximal error per band
    def error(self, k_smpl, ref_bands, weights, band_offset):
        bands = self.bands(k_smpl)
        err = (bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands)
        max_err = np.max(err, axis=0)
        err *= np.reshape(weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5, max_err
    
    def loss(self, k_smpl, ref_bands, weights, band_offset):
        bands = self.bands(k_smpl)
        err = (bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands) * np.reshape(weights, (1, -1))
        return np.linalg.norm(err) / len(k_smpl)**0.5
        #return np.max(np.abs(bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands))

    def optimize(self, k_smpl, k_smpl_weights, ref_bands, weights, band_offset, iterations, batch_div=1, train_k0=True, regularization=1, learning_rate=1.0):
        N = np.shape(self.params)[1]
        assert band_offset >= 0 and band_offset <= N - len(ref_bands[0])
        # memoize self.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.params)))
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.params)):
                f_i[ki, i] = self.f_i(k, i)
        # reshape k_smpl_weights
        k_smpl_weights = np.broadcast_to(np.reshape([k_smpl_weights], (-1, 1)), (len(k_smpl), 1))
        weights = np.broadcast_to(np.reshape([weights], (1, -1)), (1, len(ref_bands[0])))
        # start stochastic gradient descent
        last_add = np.zeros_like(self.params)
        #last_loss = self.loss(k_smpl, ref_bands, weights, band_offset)
        try:
            for iteration in range(iterations):
                batch = k_smpl[iteration % batch_div::batch_div]
                batch_weights = k_smpl_weights[iteration % batch_div::batch_div]
                batch_ref = ref_bands[iteration % batch_div::batch_div]
                batch_f_i = f_i[iteration % batch_div::batch_div]
                top_pad = np.shape(self.params)[1] - len(batch_ref[0]) - band_offset
                batch_weights = batch_weights / np.mean(batch_weights)

                # regularization
                self.params[1:] *= regularization

                # faster f using cached values
                f = (self.params[None,...] * batch_f_i[...,None,None]).sum(1)
                eigvals, eigvecs = np.linalg.eigh(f)
                #new_loss = np.linalg.norm(batch_ref - eigvals) / len(batch)**.5
                #last_loss = new_loss

                s = (np.abs(batch_f_i)**2).sum(1) # use f_i as weights, but normalize the whole step
                diff = batch_ref - eigvals[:,band_offset:][:,:len(batch_ref[0])]
                # implementation of the following einsum
                #params_add = np.einsum("bik, bjk, bk, b, bn, k -> nij", eigvecs[:,:,band_offset:N-top_pad], np.conj(eigvecs)[:,:,band_offset:N-top_pad], diff, 2 / s, batch_f_i, weights, optimize="greedy") / len(batch)
                # but faster: (I don't know why it's faster...)
                diff *= weights
                diff *= batch_weights
                diff *= np.reshape((2.0 / len(batch)) / s, (-1, 1))
                diff = eigvecs[:,:,band_offset:N-top_pad] @ (diff[...,None] * np.swapaxes(np.conj(eigvecs)[:,:,band_offset:N-top_pad], 1, 2))
                params_add = np.einsum("bij,bn->nij", diff, batch_f_i)
                if not train_k0:
                    params_add[0] *= 0.0
                params_add *= learning_rate
                # impulse acceleration (beta given by the problem)
                params_add += last_add * (1 - 1 / len(batch))
                # change parameters (alpha = 1.0 with the chosen way to normalize the gradient)
                self.params += params_add
                last_add = params_add

                if iteration % 100 == 0:
                    #print(f"loss: {new_loss:.2e} alpha: {alpha:.1e}")
                    l, err = self.error(k_smpl, ref_bands, weights, band_offset)
                    print(f"\rloss: {l:.2e} (max band-error {err})", end="")
        except KeyboardInterrupt:
            print("\naborted")
        self.normalize()
        l, err = self.error(k_smpl, ref_bands, weights, band_offset)
        print(f"\rfinal loss: {l:.2e} (max band-error {err})")

    def normalize(self):
        la, ev = np.linalg.eigh(self.params[0])
        #la, ev = np.linalg.eigh(self.f(0))
        for i in range(len(self.params)):
            self.params[i] = np.conj(ev.T) @ self.params[i] @ ev
        # normalize a little more using complex reflections on the second matrix
        if len(self.params) > 1:
            for i in range(1, len(self.params[1])):
                x = self.params[1][i-1, i]
                a = np.abs(x)
                if a != 0:
                    sign = np.conj(x) / a
                    self.params[:, :, i] *= sign
                    self.params[:, i, :] *= np.conj(sign)

    # get the complex fourier coefficients H_r for the initially specified neighbors.
    # This function also corrects for cos_reduced=True to get the actual H_r.
    def params_complex(self):
        H = self.params / 2
        if not self.sym.inversion:
            n = (len(self.params)+1)//2
            if self.cos_reduced:
                H[1:n] += H[0]
            H[1:n] += H[n:] * 1j
            H = H[:n]
        elif self.cos_reduced:
            H[1:] += H[0]
        return H

    def bands(self, k_smpl):
        return np.linalg.eigvalsh(self.f(k_smpl))

    def __call__(self, k_smpl):
        return self.bands(k_smpl)

    # gradients of the bands (group velocities) in the shape (len(k_smpl), len(k), len(bands))
    # returns (bands, bands_grad)
    def bands_grad(self, k_smpl):
        bands, ev = np.linalg.eigh(self.f(k_smpl))
        df = self.df(k_smpl)
        # all of the following 3 are about the same speed for some reason
        #grads = np.real(np.einsum("mij, mnjk, mki -> mni", np.conj(np.swapaxes(ev, 1, 2)), df, ev))
        grads = np.real(np.einsum("mji, mnjk, mki -> mni", np.conj(ev), df, ev))
        #grads = np.real(np.diagonal(np.conj(np.swapaxes(ev, 1, 2))[:,None,:,:] @ df @ ev[:,None,:,:], axis1=2, axis2=3))
        return bands, grads
    
    # hessian, gradients wrt to k of the bands at k_smpl
    # returns (bands, grads, hessians)
    def bands_grad_hess(self, k_smpl):
        bands, ev = np.linalg.eigh(self.f(k_smpl))
        df = self.df(k_smpl)
        ddf = self.ddf(k_smpl)
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
        return BandStructureModel(self.f_i, self.df_i, self.params.copy())

    def plot_bands(self, k_smpl, *args, **kwargs):
        plot_bands_generic(k_smpl, self.bands(k_smpl), *args, **kwargs)

# returns f_i_sym, df_i_sym, term_count, neighbors (transformed)
# if cos_reduced == True then the functions will use cos(kr)-1 instead of cos(kr)
def get_tight_binding_coeff_funcs(sym, basis_transform, neighbors, cos_reduced=False):
    # sc crystal
    basis_transform = np.eye(3)
    # bcc crystal
    #basis_transform = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2.0
    # fcc crystal
    #basis_transform = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0
    neighbors = (basis_transform @ np.asarray(neighbors).T).T

    cos_func = np.cos if not cos_reduced else lambda x: np.cos(x)-1
    if sym.inversion:
        coeff_funcs = [cos_func]
        coeff_dfuncs = [lambda x: -np.sin(x)]
        term_count = len(neighbors)
    else:
        coeff_funcs = [cos_func, np.sin]
        coeff_dfuncs = [lambda x: -np.sin(x), np.cos]
        term_count = len(neighbors) * 2 - 1

    def f_i_sym(k, i):
        assert i >= 0
        k = np.asarray(k)
        if i == 0:
            return np.ones_like(k[..., 0])
        res = 0.0
        scale = 2*np.pi
        r_orig = neighbors[i % len(neighbors)]
        func = coeff_funcs[i // len(neighbors)]
        for s in sym.S:
            r = s @ r_orig
            res += func(scale * (k @ r))
        return res

    # 1. derivative of f_i
    def df_i_sym(k, i):
        assert i >= 0
        k = np.asarray(k)
        d = np.zeros_like(k)
        if i == 0:
            return d
        res = d
        scale = 2*np.pi
        r_orig = neighbors[i % len(neighbors)]
        func = coeff_dfuncs[i // len(neighbors)]
        for s in sym.S:
            r = s @ r_orig
            res += func(scale * (k @ r))[..., None] * r
        return res * scale
    
    # 2. derivative of f_i
    def ddf_i_sym(k, i):
        assert i >= 0
        k = np.asarray(k)
        d = np.zeros(k.shape + (k.shape[-1],))
        if i == 0:
            return d
        res = d
        scale = 2*np.pi
        r_orig = neighbors[i % len(neighbors)]
        func = coeff_dfuncs[i // len(neighbors)]
        for s in sym.S:
            r = s @ r_orig
            res += func(scale * (k @ r))[..., None, None] * np.tensordot(r, r, axes=0)
        return res * scale**2
    
    return f_i_sym, df_i_sym, ddf_i_sym, term_count, neighbors
