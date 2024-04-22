import numpy as np
from matplotlib import pyplot as plt
from symmetry import *
import sys

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
        self.symmetrizer = None
        self.params = np.asarray(params)
        self.sym = None
        self.cos_reduced = None
        self.exp = None
        pshape = np.shape(self.params)
        assert len(pshape) == 3
        assert pshape[1] == pshape[2]
        assert pshape[0] > 0

    def init_tight_binding(symmetry: Symmetry, neighbors, band_count, cos_reduced=False, exp=False):
        #symmetry.check_neighbors(neighbors)
        f_i_tb, df_i_tb, ddf_i_tb, term_count, neighbors = get_tight_binding_coeff_funcs(symmetry, np.eye(3), neighbors, cos_reduced=cos_reduced, exp=exp)
        model = BandStructureModel(f_i_tb, df_i_tb, np.zeros((term_count, band_count, band_count)), ddf_i=ddf_i_tb)
        model.sym = symmetry
        model.cos_reduced = cos_reduced
        model.neighbors = neighbors
        model.exp = exp
        return model
    
    def init_tight_binding_from_ref(symmetry: Symmetry, neighbors, k_smpl, ref_bands, band_offset=0, additional_bands=0, cos_reduced=False, exp=False):
        f_i_tb, df_i_tb, ddf_i_tb, term_count, neighbors = get_tight_binding_coeff_funcs(symmetry, np.eye(3), neighbors, cos_reduced=cos_reduced, exp=exp)
        model = BandStructureModel.init_from_ref(f_i_tb, df_i_tb, ddf_i_tb, term_count, k_smpl, ref_bands, band_offset, additional_bands)
        # TODO handle the case where cos_reduced=False, because there the params need to be processed here
        model.sym = symmetry
        model.cos_reduced = cos_reduced
        model.neighbors = neighbors
        model.exp = exp
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
        mat = np.zeros(np.shape(k)[:-1] + self.params[0].shape, dtype=np.complex128)
        params_shape = tuple([1 for _ in range(len(np.shape(k)[:-1]))]) + self.params[0].shape
        for i in range(1, len(self.params)):
            mat += np.asarray(self.f_i(k, i))[..., None, None] * self.params[i].reshape(params_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        # expect params[0] to always be hermitian! (this is very important for symmetrization)
        mat += np.asarray(self.f_i(k, 0))[..., None, None] * self.params[0].reshape(params_shape)
        return mat
    
    # derivative of f wrt k in the direction dk, outputshape (len(k), N, N)
    def df(self, k):
        mat = np.zeros(np.shape(k) + self.params[0].shape, dtype=np.complex128)
        params_shape = tuple([1 for _ in range(len(np.shape(k)))]) + self.params[0].shape
        for i in range(1, len(self.params)):
            mat += np.asarray(self.df_i(k, i))[..., None, None] * self.params[i].reshape(params_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.df_i(k, 0))[..., None, None] * self.params[0].reshape(params_shape)
        return mat
    
    # 2. derivative of f wrt k in the direction dk, outputshape (len(k), len(k), N, N)
    def ddf(self, k):
        mat = np.zeros(np.shape(k) + (np.shape(k)[-1],) + self.params[0].shape, dtype=np.complex128)
        params_shape = (1,) + tuple([1 for _ in range(len(np.shape(k)))]) + self.params[0].shape
        for i in range(1, len(self.params)):
            mat += np.asarray(self.ddf_i(k, i))[..., None, None] * self.params[i].reshape(params_shape)
        mat += np.conj(np.swapaxes(mat, -1, -2))
        mat += np.asarray(self.ddf_i(k, 0))[..., None, None] * self.params[0].reshape(params_shape)
        return mat

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

    def optimize(self, k_smpl, k_smpl_weights, ref_bands, band_weights, band_offset, iterations, batch_div=1, train_k0=True, regularization=1, learning_rate=1.0, verbose=True, max_accel_global=None, use_pinv=True):
        N = np.shape(self.params)[1]
        assert band_offset >= 0 and band_offset <= N - len(ref_bands[0])
        # reshape k_smpl_weights
        k_smpl_weights = np.broadcast_to(np.reshape([k_smpl_weights], (-1, 1)), (len(k_smpl), 1))
        # reshape normalized band_weights
        band_weights = np.broadcast_to(np.reshape([band_weights / np.mean(band_weights)], (1, -1)), (1, len(ref_bands[0])))
        # memoize self.f_i here using a rectangular matrix
        f_i = np.zeros((len(k_smpl), len(self.params)), dtype=np.complex128)
        for ki, k in enumerate(k_smpl):
            for i in range(len(self.params)):
                f_i[ki, i] = self.f_i(k, i)
        f_i[:, 0] /= 2 # divide by 2 because it is added without the symmetrization
        # find the perfect "anti-coefficients", such that c_i @ f_i.T = I
        c_i = np.conj(f_i) # classic gradient descent
        if batch_div == 1 and use_pinv: # improved optimization
            if verbose:
                print("preparing pseudoinverse")
            # NOTE: the order of k_smpl is arbitrary and not important for the following calculation
            c_i = np.linalg.pinv(f_i.T)
            #print(c_i)
            #print(c_i @ f_i.T) # weird matrix
            #print(f_i.T @ c_i) # identity
            if max_accel_global is None:
                max_accel_global = 1.0
            # normalize (does also work without this very slow step)
            #if verbose:
            #    print("normalizing pseudoinverse")
            #for i in range(len(self.params)):
            #    div = max(np.abs((c_i * f_i[i]).sum()), 1.0)
            #    c_i /= div
            #    max_accel_global *= div
            #for i in range(len(k_smpl)):
            #    c_i[i,:] /= max(np.abs((c_i[i] * f_i).sum()), 1.0)
            if verbose:
                print(f"maximal acceleration {max_accel_global}")
            # counteract the usual treatment:
            c_i *= (np.abs(f_i)**2).sum(1, keepdims=True)
            c_i *= len(k_smpl)
        if max_accel_global is None:
            max_accel_global = len(k_smpl)
        # start stochastic gradient descent
        last_add = np.zeros_like(self.params)
        self.normalize()
        #last_loss = self.loss(k_smpl, ref_bands, band_, band_offset)
        try:
            for iteration in range(iterations):
                batch = k_smpl[iteration % batch_div::batch_div]
                batch_weights = k_smpl_weights[iteration % batch_div::batch_div]
                batch_ref = ref_bands[iteration % batch_div::batch_div]
                batch_f_i = f_i[iteration % batch_div::batch_div]
                batch_c_i = c_i[iteration % batch_div::batch_div]
                top_pad = np.shape(self.params)[1] - len(batch_ref[0]) - band_offset
                batch_weights = batch_weights / np.mean(batch_weights)
                max_accel = min(len(batch), max_accel_global)

                # regularization
                if regularization != 1.0:
                    self.params[1:] *= regularization

                # faster f using cached values
                f = (self.params[None,...] * batch_f_i[...,None,None]).sum(1)
                f += np.conj(np.swapaxes(f, -1, -2))
                eigvals, eigvecs = np.linalg.eigh(f)
                #new_loss = np.linalg.norm(batch_ref - eigvals) / len(batch)**.5
                #last_loss = new_loss

                s = (np.abs(batch_f_i)**2).sum(1) # use f_i as weights, but normalize the whole step
                diff = batch_ref - eigvals[:,band_offset:][:,:len(batch_ref[0])]
                if batch_div == 1:
                    max_err = np.max(np.abs(diff), axis=0)
                # implementation of the following einsum
                #params_add = np.einsum("bik, bjk, bk, b, bn, k -> nij", eigvecs[:,:,band_offset:N-top_pad], np.conj(eigvecs[:,:,band_offset:N-top_pad]), diff, 2 / s, batch_f_i, weights, optimize="greedy") / len(batch)
                # but faster: (I don't know why it's faster...)
                diff *= band_weights

                if batch_div == 1:
                    loss = np.linalg.norm(diff) / len(batch)**.5 # this is how the loss is computed in self.error
                
                diff *= batch_weights
                diff *= np.reshape((0.5 / len(batch)) / s, (-1, 1))
                diff = eigvecs[:,:,band_offset:N-top_pad] @ (diff[...,None] * np.conj(np.swapaxes(eigvecs[:,:,band_offset:N-top_pad], 1, 2)))
                params_add = np.einsum("bij,bn->nij", diff, batch_c_i)
                if self.symmetrizer is not None:
                    params_add = self.symmetrizer(params_add)
                if not train_k0:
                    params_add[0] *= 0.0
                params_add *= learning_rate
                # impulse acceleration (beta given by the problem)
                params_add += last_add * (1 - 1 / max_accel)
                # change parameters (alpha = 1.0 with the chosen way to normalize the gradient)
                self.params = self.params + params_add
                last_add = params_add

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
        if self.symmetrizer is not None:
            self.params = self.symmetrizer(self.params)
            return # don't use the normalization if a symmetrizer is specified, as it might mess with that.
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
        #la, ev = np.linalg.eigh(self.f(0))
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

    # get the complex fourier coefficients H_r for the initially specified neighbors.
    # This function specifically works for models created with init_tight_binding_from_ref
    # This function also corrects for cos_reduced=True to get the actual H_r.
    def params_complex(self):
        if self.exp:
            H_r = np.array(self.params)
            if self.cos_reduced:
                H_r[0] -= np.sum(H_r[1:] + np.conj(np.swapaxes(H_r[1:], -1, -2)), axis=0)
            return H_r
        # TODO this probably doesn't need to be divided by 2 anymore...
        # TODO add a test for this function by comparing different models!
        H_r = np.array(self.params)
        if not self.sym.inversion:
            n = (len(self.params)+1)//2
            if self.cos_reduced:
                #H_r[1:n] += H_r[0]
                H_r[0] -= np.sum(H_r[1:n] + np.conj(np.swapaxes(H_r[1:n], -1, -2)), axis=0)
                pass
            H_r[1:n] += H_r[n:] * -1j
            H_r = H_r[:n]
        elif self.cos_reduced:
            H_r[0] -= np.sum(H_r[1:] + np.conj(np.swapaxes(H_r[1:], -1, -2)), axis=0)
        return H_r
    
    # set the params from the complex H_r form of parameters
    def set_params_complex(self, H_r):
        if self.exp:
            self.params = np.array(H_r)
            if self.cos_reduced:
                self.params[0] += np.sum(self.params[1:] + np.conj(np.swapaxes(self.params[1:], -1, -2)), axis=0)
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

    def save(self, filename):
        opt = np.get_printoptions()
        np.set_printoptions(precision=16, suppress=False, threshold=100000)
        with open(filename, "w") as file:
            file.write(repr(self.params_complex()) + ",\\\n")
            file.write(repr(self.neighbors) + ",\\\n")
            file.write(repr(self.sym.S) + ",\\\n")
            file.write(repr(self.sym.inversion) + "\n")
        np.set_printoptions(**opt) # reset printoptions

    # import a tight binding model into the given param format (cos_reduced, exp)
    def load(filename, cos_reduced=False, exp=False):
        with open(filename, "r") as file:
            H_r_repr = " ".join(file.readlines())
            H_r, neighbors, S, inversion = eval(H_r_repr.replace("array", "np.array"))
            model = BandStructureModel.init_tight_binding(Symmetry(S, inversion=inversion), neighbors, len(H_r[0]), cos_reduced=True, exp=False)
            model.set_params_complex(H_r)
        return model

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
# if exp == True then the exponential functions e^ikR will be used directly.
def get_tight_binding_coeff_funcs(sym, basis_transform, neighbors, cos_reduced=False, exp=False):
    neighbors = np.asarray(neighbors)

    cos_func = np.cos if not cos_reduced else lambda x: np.cos(x)-1
    if exp:
        coeff_funcs = [(lambda x: np.exp(1j*x)) if not cos_reduced else (lambda x: np.exp(1j*x)-1)]
        coeff_dfuncs = [lambda x: 1j*np.exp(1j*x)]
        term_count = len(neighbors)
    elif sym.inversion:
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
        d = np.zeros_like(k)
        if i == 0:
            return d
        res = d
        scale = 2*np.pi
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
        d = np.zeros(k.shape + (k.shape[-1],))
        if i == 0:
            return d
        res = d
        scale = 2*np.pi
        part = i // len(neighbors)
        r_orig = neighbors[i % len(neighbors) + part]
        func = coeff_dfuncs[part]
        for s in sym.S:
            r = s @ r_orig
            res = res - func(scale * (k @ r))[..., None, None] * np.tensordot(r, r, axes=0)
        return res * scale**2
    
    return f_i_sym, df_i_sym, ddf_i_sym, term_count, neighbors


def test_bandstructure_params():
    neighbors = ((0, 0, 0), (1, 0, 0))#, (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    k_smpl = np.array([(0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.4, 0.5, 0.6), (0.6, 0.7, 0.8), (0.7, 0.8, 0.9)])
    k_smpl, _ = Symmetry.cubic(True).realize_symmetric(k_smpl)

    for sym in [Symmetry.none(), Symmetry.inv()]:
        tb00 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=False, exp=False) # 0
        tb01 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=False, exp=True)  # 1
        tb10 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=True,  exp=False) # 2
        tb11 = BandStructureModel.init_tight_binding(sym, neighbors, 2, cos_reduced=True,  exp=True)  # 3
        tb = [tb00, tb01, tb10, tb11]

        # set one model and convert to normal H_r matrices and apply them to the other models
        # then test if the hamiltonians at some non symmetric k-points are the same
        tb_other = list(enumerate(tb)) # even check with itself
        for i in range(len(tb)):
            tb_test = tb[i]
            tb_test.params = np.random.standard_normal(tb_test.params.shape) + 1j*np.random.standard_normal(tb_test.params.shape)
            tb_test.normalize() # fixes non symmetric matrices from random start
            tb_test_f = tb_test.f(k_smpl)
            tb_test_df = tb_test.df(k_smpl)
            tb_test_ddf = tb_test.ddf(k_smpl)
            #print()
            #print("first params", np.ravel(tb_test.params))
            #print(tb_test.f([(0,0,0)]), tb_test.f([(0,0,1/2)]), tb_test.f([(0,0,1/4)]))
            H_r = tb_test.params_complex()
            #print("first H_r", np.ravel(H_r))
            for j, tb_test2 in tb_other:
                tb_test2.set_params_complex(H_r)
                #print("tb_test2", np.ravel(tb_test2.params))
                #print(tb_test2.f([(0,0,0)]), tb_test2.f([(0,0,1/2)]), tb_test2.f([(0,0,1/4)]))
                assert np.linalg.norm(tb_test_f - tb_test2.f(k_smpl)) < 1e-7, f"reconstructed Hamiltonians don't match ({i} vs {j}, inversion: {sym.inversion})"
                assert np.linalg.norm(tb_test_df - tb_test2.df(k_smpl)) < 1e-7, f"reconstructed Hamiltonian 1. derivative doesn't match ({i} vs {j}, inversion: {sym.inversion})"
                assert np.linalg.norm(tb_test_ddf - tb_test2.ddf(k_smpl)) < 1e-7, f"reconstructed Hamiltonian 2. derivative doesn't match ({i} vs {j}, inversion: {sym.inversion})"
                H_r2 = tb_test2.params_complex()
                assert np.linalg.norm(H_r - H_r2) < 1e-14, f"Reconstruction doesn't match ({i} vs {j}, inversion: {sym.inversion})"

def test_bandstructure_optimize():
    neighbors = ((0, 0, 0), (1, 0, 0))#, (1, 1, 0), (1, 1, 1))
    neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    sym = Symmetry.none()
    n = 5
    tb00 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=False, exp=False) # 0
    tb01 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=False, exp=True)  # 1
    tb10 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=True,  exp=False) # 2
    tb11 = BandStructureModel.init_tight_binding(sym, neighbors, n, cos_reduced=True,  exp=True)  # 3
    tb = [tb00, tb01, tb10, tb11]

    for k_count in [1]:
        # TODO use k_count
        k_smpl = np.array([(0.1, 0.2, 0.3),])
        ref_bands = np.arange(n).astype(np.float64)[None,:]

        # test whether a single optimize step solves the single k case
        for i, tb in enumerate(tb):
            for kw in [1, 1.5]:
                for bw in [1, 1.5]:
                    tb.params = np.random.random(tb.params.shape)
                    tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, verbose=False, use_pinv=False)
                    assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} was incorrect for model {i}, kw = {kw}, bw = {bw}"
                    tb.params = np.random.random(tb.params.shape)
                    tb.optimize(k_smpl, kw, ref_bands, bw, 0, 1, verbose=False, use_pinv=True)
                    assert np.linalg.norm(tb.bands(k_smpl) - ref_bands) < 1e-7, f"{tb.bands(k_smpl)} was incorrect for model {i}, kw = {kw}, bw = {bw}"
