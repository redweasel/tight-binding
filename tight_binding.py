import numpy as np
from matplotlib import pyplot as plt
from symmetry import *
from unitary_representations import *

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

class TightBindingModel:
    # initialize a tight binding model from a symmetry and initial parameters
    # TODO add a unitary representation for the symmetry of the hamilton operator
    def __init__(self, unitary_repr: UnitaryRepresentation, neighbors, params):
        assert len(neighbors) > 0
        assert np.linalg.norm(neighbors[0]) == 0, "the first neighbor needs to be the 0 coordinate"
        assert np.shape(unitary_repr.U)[1] == np.shape(params)[1]
        self.neighbors = neighbors
        self.unitary_repr = unitary_repr
        self.sym = unitary_repr.sym
        self.symmetrizer = self.unitary_repr.symmetrizer(self.neighbors)
        #self.unitary_repr.sym.check_neighbors(neighbors) # done in symmetrizer (this check takes long, so don't do it twice)
        self.params = np.asarray(params)
        pshape = np.shape(self.params)
        assert len(pshape) == 3
        assert pshape[1] == pshape[2]
        # checks, that neighbors and sym fits params
        param_count = len(neighbors)
        if not self.sym.inversion:
            param_count = param_count * 2 - 1 # add separate sin Terms
        else:
            # make the onsite matrix blockdiagonal
            n = unitary_repr.inv_split
            self.params[0][:n,n:] = 0
            self.params[0][n:,:n] = 0
        assert pshape[0] == param_count
        # TODO sanity checks for symmetrically equivalent neighbors (those are not allowed)

    # initialize a tight binding model from a reference computation
    def init_from_ref(u_repr: UnitaryRepresentation, neighbors, k_smpl, ref_bands, use_repr_order=False, verbose=True):
        assert len(k_smpl[0]) == len(neighbors[0])
        dim = len(neighbors[0])
        param_count = len(neighbors)
        # first matrix is always constant term
        assert np.linalg.norm(neighbors[0]) == 0
        if not u_repr.sym.inversion:
            param_count = param_count * 2 - 1 # double up everything that isn't the 0 term
        # make the model correct for k=0
        # that means, it also needs to respect symmetries
        # 1. get the bandstructure at k=0 (should always be included!)
        k0_index = np.argmin(np.linalg.norm(k_smpl, axis=-1))
        assert np.linalg.norm(k_smpl[k0_index]) == 0
        k0_bands = ref_bands[k0_index]
        # TODO quantize k0_bands in a stable manner (can't use np.round(k0_bands, n), since it can separate very closeby points)
        k0_bands = np.round(k0_bands, 5) # bad/wrong
        #print(k0_bands)
        # count degeneracies <-> symmetry!
        unique = set(k0_bands)
        sym = []
        for i in range(1, dim+1):
            sym.append([b for b in unique if len([1 for b2 in k0_bands if b2 == b]) == i])
        if verbose:
            print("unique band energies:", len(unique))
            print("irreducible symmetry sizes (data): ", { i+1: len(s) for i, s in enumerate(sym) })
        # find the symmetric subspaces for the representation
        groups, counts = u_repr.subspaces()
        groups2 = [list(groups[counts == i]) for i in range(1, dim+1)]
        if verbose:
            print("irreducible symmetry sizes (model):", { i+1: len(g) for i, g in enumerate(groups2) })
        # check if the data can fit into the unitary representation
        # big symmetry groups in the data can be represented by small symmetry groups in the model
        # small symmetry groups in the data can NOT be represented by big symmetry groups in the model,
        # except when the additional bands are used to fill up the data to make the group bigger
        assert len(groups) >= len(unique), "Unitary representation allows less unique eigenvalues than needed to represent the data"
        # now try to match the data as good as possible
        model_bands = np.zeros(len(u_repr.U[0]), np.complex128)
        model_bands += np.min(k0_bands) - 0.01
        additional_left = len(model_bands) - len(k0_bands)
        # NOTE there is different options in the following loop.
        # TODO make a list of all the options to get better starting positions
        # TODO don't allow additional_bands in the middle of the bandstructure!
        if not use_repr_order:
            # for now just randomize the order
            for g in groups2:
                np.random.shuffle(g)

            for data_sym_size_i in range(0, dim):
                # start with small symmetries in the data and use up the additional_left in the most conservative way possible
                dsym = sym[data_sym_size_i]
                for s in dsym:
                    used = False
                    for model_sym_size_i in range(0, dim):
                        if model_sym_size_i - data_sym_size_i > additional_left:
                            raise ValueError(f"The given representation doesn't fit the data. There is not enough irreducible representations of dimension {data_sym_size_i} or smaller.")
                        msym = groups2[model_sym_size_i]
                        if len(msym) > 0:
                            g = msym[0]
                            model_bands[np.array(g)] = s
                            del msym[0]
                            additional_left -= max(0, model_sym_size_i - data_sym_size_i)
                            used = True
                            break
                    if not used:
                        # didn't find a symmetry to match
                        # TODO make the error more explicit (can this actually happen?)
                        raise ValueError("The given representation doesn't fit the data.")
        else:
            assert use_repr_order
            # TODO BIG PROBLEM!!! inversion symmetry sorting makes the order of unitary representations not choosable!
            # here assume that the order of the unitary representations matches the data (checking that)
            # then order the eigenvalues accordingly
            groups = np.array(groups)
            groups = groups[np.argsort(np.argmax(groups * np.arange(len(groups[0]), 0, -1)[None,:], axis=-1))]
            sorted_unique = sorted(list(unique))
            if verbose:
                print("ordered degeneracies:")
                print("data: ", [len([1 for b2 in k0_bands if b2 == value]) for value in sorted_unique])
                print("model:", [np.sum(np.array(g).astype(np.int32)) for g in groups])
            last_i = min(len(sorted_unique), len(groups)) - 1
            for i, (value, g) in enumerate(zip(sorted_unique, groups)):
                g = np.array(g)
                data_sym_size = len([1 for b2 in k0_bands if b2 == value])
                model_sym_size = np.sum(g.astype(np.int32))
                if i == 0 or i == last_i:
                    # TODO handle other offsets, such that there can be multiple additional bands below as well
                    # allow model_sym_size to be bigger than data_sym_size
                    if model_sym_size - data_sym_size > additional_left:
                        raise ValueError(f"The given representation doesn't fit the data. The additional bands are used up too early.")
                    additional_left -= max(0, model_sym_size - data_sym_size)
                else:
                    if data_sym_size != model_sym_size:
                        raise ValueError(f"The representations in the middle of the band structure must match the data, but there was a representation of size {model_sym_size} paired with data with degeneracy {data_sym_size}")
                model_bands[g] = value
        # now use up the rest for the remaining additional symmetries
        # these are redundant I think... so I'm printing it in verbose mode
        if additional_left > 0:
            if verbose:
                print(f"{additional_left} redundant bands found -> filling with minimum band value.")
            # it's already filled from the beginning
            pass
        # add random matrices for the k dependence to break the gradient descent subspace
        scale = (np.max(model_bands) - np.min(model_bands)) * 0.001 # TODO make this per band as otherwise it will become really large for far apart bands
        params = [np.diag(model_bands)] + [random_hermitian(len(model_bands)) * scale for _ in range(param_count-1)]
        tb = TightBindingModel(u_repr, neighbors, params)
        tb.normalize()
        #band_order = np.argsort(model_bands)
        return tb

    def dim(self):
        return len(self.sym.S[0]) # dimension from symmetry

    def f_i(self, k, i):
        assert i >= 0
        k = np.asarray(k)
        if i == 0:
            if self.sym.inversion:
                return np.ones_like(k[..., 0]), np.zeros_like(k[..., 0])
            else:
                return np.ones_like(k[..., 0])
        c = 0.0
        s = 0.0
        scale = 2*np.pi
        r_orig = self.neighbors[1 + ((i - 1) % (len(self.neighbors) - 1))]
        #for sym in self.sym.S:
        #    r = sym @ r_orig
        if True:
            r = r_orig
            t = scale * (k @ r)
            c += np.cos(t) - 1.0 # minus 1.0 to make params[0] the matrix at k=0, which can be set exact from the data
            s += np.sin(t)
        if self.sym.inversion:
            return c, s # return both, as they are both part of the operation
        else:
            return c if i < len(self.neighbors) else s

    # derivative of f_i
    def df_i(self, k, i):
        assert i >= 0
        k = np.asarray(k)
        d = np.zeros_like(k)
        if i == 0:
            if self.sym.inversion:
                return d, d
            else:
                return d
        c = d
        s = d.copy()
        scale = 2*np.pi
        r_orig = self.neighbors[1 + ((i - 1) % (len(self.neighbors) - 1))]
        #for sym in self.sym.S:
        #    r = sym @ r_orig
        if True:
            r = r_orig
            t = scale * (k @ r)
            c -= scale * np.sin(t)[...,None] * np.asarray(r)
            s += scale * np.cos(t)[...,None] * np.asarray(r)
        if self.sym.inversion:
            return c, s # return both, as they are both part of the operation
        else:
            return c if i < len(self.neighbors) else s
    
    # second derivative of f_i
    def ddf_i(self, k, i):
        assert i >= 0
        k = np.asarray(k)
        d = np.zeros(k.shape + (k.shape[-1],))
        if i == 0:
            if self.sym.inversion:
                return d, d
            else:
                return d
        c = d
        s = d.copy()
        scale = 2*np.pi
        r_orig = self.neighbors[1 + ((i - 1) % (len(self.neighbors) - 1))]
        #for sym in self.sym.S:
        #    r = sym @ r_orig
        if True:
            r = r_orig
            t = scale * (k @ r)
            r_sqr = scale**2 * (np.asarray(r)[:,None] * np.asarray(r)[None,:])
            c -= np.cos(t)[...,None,None] * r_sqr
            s -= np.sin(t)[...,None,None] * r_sqr
        if self.sym.inversion:
            return c, s # return both, as they are both part of the operation
        else:
            return c if i < len(self.neighbors) else s

    # full function for the band structure
    def f(self, k):
        mat = np.zeros(np.shape(k)[:-1] + self.params[0].shape, dtype=self.params.dtype)
        params = np.reshape(self.params, (len(self.params),) + (1,) * len(np.shape(k)[:-1]) + self.params[0].shape)
        if self.sym.inversion:
            n = self.unitary_repr.inv_split
            for i in range(len(self.params)):
                c, s = self.f_i(k, i)
                c = np.asarray(c)[...,None,None]
                s = np.asarray(s)[...,None,None]
                mat[:,:n,:n] += c * params[i][...,:n,:n]
                mat[:,n:,n:] += c * params[i][...,n:,n:]
                mat[:,:n,n:] += s * params[i][...,:n,n:]
                mat[:,n:,:n] += s * params[i][...,n:,:n]
        else:
            for i in range(len(self.params)):
                mat += np.asarray(self.f_i(k, i))[...,None,None] * params[i]
        return mat
    
    # derivative of f wrt k in the direction dk, outputshape (dim(k), N, N)
    def df(self, k):
        mat = np.zeros(np.shape(k) + self.params[0].shape, dtype=self.params.dtype)
        params = np.reshape(self.params, (len(self.params),) + (1,) * len(np.shape(k)) + self.params[0].shape)
        if self.sym.inversion:
            n = self.unitary_repr.inv_split
            for i in range(len(self.params)):
                c, s = self.df_i(k, i)
                c = np.asarray(c)[...,None,None]
                s = np.asarray(s)[...,None,None]
                mat[...,:n,:n] += c * params[i][...,:n,:n]
                mat[...,n:,n:] += c * params[i][...,n:,n:]
                mat[...,:n,n:] += s * params[i][...,:n,n:]
                mat[...,n:,:n] += s * params[i][...,n:,:n]
        else:
            for i in range(len(self.params)):
                mat += np.asarray(self.df_i(k, i))[...,None,None] * params[i]
        return mat
    
    # second derivative of f wrt k in the direction dk, outputshape (dim(k), dim(k), N, N)
    def ddf(self, k):
        mat = np.zeros(np.shape(k) + (np.shape(k)[-1],) + self.params[0].shape, dtype=self.params.dtype)
        params = np.reshape(self.params, (len(self.params),) + (1,) * (1+len(np.shape(k))) + self.params[0].shape)
        if self.sym.inversion:
            n = self.unitary_repr.inv_split
            for i in range(len(self.params)):
                c, s = self.ddf_i(k, i)
                c = np.asarray(c)[...,None,None]
                s = np.asarray(s)[...,None,None]
                mat[...,:n,:n] += c * params[i][...,:n,:n]
                mat[...,n:,n:] += c * params[i][...,n:,n:]
                mat[...,:n,n:] += s * params[i][...,:n,n:]
                mat[...,n:,:n] += s * params[i][...,n:,:n]
        else:
            for i in range(len(self.params)):
                mat += np.asarray(self.ddf_i(k, i))[...,None,None] * params[i]
        return mat

    def loss(self, k_smpl, ref_bands, weights, band_offset):
        bands = self.bands(k_smpl)
        return np.linalg.norm((bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands) * np.reshape(weights, (1, -1))) / len(k_smpl)**0.5

    def optimize(self, k_smpl, k_smpl_weights, ref_bands, weights, band_offset, iterations, batch_div=1, learning_rate=1.0, train_k0=True):
        N = np.shape(self.params)[1]
        assert band_offset >= 0 and band_offset <= N - len(ref_bands[0])
        # memoize self.f_i here using a rectangular matrix
        if self.sym.inversion:
            f_i_c = np.zeros((len(k_smpl), len(self.params)))
            f_i_s = np.zeros((len(k_smpl), len(self.params)))
            for i in range(len(self.params)):
                c, s = self.f_i(k_smpl, i)
                f_i_c[:, i] = c
                f_i_s[:, i] = s
            # special multiplication function for inversion symmetry
            def mul_cs(c, s, matrix):
                n = self.unitary_repr.inv_split
                mat = np.zeros(np.broadcast_shapes(np.shape(c[..., None, None]), np.shape(matrix[None, ..., :, :])), dtype=np.complex128)
                mat[..., :n,:n] = c[..., None, None] * matrix[None, ..., :n, :n]
                mat[..., n:,n:] = c[..., None, None] * matrix[None, ..., n:, n:]
                mat[..., :n,n:] = s[..., None, None] * matrix[None, ..., :n, n:]
                mat[..., n:,:n] = s[..., None, None] * matrix[None, ..., n:, :n]
                return mat
            def einsum_cs(c, s, diff):
                n = self.unitary_repr.inv_split
                mat = np.zeros(c.shape[1:2] + diff.shape[1:], dtype=np.complex128)
                mat[:,:n,:n] = np.einsum("bij,bn->nij", diff[:,:n,:n], c)
                mat[:,n:,n:] = np.einsum("bij,bn->nij", diff[:,n:,n:], c)
                mat[:,:n,n:] = np.einsum("bij,bn->nij", diff[:,:n,n:], s)
                mat[:,n:,:n] = np.einsum("bij,bn->nij", diff[:,n:,:n], s)
                return mat
        else:
            f_i_c = np.zeros((len(k_smpl), len(self.params)))
            for i in range(len(self.params)):
                f_i_c[:, i] = self.f_i(k_smpl, i)
            f_i_s = f_i_c
            # special multiplication function for inversion symmetry
            def mul_cs(c, _s, matrix):
                return c[..., None, None] * matrix[None, ..., :, :]
            def einsum_cs(c, _s, diff):
                return np.einsum("bij,bn->nij", diff, c * sc[:,None])
        self.params = self.symmetrizer(self.params)
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
                batch_f_i_c = f_i_c[iteration % batch_div::batch_div]
                batch_f_i_s = f_i_s[iteration % batch_div::batch_div]
                top_pad = np.shape(self.params)[1] - len(batch_ref[0]) - band_offset
                batch_weights = batch_weights / np.mean(batch_weights)

                # faster f using cached values
                f = mul_cs(batch_f_i_c, batch_f_i_s, self.params).sum(1)
                eigvals, eigvecs = np.linalg.eigh(f)
                #new_loss = np.linalg.norm(batch_ref - eigvals) / len(batch)**.5
                #last_loss = new_loss

                # TODO why does the following not work?
                #sc = (2.0 / len(batch)) / (np.abs(batch_f_i_c)**2).sum(1) # use f_i as weights, but normalize the whole step
                #ss = (2.0 / len(batch)) / (np.abs(batch_f_i_s)**2).sum(1) # TODO this normalization can be done as preprocessing
                ss = sc = (2.0 / len(batch)) / (np.abs(batch_f_i_c)**2 + np.abs(batch_f_i_s)**2).sum(1)
                diff = batch_ref - eigvals[:,band_offset:][:,:len(batch_ref[0])]
                diff *= weights
                diff = eigvecs[:,:,band_offset:N-top_pad] @ (diff[...,None] * np.swapaxes(np.conj(eigvecs)[:,:,band_offset:N-top_pad], 1, 2))
                params_add = einsum_cs(batch_f_i_c * sc[:,None], batch_f_i_s * ss[:,None], diff)
                params_add = self.symmetrizer(params_add)
                params_add *= learning_rate
                if not train_k0:
                    params_add[0] *= 0.0
                # impulse acceleration (beta given by the problem)
                params_add += last_add * (1 - 1 / len(batch))
                # change parameters (alpha = 1.0 with the chosen way to normalize the gradient)
                self.params += params_add
                last_add = params_add

                if iteration % 100 == 0:
                    #print(f"loss: {new_loss:.2e} alpha: {alpha:.1e}")
                    l = self.loss(k_smpl, ref_bands, weights, band_offset)
                    print(f"\rloss: {l:.2e}", end="")
        except KeyboardInterrupt:
            print("\naborted")
        self.normalize()
        l = self.loss(k_smpl, ref_bands, weights, band_offset)
        print(f"\rfinal loss: {l:.2e}")

    def normalize(self):
        # resymmetrize
        self.params = self.symmetrizer(self.params)
        if True:
            return
        # normalize, respecting the symmetry conditions (doesn't work yet...)
        #_, ev = np.linalg.eigh(self.f(((0,)*self.dim(),))) # this keeps the symmetry intact
        #ev = ev[0]
        _, ev = np.linalg.eigh(self.params[0]) # this keeps the symmetry intact
        # stable sort ev such that the 0 structure of params[0] is kept (important for symmetry)
        sorting = np.argsort(np.argmin(np.abs(ev) < 1e-7, axis=0))
        ev = ev.T[sorting].T
        for i in range(len(self.params)):
            self.params[i] = np.conj(ev.T) @ self.params[i] @ ev
        if True:
            return
        # normalize a little more using complex reflections on the second matrix (keeps inversion symmetry intact)
        if len(self.params) > 1:
            for i in range(1, len(self.params[1])):
                x = self.params[1][i-1, i]
                a = np.abs(x)
                if a != 0:
                    sign = np.conj(x) / a
                    self.params[:, :, i] *= sign
                    self.params[:, i, :] *= np.conj(sign)
                else:
                    pass # TODO switch to a different cell to normalize
        # normalize continuous DoF (TODO)

    # get the complex fourier coefficients H_r
    def params_complex(self):
        H = self.params / 2
        if self.sym.inversion:
            k = self.unitary_repr.inv_split
            H[:,:k,k:] *= 1j
            H[:,k:,:k] *= 1j
        else:
            H[1:] += H[(len(self.params)+1)//2:] * 1j
            H = H[:(len(self.params)+1)//2]
        return H
    
    # set params from the complex fourier coefficients H_r
    def set_from_complex(self, H_r):
        H = H_r * 2
        if self.sym.inversion:
            k = self.unitary_repr.inv_split
            H[:,:k,k:] *= -1j
            H[:,k:,:k] *= -1j
        else:
            H[1:] += H[(len(self.params)+1)//2:] * 1j
            H = H[:(len(self.params)+1)//2]
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
        grads = np.real(np.einsum("mji, mnjk, mki -> mni", np.conj(ev), df, ev))
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
        return TightBindingModel(self.f_i, self.df_i, self.params.copy())
    
    # make a tight binding model for a supercell with [nx, ny, nz] = repeats
    def copy_transformed(self, transform):
        # the repeated tight binding model has shifted versions of the bandstructure added on top
        # transform can be applied to the neighbors, but then there is neighbors
        # in the new unit cell, which need to be dealt with
        # TODO something something direct_sum
        return ...

    def plot_bands(self, k_smpl, *args, **kwargs):
        plot_bands_generic(k_smpl, self.bands(k_smpl), *args, **kwargs)
