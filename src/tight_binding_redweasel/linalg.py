import numpy as np
import scipy
from collections.abc import Callable


# direct sum of two matrices (block diagonal concatenation)
def _direct_sum(a, b):
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    assert len(a_shape) == len(b_shape), "No shape broadcasting support"
    d = len(a_shape)
    if d == 2:
        return np.block([[a, np.zeros((a_shape[0], b_shape[1]))], [np.zeros((b_shape[0], a_shape[1])), b]])
    # consider the first few dimensions to be listing dimensions and the last two to be the matrix dimensions.
    # NOTE: A more literal definition of direct sum for tensors would be to sum on every dimension!
    # However I only need matrices and this is my implementation, so I'm doing list of matrices.
    res = np.zeros(np.broadcast_shapes(a_shape[:-2], b_shape[:-2])
                   + (a_shape[-2] + b_shape[-2], a_shape[-1] + b_shape[-1]),
                   dtype=np.promote_types(a.dtype, b.dtype))
    res[...,:a_shape[-2],:a_shape[-1]] = a
    res[...,a_shape[-2]:,a_shape[-1]:] = b
    return res


def kron(*a):
    if len(a) == 0:
        raise ValueError("no parameters given")
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return np.kron(*a)
    return kron(np.kron(a[0], a[1]), *a[2:])


def direct_sum(*a):
    if len(a) == 0:
        raise ValueError("no parameters given")
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return _direct_sum(*a)
    return direct_sum(_direct_sum(a[0], a[1]), *a[2:])


def merge_axes(arr: np.ndarray, start_axis=-2, count=2) -> np.ndarray:
    """Merge a range of adjacent axes."""
    assert count >= 1, "the number of merged axes must be positive"
    shape = list(np.shape(arr))
    start_axis = start_axis if start_axis >= 0 else len(shape) + start_axis
    assert 0 <= start_axis < len(shape), f"axis {start_axis} is out of bounds for shape {shape}"
    shape[start_axis] = int(np.prod(shape[start_axis:start_axis+count]))
    del shape[start_axis+1:start_axis+count]
    return np.reshape(arr, shape)


def random_hermitian(n):
    h = (np.random.random((n, n)) * 2 - 1) + 1j * (2 * np.random.random((n, n)) - 1)
    return h + np.conj(h.T)


def random_unitary(n):
    return np.linalg.qr(np.random.standard_normal((n, n)) + np.random.standard_normal((n, n))*1j)[0]


def geigh(H: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(H.shape) >= 3:
        if np.linalg.norm(S - np.eye(S.shape[-1])) < 1e-8:
            # fast path
            return np.linalg.eigh(H)
        else:
            # TODO check if using np.linalg.cholesky to factor S and then np.linalg.eigh is faster
            res_la = np.zeros(H.shape[:-1], dtype=np.float64)
            res_ev = np.zeros(H.shape, dtype=H.dtype)
            for i in range(len(H)):
                la, ev = scipy.linalg.eigh(H[i], S[i])
                res_la[i] = la
                res_ev[i] = ev
            return np.array(res_la), np.array(res_ev)
    else:
        return scipy.linalg.eigh(H, S)


def geigvalsh(H: np.ndarray, S: np.ndarray) -> np.ndarray:
    if len(np.shape(H)) >= 3:
        res_la = np.zeros(H.shape[:-1], dtype=np.float64)
        for i in range(len(H)):
            res_la[i] = scipy.linalg.eigvalsh(H[i], S[i])
        return res_la
    else:
        return scipy.linalg.eigvalsh(H, S)

def geigh_grad(H: np.ndarray, S: np.ndarray, dH: np.ndarray, dS: np.ndarray):
    """Compute the generalized eigenvalues and eigenvectors using `geigh`.
    Then also compute the gradient w.r.t. a set of first order perturbation matrices.
    """
    if np.linalg.norm(S - np.eye(S.shape[-1])) < 1e-8 and np.linalg.norm(dS) < 1e-8:
        # fast path
        return eigh_grad(H, dH)
    bands, ev = geigh(H, S)
    path = ['einsum_path', (0, 1), (0, 1)] # precomputed path for best speed
    grads = np.real(np.einsum("...ji,...njk,...ki->...ni", np.conj(ev), dH, ev, optimize=path))
    # TODO check!
    grads -= np.real(np.einsum("...ji,...njk,...ki->...ni", np.conj(ev), dS, ev, optimize=path)) * bands[:,None,:]
    return bands, grads, ev

def eigh_grad(H: np.ndarray, dH: np.ndarray):
    """Compute the eigenvalues and eigenvectors using `np.linalg.eigh`.
    Then compute the gradient w.r.t. a set of first order perturbation matrices.

    NOTE: This method only works exactly for non degenerate matrices.
    Otherwise the gradients in eigensubspaces will be mixed.
    That doesn't lead to singularities, so it is ok to use it in an integration method,
    where the degenerate subspace has measure 0.

    Args:
        H (ndarray(..., N, N)): Hermitian matrix to diagonalize
        dH (ndarray(..., M, N, N)): H matrix derivatives w.r.t. M arbitrary variables.

    Returns:
        (tuple): (eigvals: ndarray(..., N), grads: ndarray(..., M, N), ev: ndarray(..., N, N))
    """
    eigvals, ev = np.linalg.eigh(H)
    path = ['einsum_path', (0, 1), (0, 1)] # precomputed path for best speed
    grads = np.real(np.einsum("...ji,...njk,...ki->...ni", np.conj(ev), dH, ev, optimize=path))
    return eigvals, grads, ev

def eigh_grad_hess(H: np.ndarray, dH: np.ndarray, ddH: np.ndarray, epsilon=1e-6):
    """Compute the eigenvalues and eigenvectors using `np.linalg.eigh`.
    Then compute the gradient w.r.t. a set of first order perturbation matrices.
    Then also compute the hessian w.r.t. the same set of variables with additional second order perturbation matrices.

    Args:
        H (ndarray(..., N, N)): Hermitian matrix to diagonalize
        dH (ndarray(..., M, N, N)): H matrix first order derivatives w.r.t. M arbitrary variables.
        ddH (ndarray(..., M, M, N, N)): H matrix second order derivatives w.r.t. M arbitrary variables. All combinations are required.
        epsilon (float, optional): tolerance threshold for degenerate bands. Choosing this too high will impact performance and precision (by "smoothing" the results like a finite difference quotient). Defaults to 1e-6.

    Returns:
        (tuple): (eigvals: ndarray(..., N), grads: ndarray(..., M, N), hessian: ndarray(..., M, M, N))
    """
    H_shape = np.shape(H)
    dH_shape = np.shape(dH)
    ddH_shape = np.shape(ddH)
    assert H_shape[-2:] == dH_shape[-2:] and dH_shape[-2:] == ddH_shape[-2:], "matrix size must be equal for all terms H, dH, ddH"
    assert H_shape[:-2] == dH_shape[:-3] and dH_shape[:-3] == ddH_shape[:-4], "enumerating dimensions must be equal for all terms H, dH, ddH"
    m = dH_shape[-3]
    assert m == ddH_shape[-3] and m == ddH_shape[-4], "number of variables M must be equal for the terms dH, ddH"
    eigvals, ev = np.linalg.eigh(H)
    # first order perturbation theory terms
    path = ['einsum_path', (0, 1), (0, 1)] # precomputed path for best speed
    #dH_ev = np.einsum("...ji,...njk,...kl->...nil", np.conj(ev), np.concatenate([dH, merge_axes(ddH, -4, 2)], axis=-3), ev, optimize=path)
    dH_ev = np.einsum("...ji,...njk,...kl->...nil", np.conj(ev), dH, ev, optimize=path)
    # hessian contribution from second derivative, first order perturbation theory
    path = ['einsum_path', (0, 2), (0, 1)] # precomputed path for best speed
    hess1 = np.real(np.einsum("...ji,...nmjk,...ki->...nmi", np.conj(ev), ddH, ev, optimize=path))
    # second order perturbation theory terms
    diff = eigvals[...,:,None] - eigvals[...,None,:]
    mask = np.abs(diff) < epsilon
    select = np.sum(np.triu(mask, k=1), axis=(-1, -2)) > 0
    if select.sum():
        # for this, an additional diagonalisation is needed for all matrices, which have degenerate eigenvalues.
        # To keep the efficiency high, only do the second diagonalisation for those which need it.
        # TODO select only works if select.shape != tuple()
        assert len(np.shape(dH_ev[select])) == len(dH_ev.shape)
        _, ev_dH = np.linalg.eigh(dH_ev[select] * mask[select][...,None,:,:])
        # stable sort ev such that the order of la doesn't change!
        sorting = np.argsort(np.argmin(np.abs(ev_dH) < 1e-7, axis=-2, keepdims=True), axis=-1)
        ev_dH = np.take_along_axis(ev_dH, sorting, axis=-1)
        # apply basis transformation to the selected dH_ev
        dH_ev[select] = np.einsum("...ji,...jk,...kl->...il", np.conj(ev_dH), dH_ev[select], ev_dH)
        #hess1[select] = np.einsum("...ji,...jk,...kl->...il", np.conj(ev_dH), hess1[select], ev_dH)
    # first order eigenvalue change (gradient, actually needs the basis change!)
    grads = np.real(np.diagonal(dH_ev, axis1=-2, axis2=-1))
    # assume that the perturbation term commutes with the hamiltonian.
    # otherwise it can cause non linear sqrt terms in the series development.
    inv_mask = ~mask
    dH_ev2 = dH_ev * inv_mask[...,None,:,:] # implicit copy
    dH_ev2 /= (diff + 1e-40)[...,None,:,:]
    # hessian contribution from first derivative, second order perturbation theory
    hess2 = np.real(np.einsum("...pik,...qki->...pqi", dH_ev, dH_ev2))
    return eigvals, grads, hess1 - 2*hess2


def pointcloud_distance(pointcloud1, pointcloud2):
    # TODO try using scipy.optimize.linear_sum_assignment
    pointcloud1 = np.asarray(pointcloud1)
    pointcloud1 = pointcloud1.reshape(pointcloud1.shape[0], -1)
    pointcloud2 = np.asarray(pointcloud2)
    pointcloud2 = pointcloud2.reshape(pointcloud2.shape[0], -1)
    # for each point in pointcloud1 find the closest in pointcloud2, add the distance, then remove that
    dist = 0.0
    for i, p1 in enumerate(pointcloud1):
        d = np.linalg.norm(p1 - pointcloud2, axis=-1)
        min_index = np.argmin(d.flat)
        dist += d.flat[min_index]
        d = np.delete(d.flat, min_index)
    return dist


def lattice_in_sphere(A, dist) -> np.ndarray:
    # first make A as orthogonal as possible using QR decomposition
    for _ in range(3):
        Q, R = np.linalg.qr(A)
        R /= np.diag(R)[:,None]
        R_inv = np.linalg.inv(R)
        assert abs(abs(np.linalg.det(R_inv)) - 1.0) < 1e-7
        R_inv = np.round(R)
        assert abs(abs(np.linalg.det(R_inv)) - 1.0) < 1e-7
        A = A @ R_inv
    # TODO use a supercell like fcc or bcc to make the following more efficient
    # now figure out how many lines per direction are needed
    U, S, Vh = np.linalg.svd(A)
    n = len(S)
    # S defines an axis aligned bounding box. That then get's rotated by Vh.
    # The new bounding box can be computed using the mins and max of the resulting 2^n corner vectors.
    # That means this algorithms is exponential in the number of dimensions,
    # however remember that the number of points is also exponential in the dimension.
    corners = (Vh.T @ (np.stack(np.meshgrid(*n*((-1, 1),)), axis=-1).reshape(-1, n) / S).T).T
    # now create the ranges and from there the whole grid
    ranges = [np.arange(-int(dist*np.max(corners[:,i])), 1+int(dist*np.max(corners[:,i]))) for i in range(n)]
    pos = S * (Vh @ np.stack(np.meshgrid(*ranges), axis=-1).reshape(-1, n).T).T
    # filter the result to only include positions in the sphere
    pos = pos[np.linalg.norm(pos, axis=-1) < dist]
    # finish the transformation and copy the array to get a continuous array.
    return np.array((U @ pos.T).T)


# A is a linear symmetric python function
# b is of the same dimension as A(x0)
# apart from solving a normal linear eq, it can also be used
# to calculate the pseudo inverse of A efficiently like this:
# A_pinv = conjugate_gradient_solve(lambda x: A @ x, np.identity(4), np.diag(1/(np.diag(A) + 1e-12)))
def conjugate_gradient_solve(A: Callable[[np.ndarray], np.ndarray], b: np.ndarray, x0=None, err=1e-9, max_i=None) -> np.ndarray:
    if x0 is None:
        x = np.zeros_like(b)
        r = -b
    else:
        x = x0
        r = A(x) - b
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