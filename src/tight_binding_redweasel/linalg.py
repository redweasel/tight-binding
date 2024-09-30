import numpy as np
import scipy
from typing import Tuple, Callable


# direct sum of two matrices (block diagonal concatenation)
def direct_sum2(a, b):
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
        return direct_sum2(*a)
    return direct_sum(direct_sum2(a[0], a[1]), *a[2:])


def random_hermitian(n):
    h = (np.random.random((n, n)) * 2 - 1) + 1j * (2 * np.random.random((n, n)) - 1)
    return h + np.conj(h.T)


def random_unitary(n):
    return np.linalg.qr(np.random.standard_normal((n, n)) + np.random.standard_normal((n, n))*1j)[0]


def geigh(H: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(H.shape) == 3:
        if np.linalg.norm(S - np.eye(S.shape[-1])) < 1e-8:
            # fast path
            return np.linalg.eigh(H)
        else:
            # TODO check if using np.linalg.cholesky to factor S and then np.linalg.eigh is faster
            res_la = np.zeros(H.shape[:2], dtype=H.dtype)
            res_ev = np.zeros(H.shape, dtype=H.dtype)
            for i in range(len(H)):
                la, ev = scipy.linalg.eigh(H[i], S[i])
                res_la.append(la)
                res_ev.append(ev)
            return np.array(res_la), np.array(res_ev)
    else:
        return scipy.linalg.eigh(H, S)


def geigvalsh(H: np.ndarray, S: np.ndarray) -> np.ndarray:
    if len(np.shape(H)) == 3:
        res_la = []
        for i in range(len(H)):
            la = scipy.linalg.eigvalsh(H[i], S[i])
            res_la.append(la)
        return np.array(res_la)
    else:
        return scipy.linalg.eigvalsh(H, S)


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