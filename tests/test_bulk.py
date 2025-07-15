import numpy as np

from tight_binding_redweasel import (BandStructureModel, Symmetry, dos, bulk)

def test_1d_integration():
    T = [1, 10, 100, 300]
    beta = [1 / (dos.k_B * T_) for T_ in T]
    mu = [0]*4
    x = bulk._integration_points(0, min(T), max(T), 9)
    x, w = bulk._integration_weights(x, beta, mu, -100, 100, rtol=1e-7)
    #print(x)
    #print(w)
    # now all polynomials up to 8th order should be integrated exactly.
    correct_results = [1.0000000244300449, 1.0000024430045091, 1.0002443004509076, 1.0021987040581664]
    res = np.sum((x[None,:] - 1)**2 * w, axis=1)
    err = np.linalg.norm(res / correct_results - 1)
    res = list(res)
    assert err < 1e-7, f"incorrect result {res} instead of {correct_results} (error {err})"
    correct_results = [1.0000006840414388, 1.0000684058809426, 1.0068579756165343, 1.0629968759000317]
    res = np.sum((x[None,:] - 1)**8 * w, axis=1)
    err = np.linalg.norm(res / correct_results - 1)
    res = list(res)
    assert err < 2e-7, f"incorrect result {res} instead of {correct_results} (error {err})"
