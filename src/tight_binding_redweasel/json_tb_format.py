import numpy as np
import json

# This file makes an interface for my calculations with
# my Rust code which uses serde to serialize its (ndarray based) models.
# However it is also useful as a general purpose exchange format for
# tight binding models as it is really easy to implement in most languages using a json package.

def load(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    band_count = data["band_count"]
    neighbors = np.array(data["neighbors"])
    params = np.array([h["data"] for h in data["params"]]).reshape(len(neighbors), band_count, band_count, -1)
    if params.shape[3] == 2:
        # complex valued
        params = params[...,0] + params[...,1]*1j
    else:
        assert params.shape[3] == 1
        # real valued (this format isn't used, but could be)
        params = params[...,0]
    order = np.argsort(np.linalg.norm(neighbors, axis=-1))
    return neighbors[order], params[order]

def save(filename, neighbors, params):
    neighbors = np.asarray(neighbors, dtype=np.float64)
    # build the correct format from 
    # {"band_count":12,"params":[{"v":1,"dim":[12,12],"data":[[1.0,1.0],...]},...],"neighbors":[[0.0,0.0,0.0],...]}
    with open(filename, "w") as file:
        band_count = len(params[0])
        # round the numbers to 0 if they are VERY close
        params = np.where(np.abs(np.real(params)) < 1e-14, 1j*np.imag(params), params)
        params = np.where(np.abs(np.imag(params)) < 1e-14, np.real(params), params)
        # save them in the json format
        params_repr = [{"v": 1, "dim": [band_count, band_count], "data": [[np.real(v), np.imag(v)] for v in h.flat]} for h in params]
        json.dump({"band_count": band_count, "params": params_repr, "neighbors": [tuple(n) for n in neighbors]}, file)

