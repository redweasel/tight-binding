# This file contains general implementations for loss functions for interpolation models.
import numpy as np
import scipy.optimize

def model_error(bands, ref_bands, band_weights, band_offset) -> tuple[float, np.ndarray]:
    """
    Returns:
        (float, ndarray(N_b)): the weighted loss (standard deviation) and the maximal error per band
    """
    assert len(bands) == len(ref_bands)
    assert len(band_weights) == len(ref_bands[0])
    assert len(np.shape(ref_bands)) == 2
    assert len(np.shape(bands)) == 2
    err = (bands[:, band_offset:][:, :len(ref_bands[0])] - ref_bands)
    max_err = np.max(np.abs(err), axis=0)
    err *= np.reshape(band_weights, (1, -1))
    return np.linalg.norm(err) / len(bands)**0.5, max_err

def model_loss(bands, ref_bands, band_weights, band_offset) -> float:
    """
    Returns:
        float: the weighted loss (standard deviation)
    """
    assert len(bands) == len(ref_bands)
    assert len(band_weights) == len(ref_bands[0])
    assert len(np.shape(ref_bands)) == 2
    assert len(np.shape(bands)) == 2
    err = (bands[:, band_offset:][:, :len(ref_bands[0])] -
            ref_bands) * np.reshape(band_weights, (1, -1))
    return np.linalg.norm(err) / len(bands)**0.5
    # return np.max(np.abs(bands[:,band_offset:][:,:len(ref_bands[0])] - ref_bands))

def model_windowed_loss(bands, ref_bands, min_energy, max_energy, allow_skipped_bands=False) -> float:
    """Compute the mean squared error for a model, but only consider energies in an energy window.
    Additionally, there is more freedom for band assignment. For the most general version set
    `allow_skipped_bands=True`, otherwise the bands are assumed to be sorted without skipped bands.

    Args:
        model (_type_): _description_
        k_smpl (_type_): _description_
        ref_bands (_type_): _description_
    """
    assert len(bands) == len(ref_bands)
    assert len(np.shape(ref_bands)) == 2
    assert len(np.shape(bands)) == 2
    if allow_skipped_bands:
        cost = 0
        for i in range(len(bands)):
            select = (min_energy <= ref_bands[i]) & (ref_bands[i] <= max_energy)
            cost_matrix = (ref_bands[i][select,None] - bands[i])**2
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            cost += cost_matrix[row_ind, col_ind].sum()
        return (cost / len(bands))**.5
    else:
        compare_bands = min(len(bands[0]), len(ref_bands[0]))
        max_offset = max(len(bands[0]), len(ref_bands[0])) - compare_bands
        select = (min_energy <= ref_bands) & (ref_bands <= max_energy)
        cost = float("inf")
        # find out bandoffset by checking all bandoffsets and finding the one with the lowest error
        for i in range(max_offset):
            if len(bands[0]) < len(ref_bands[0]):
                diff = (ref_bands[:,i:i+compare_bands] - bands) * select[:,i:i+compare_bands]
            else:
                diff = (ref_bands - bands[:,i:i+compare_bands]) * select
            cost = min(cost, np.linalg.norm(diff))
        return cost / len(bands)**.5

def model_error_hist(bands, ref_bands, N=50, allow_skipped_bands=False):
    """Compute a histogram showing the maximal errors of a model over the energy range.

    Args:
        bands (_type_): _description_
        ref_bands (_type_): _description_
        N (int, optional): _description_. Defaults to 50.

    Returns:
        (ndarray[N], ndarray[N]): bin centers, maximal error in the bin.
    """
    assert len(bands) == len(ref_bands)
    assert len(np.shape(ref_bands)) == 2
    assert len(np.shape(bands)) == 2
    bin_centers = np.linspace(np.maximum(np.min(ref_bands), np.min(bands)), np.minimum(np.max(ref_bands), np.max(bands)), N)
    h = bin_centers[1] - bin_centers[0]
    if allow_skipped_bands:
        error = []
        ref_bands2 = []
        for i in range(len(ref_bands)):
            cost_matrix = np.abs(ref_bands[i][:,None] - bands[i])
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            error.append(cost_matrix[row_ind, col_ind])
            ref_bands2.append(ref_bands[i][row_ind])
        error = np.array(error)
        ref_bands = np.array(ref_bands2)
    else:
        # find out bandoffset by checking all bandoffsets and finding the one with the lowest error
        compare_bands = min(len(bands[0]), len(ref_bands[0]))
        max_offset = max(len(bands[0]), len(ref_bands[0])) - compare_bands
        error = None
        ref_bands2 = ref_bands
        cost = float("inf")
        for i in range(max_offset):
            if len(bands[0]) < len(ref_bands[0]):
                diff = (ref_bands[:,i:i+compare_bands] - bands)
            else:
                diff = (ref_bands - bands[:,i:i+compare_bands])
            c = np.linalg.norm(diff)
            if c < cost:
                cost = c
                error = np.abs(diff)
                if len(bands[0]) < len(ref_bands[0]):
                    ref_bands2 = ref_bands[:,i:i+compare_bands]
        ref_bands = ref_bands2
    return bin_centers, np.array([np.max(list(error[np.abs(ref_bands - e) <= h/2])+[0]) for e in bin_centers])