import numpy as np
from .bandstructure import *
from .symmetry import *

# This is an implementation of a pure Slater-Koster
# tight-binding interpolation model, fit using gradient descent.
# The implementation just gives a symmetrizer for the bandstructure model.

# The implementation just considers the special cases from the Slater and Koster paper.
# The core are functions to convert from matrices to Slater and Koster parameters and back.

def to_fcc(H_r, neighbors):
    # normalize the H_r first, such that H_0 is diagonal
    return ...

class SlaterKoster:
    def __init__(self, model):
        assert model in ["sc", "fcc", "bcc", "diamond"]
        cell_matrix = np.asarray(cell_matrix)
        # generate neighbors
        self.neighbors = []
        # use BandstructureModel
        self.model = BandStructureModel.init_tight_binding(sym, self.neighbors, )
    
    def symmetrizer(self):
        pass


