import numpy as np
from .bandstructure import *
from .symmetry import *

# This is an implementation of a pure Slater-Koster
# tight-binding interpolation model.

# The implementation just considers the special cases from the Slater and Koster paper.
# The core are functions to convert from matrices to Slater and Koster parameters and back.

# TODO implement this stuff to test it against the other models.

def to_fcc(H_r, neighbors):
    # normalize the H_r first, such that H_0 is diagonal
    return ...

class SlaterKoster:
    def __init__(self, model):
        assert model in ["sc", "fcc", "bcc", "diamond"]
        cell_matrix = np.asarray(cell_matrix)
        # generate neighbors
        self.neighbors = []
        # all of the examples are currently cubic symmery
        sym = Symmetry.cubic(model != "diamond")
        # use BandstructureModel
        self.model = BandStructureModel.init_tight_binding(sym, self.neighbors, )
    
    def symmetrizer(self):
        """
        Returns:
            (function): A symmetrizer that covers the entire symmetry of this model.
        """
        # TODO


