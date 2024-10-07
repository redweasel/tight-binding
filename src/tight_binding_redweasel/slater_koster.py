import numpy as np
from . import bandstructure as _bands
from . import symmetry as _sym
from . import hamiltonian_symmetry as _hsym
from . import kpaths as _kpaths
from typing import Callable, Tuple

# This is an implementation of a pure Slater-Koster
# tight-binding interpolation model.

# The implementation just considers the special cases from the Slater and Koster paper.
# The core are functions to convert from matrices to Slater and Koster parameters and back.

# TODO implement this stuff to test it against the other models.

def to_fcc(H_r, neighbors):
    # normalize the H_r first, such that H_0 is diagonal
    return ...

class SlaterKoster:
    """A simple class to make a Slater-Koster type tight-binding model.
    This class is used to extract the independent parameters in the tight-binding model
    that is restricted by a HamiltonianSymmetry and given neighbors.
    It can then also be used to set those parameters. For anything more advanced,
    refer to self.model, which is a BandStructureModel.
    """

    def __init__(self, hsym: _hsym.HamiltonianSymmetry, neighbors):
        self.neighbors = neighbors
        self.hsym = hsym
        # generate the symmetrizer
        self.symmetrizer = hsym.symmetrizer(neighbors)
        # use BandstructureModel
        self.model = _bands.BandStructureModel.init_tight_binding(_sym.Symmetry.none(self.hsym.sym.dim()), self.neighbors, self.hsym.dim(), cos_reduced=False, exp=True)
        self.model.symmetrizer = self.symmetrizer
    
    def parameters(self, real=False) -> dict:
        """
        Figure out all the independent parameters.

        Returns:
            (dict): A dictionary with keys (names generated from hsym names) and tuples (k, i, j, fac) as values.
                The values are the index in the params matrices.
                "fac" is the factor by which to multiply the entry before symmetrisation.
        """
        # slow but complete
        # go through every index k, i>j which has not yet been set to a value.
        found_params = dict()
        dummy_params = np.zeros(np.shape(self.model.params), dtype=complex)
        n = self.hsym.dim()
        init_value = 1.0 if real else 1.0 + 1.0j
        for k in range(len(self.neighbors)):
            for i in range(n):
                for j in range(i, n):
                    if np.abs(dummy_params[k][i, j]) > 1e-5:
                        continue
                    dummy_params[k][i, j] = init_value
                    dummy_params = self.symmetrizer(dummy_params)
                    if np.abs(dummy_params[k][i, j]) <= 1e-5:
                        continue
                    # found a new parameter
                    # -> name it first
                    label = "E" if k == 0 else f"V{k}"
                    label = label + self.hsym.get_band_name(i)
                    label = label + self.hsym.get_band_name(j)
                    # figure out the correct factor
                    if np.abs(np.imag(dummy_params[k][i, j])) < 1e-5:
                        # fac is float
                        fac = float(np.real(dummy_params[k][i, j]))
                    elif np.abs(np.real(dummy_params[k][i, j])) < 1e-5:
                        # fac is float
                        fac = float(np.imag(dummy_params[k][i, j]))
                    else:
                        # imaginary values allowed -> fac is complex
                        fac = complex(dummy_params[k][i, j] / init_value)
                    # now add it to the dictionary
                    # TODO this is a bit messy with the complex values...
                    # -> improve it once I know what I want here.
                    found_params[label] = (k, i, j, fac)
        return found_params

    def set_parameter(self, parameter_tuple: tuple, value: complex):
        """Takes one of the tuple values from `self.parameters()`
        and sets the corresponding parameter to a given value.

        Args:
            parameter_tuple (tuple): Tuple from `self.parameters`
            value (float, complex): value to be inserted for the specified parameter.
        """
        k, i, j, fac = parameter_tuple
        self.model.params[k][i, j] = value / fac
        self.model.params = self.symmetrizer(self.model.params)

    def interactive_widget(self, kpath: _kpaths.KPath, ylim: Tuple[float, float], ref_model: Callable=None, slider_range=5.0, slider_step=0.1, latex=False, use_complex=False):
        from matplotlib import pyplot as plt
        import ipywidgets as widgets
        import re

        params_dict = self.parameters()
        if use_complex:
            params_dict = params_dict | {"i"+name: (k, i, j, fac*1j) for name, (k, i, j, fac) in params_dict.items() if type(fac) == complex}

        def update(**kwargs):
            self.model.params[:] = 0.0
            for name, (k, i, j, fac) in params_dict.items():
                # don't use set_parameter, symmetrize just once at the end
                # self.set_parameter((k, i, j, fac), kwargs[name])
                self.model.params[k][i, j] += kwargs[name] / fac
            self.model.params = self.symmetrizer(self.model.params)

            kpath.plot(self.model, ylim=ylim)
            if ref_model is not None:
                kpath.plot(ref_model, '--', label_bands="left", ylim=ylim)
            plt.show()

        def desc(name):
            if latex:
                return "$"+re.sub(r"(?<=[^0-9])([0-9]+)", r"_{\1}", name)+"$"
            else:
                for big, small in {'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉'}.items():
                    name = name.replace(big, small)
                return name

        def value_readout(k, i, j, fac):
            return np.real(self.model.params[k, i, j]) if abs(np.imag(fac)) < 1e-5 else np.imag(self.model.params[k, i, j])

        # interact, but with sliders on the left instead of above
        sliders = { name: widgets.FloatSlider(value=value_readout(*value), min=-slider_range, max=slider_range, step=slider_step, description=desc(name)) for name, value in params_dict.items() }
        return widgets.HBox([
            widgets.VBox(list(sliders.values())),
            widgets.interactive_output(update, sliders),
        ])