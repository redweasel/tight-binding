import numpy as np
from . import bandstructure as _bands
from . import symmetry as _sym
from . import hamiltonian_symmetry as _hsym
from . import kpaths as _kpaths
from . import linalg as _lin
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
    
    @staticmethod
    def from_orbitals(A, sym: _sym.Symmetry, basis, orbital_integrals: dict, max_distance):
        """Initialize the Hamiltonian from Orbital Integrals.
        This allows s, p and d orbitals to be used with a given basis.
        The Lattice is arbitrary. The functions for the orbitals integrals
        have to handle the necessary cases though.

        Note, the HamilonianSymmetry does currently not support translation symmetries.

        Args:
            A (arraylike(dim, dim)): The real lattice vectors (as columns of the matrix).
            sym (Symmetry): The reciprocal space symmetry associated with the bandstructure.
            basis (list): Atoms per cell like '[("Si", "sp", (0, 0, 0)), ("Si", "sp", (0.5, 0.5, 0.5))]'.
                This is a list of tuples with (name of atom, orbitals, crystal space position).
            orbital_integrals (dict): The dictionary of orbital integrals between the atom sites.
                It has the format `{"Si/Si:V_s_s": func, "Si/Si:V_py_px": func2}` where `func` is a function that takes a 3D offset vector
                (works also in lower dimensions) and returns the complex value of the integral. Usually a lot of these functions are the same.
                For parameters which just depend on the distance, where all values are tabellated, a list can be given instead of a function.
                The index in the list is then the n-th distance from the list of all possible distances in the lattice.
            max_distance (float): The real space distance of the neighbor term. This applied to the neighbors computed by A @ n.
        """
        # TODO document d orbitals
        assert sym.is_orthogonal(), "only orthogonal symmetries allowed. Remember this is the symmetry of the reciprocal space, not the reciprocal crystal-unit space."
        # find all lattice points in a sphere.
        neighbors = _lin.lattice_in_sphere(A, max_distance)
        # check what types of values are given in orbital_integrals
        use_distances = False
        use_symmetries = False
        for _, func in orbital_integrals.items():
            if type(func) is list:
                use_distances = True
            elif type(func) is dict:
                use_symmetries = True
        if use_distances:
            # find the list of all possible atom distances
            distances = set()
            for n in neighbors:
                for _, _, pos1 in basis:
                    for _, _, pos2 in basis:
                        # TODO improve... this makes choices of A with small determinant not work!
                        d = np.linalg.norm(n + A @ (np.array(pos1) - np.array(pos2)))
                        if d <= max_distance:
                            distances.add(np.round(d, 5))
            distances = sorted(list(distances))
            def index_from_r(r):
                r = np.round(r, 5)
                if r <= max_distance:
                    return distances.index(r)
                else:
                    return 100
        
        dim = len(A)
        # TODO get rid of the warning for duplicate k points...
        neighbors = _sym.Symmetry.inv(dim).complete_neighbors(neighbors)
        # now create the parameters
        band_count = sum((sum(({'s': 1, 'p': dim, 'd': [1,2,5][dim-1]}[letter] for letter in orbitals)) for _, orbitals, _ in basis))
        params = np.zeros((len(neighbors), band_count, band_count), dtype=np.complex128)
        suborbitals = {'s': ['s'], 'p': ['px','py','pz'][:dim], 'd': ['dyz','dxz','dxy','dx2y2','dz'][:[1,2,5][dim-1]]}
        k1 = 0
        for name1, orbitals1, pos1 in basis:
            pos1 = np.array(pos1)
            for orbital1 in orbitals1:
                for sub1 in suborbitals[orbital1]:
                    k2 = 0
                    for name2, orbitals2, pos2 in basis:
                        pos2 = np.array(pos2)
                        for orbital2 in orbitals2:
                            for sub2 in suborbitals[orbital2]:
                                # generate the name of the orbital integral (try all permutations)
                                orbital_integral_name11 = f"{name1}/{name2}:V_{sub1}_{sub2}"
                                orbital_integral_name21 = f"{name2}/{name1}:V_{sub1}_{sub2}"
                                # switched orbitals mean complex conjugate of the result!
                                orbital_integral_name12 = f"{name1}/{name2}:V_{sub2}_{sub1}"
                                orbital_integral_name22 = f"{name2}/{name1}:V_{sub2}_{sub1}"
                                # find the function to use
                                func = None
                                for orbital_integral_name, swap in [(orbital_integral_name11, False), (orbital_integral_name21, False), (orbital_integral_name12, True), (orbital_integral_name22, True)]:
                                    if orbital_integral_name in orbital_integrals:
                                        func = orbital_integrals[orbital_integral_name]
                                        break
                                assert func is not None, f"Orbital function {orbital_integral_name11} is missing"
                                inv = {'s':1,'p':-1,'d':1}[orbital1] * {'s':1,'p':-1,'d':1}[orbital2]
                                if type(func) is list:
                                    assert inv == 1, f"can't specify the orbital integral of {orbital1}{orbital2} just by distance"
                                elif type(func) is dict:
                                    # here func has x-offsets and values.
                                    # However the dict is reduced by symmetry,
                                    # so that needs to be realized here
                                    axes = set()
                                    axes.add({'s':-1,'px':0,'py':1,'pz':2,'dyz':0,'dxz':1,'dxy':2}[sub1])
                                    axes.add({'s':-1,'px':0,'py':1,'pz':2,'dyz':0,'dxz':1,'dxy':2}[sub2])
                                    if -1 in axes:
                                        axes.remove(-1)
                                    axes = list(axes)
                                    proj = np.zeros(dim)
                                    proj[axes] = 1
                                    proj = np.diag(proj)
                                for i1, n1 in enumerate(neighbors):
                                    # TODO how does this work? Is this correct? Check!
                                    x = n1 + A @ (pos1 - pos2)
                                    if type(func) is list:
                                        index = index_from_r(np.linalg.norm(x))
                                        p = func[index] if index < len(func) else 0.0
                                    elif type(func) is dict:
                                        p = 0.0 # if no value is found
                                        r = np.linalg.norm(x)
                                        for pos, value in func.items():
                                            rpos = np.linalg.norm(pos)
                                            # TODO these thresholds assume A is normalized...
                                            if abs(rpos - r) < 1e-4:
                                                if np.linalg.norm(proj @ (pos - x)) < 1e-4:
                                                    p = value
                                                    break
                                                elif np.linalg.norm(proj @ (pos + x)) < 1e-4:
                                                    p = np.conj(value) * inv
                                                    break
                                                if inv == 1 and len(axes) == 2:
                                                    # use more symmetries... px py has an additional symmetry with a negative sign!
                                                    proj2 = np.array(proj)
                                                    proj2[axes[0], axes[0]] = -1
                                                    if np.linalg.norm(proj @ pos - proj2 @ x) < 1e-4:
                                                        p = -np.conj(value)
                                                        break
                                                    elif np.linalg.norm(proj @ pos + proj2 @ x) < 1e-4:
                                                        p = -np.conj(value)
                                                        break
                                    else:
                                        p = func(x)
                                    params[i1, k1, k2] = np.conj(p)*inv if swap else p
                                k2 += 1
                    k1 += 1
        # for the slater koster model, we also need a HamiltonianSymmetry.
        # To build this, we need the unitary representations of s, p and d.
        hsym = _hsym.HamiltonianSymmetry(sym)
        for name, orbitals, pos in basis:
            for letter in orbitals:
                if letter == 's':
                    hsym.append_s(pos, name)
                elif letter == 'p':
                    hsym.append_p(pos, name)
                elif letter == 'd':
                    hsym.append_d3(pos, name)
                    if dim >= 3:
                        hsym.append_d2(pos, name)
        sk = SlaterKoster(hsym, neighbors)
        sk.model.params[:] = params
        # TODO check symmetry of params (of model.f()) here!
        return sk

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