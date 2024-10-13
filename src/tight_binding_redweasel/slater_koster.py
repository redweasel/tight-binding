import numpy as np
from . import bandstructure as _bands
from . import symmetry as _sym
from . import hamiltonian_symmetry as _hsym
from . import kpaths as _kpaths
from . import unitary_representations as _urepr
from . import linalg as _lin
from typing import Callable, Tuple, Self

# This is an implementation of a pure Slater-Koster
# tight-binding interpolation model.

# The implementation just considers the special cases from the Slater and Koster paper.
# The core are functions to convert from matrices to Slater and Koster parameters and back.

# TODO implement this stuff to test it against the other models.

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
    def from_orbitals(A, sym: _sym.Symmetry, basis, orbital_integrals: dict, max_distance) -> Self:
        """Initialize the Hamiltonian from Orbital Integrals.
        This allows s, p and d orbitals to be used with a given basis.
        The Lattice is arbitrary. The functions for the orbitals integrals
        have to handle the necessary cases though.

        Note, the HamilonianSymmetry does currently not support translation symmetries.

        Args:
            A (arraylike(dim, dim)): The real lattice vectors (as columns of the matrix). It's best to use a normalized/scaled version here,
                as many tolerance checks are fixed to work for values around 1.0.
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
        # TODO check if the symmetry is projective and check if it matches the basis!
        assert sym.is_orthogonal(), "only orthogonal symmetries allowed. Remember this is the symmetry of the reciprocal space, not the reciprocal crystal-unit space."
        # -> sym = sym.dual()
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
        neighbors = _sym.Symmetry.none(dim).complete_neighbors(neighbors)
        # now create the parameters
        band_count = sum((sum(({'s': 1, 'p': dim, 'd': [1,2,5][dim-1]}[letter] for letter in orbitals)) for _, orbitals, _ in basis))
        params = np.zeros((len(neighbors), band_count, band_count), dtype=np.complex128)
        suborbitals = {'s': ['s'], 'p': ['px','py','pz'][:dim], 'd': ['dyz','dxz','dxy','dx2y2','dz'][:[1,2,5][dim-1]]}
        urepr = {'s': _urepr.UnitaryRepresentation(sym, 1)}
        urepr['p'] = _urepr.UnitaryRepresentation(sym, 3)
        urepr['p'].U = sym.S
        urepr['p'].inv_split = 0
        urepr['d'] = _urepr.UnitaryRepresentation(sym, 3)
        urepr['d'].U = sym.S
        urepr['d'].inv_split = 3
        # TODO make the following work for all symmetries...
        urepr['d'] = urepr['d'] + _urepr.UnitaryRepresentation.d3(False, inversion=sym.inversion)
        inv_urepr = {'s': 1, 'p': -1, 'd': 1, 'f': -1}

        # fill up orbital_integrals
        for name1, orbitals1, pos1 in basis:
            pos1 = np.array(pos1)
            for orbital1 in orbitals1:
                for sub1_i, sub1 in enumerate(suborbitals[orbital1]):
                    for name2, orbitals2, pos2 in basis:
                        pos2 = np.array(pos2)
                        for orbital2 in orbitals2:
                            inv = inv_urepr[orbital1] * inv_urepr[orbital2]
                            for sub2_i, sub2 in enumerate(suborbitals[orbital2]):
                                func = None
                                # extend search to symmetric orbitals
                                for s_i, (s_op, urepr1, urepr2) in enumerate(zip(sym.S, urepr[orbital1].U, urepr[orbital2].U)):
                                    #assert abs(np.linalg.det(s_op) - 1) < 1e-5
                                    #if np.any(s_op < 0): # TODO prefer these by sorting the loop
                                    #    continue
                                    suborb1_rot = np.array(urepr1[:,sub1_i])
                                    suborb2_rot = np.array(urepr2[:,sub2_i])
                                    sub1_rot = np.argmax(np.abs(suborb1_rot))
                                    sub2_rot = np.argmax(np.abs(suborb2_rot))
                                    # first extract the value before setting it to 0
                                    s_inv = suborb1_rot[sub1_rot] * suborb2_rot[sub2_rot]
                                    # TODO the following is what I assume works, but probably doesn't...
                                    suborb1_rot[sub1_rot] = 0
                                    if np.linalg.norm(suborb1_rot) > 1e-5:
                                        continue
                                    #assert np.linalg.norm(suborb1_rot) < 1e-5, f"{suborb1_rot}"
                                    suborb2_rot[sub2_rot] = 0
                                    if np.linalg.norm(suborb2_rot) > 1e-5:
                                        continue
                                    #assert np.linalg.norm(suborb2_rot) < 1e-5, f"{suborb2_rot}"
                                    # get the names of the orbitals
                                    sub1_rot = suborbitals[orbital1][sub1_rot]
                                    sub2_rot = suborbitals[orbital2][sub2_rot]
                                    # generate the name of the orbital integral (try all permutations)
                                    for names in [f"{name1}/{name2}", f"{name2}/{name1}"]:
                                        for orbital_integral_name, swap in [(f"{names}:V_{sub1_rot}_{sub2_rot}", False), (f"{names}:V_{sub2_rot}_{sub1_rot}", True)]:
                                            if orbital_integral_name in orbital_integrals:
                                                res_inv = inv * s_inv if swap else s_inv
                                                func = orbital_integrals[orbital_integral_name]
                                                if s_i > 0:
                                                    # set func to the transformed value
                                                    if type(func) == list:
                                                        func = [res_inv * v for v in func]
                                                    elif type(func) == dict:
                                                        func = {tuple(s_op.T @ pos): res_inv * v for pos, v in func.items()}
                                                    else:
                                                        # package "func" inside a function to build the lambda expression independend of the local variable with name "func"
                                                        def pack(func):
                                                            return lambda x: res_inv * func(s_op @ x)
                                                        func = pack(func)
                                                    # save the transformed func in orbital_integrals
                                                    # (no copy of the dict -> the instance of the input dict will get extended)
                                                    orbital_integrals[f"{name1}/{name2}:V_{sub1}_{sub2}"] = func
                                                break
                                        if func is not None: break
                                    if func is not None: break
                                    # TODO keep searching and if there is a second func found, that doesn't match, report that as an error
                                assert func is not None, f"Orbital function {name1}/{name2}:V_{sub1}_{sub2} is missing"

        # now compute the parameter matrices H_r
        k1 = 0
        for name1, orbitals1, pos1 in basis:
            pos1 = np.array(pos1)
            for orbital1 in orbitals1:
                for sub1_i, sub1 in enumerate(suborbitals[orbital1]):
                    k2 = 0
                    for name2, orbitals2, pos2 in basis:
                        pos2 = np.array(pos2)
                        for orbital2 in orbitals2:
                            inv = inv_urepr[orbital1] * inv_urepr[orbital2]
                            for sub2_i, sub2 in enumerate(suborbitals[orbital2]):
                                func = None
                                # generate the name of the orbital integral (try all permutations)
                                for names in [f"{name1}/{name2}", f"{name2}/{name1}"]:
                                    for orbital_integral_name, swap in [(f"{names}:V_{sub1}_{sub2}", False), (f"{names}:V_{sub2}_{sub1}", True)]:
                                        if orbital_integral_name in orbital_integrals:
                                            res_inv = inv if swap else 1
                                            func = orbital_integrals[orbital_integral_name]
                                            break
                                    if func is not None: break
                                assert func is not None, f"Orbital function {name1}/{name2}:V_{sub1}_{sub2} is missing (internal error)"
                                if type(func) == list:
                                    assert inv == 1, f"can't specify the orbital integral of {orbital1}{orbital2} just by distance"
                                elif type(func) == dict:
                                    suborb1 = np.zeros(len(suborbitals[orbital1]))
                                    suborb1[sub1_i] = 1
                                    suborb2 = np.zeros(len(suborbitals[orbital2]))
                                    suborb2[sub2_i] = 1
                                for i1, n1 in enumerate(neighbors):
                                    x = n1 + A @ (pos1 - pos2)
                                    if type(func) is list:
                                        index = index_from_r(np.linalg.norm(x))
                                        p = func[index] if index < len(func) else 0.0
                                    elif type(func) is dict:
                                        p = None # if no value is found
                                        r = np.linalg.norm(x)
                                        for pos, value in func.items():
                                            rpos = np.linalg.norm(pos)
                                            # TODO these thresholds assume A is normalized...
                                            if abs(rpos - r) < 1e-5:
                                                for s_op2, urepr1, urepr2 in zip(sym.S, urepr[orbital1].U, urepr[orbital2].U):
                                                    suborb1_rot = urepr1 @ suborb1
                                                    suborb2_rot = urepr2 @ suborb2
                                                    # here suborb1_rot = ±suborb1, suborb2_rot = ±suborb2
                                                    s_inv = (suborb1_rot @ suborb1) * (suborb2_rot @ suborb2)
                                                    #assert abs(abs(s_inv) - 1) < 1e-5, f"{suborb1_rot} {suborb2_rot}"
                                                    if abs(abs(s_inv) - 1) < 1e-5:
                                                        if np.linalg.norm(s_op2 @ pos - x) < 1e-5:
                                                            p = value * s_inv
                                                            break
                                                        if np.linalg.norm(s_op2 @ pos + x) < 1e-5:
                                                            p = value * s_inv * inv
                                                            break
                                            if p is not None:
                                                break
                                        if p is None:
                                            p = 0.0
                                    else:
                                        p = func(x)
                                    params[i1, k1, k2] = p * res_inv
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
        # remove all neighbors which are completely zero
        select = np.linalg.norm(params, axis=(-1, -2)) > 0.0
        neighbors = np.array(np.asarray(neighbors)[select])
        params = np.array(params[select])
        # construct the model and return it
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

class SlaterKosterOriginal:
    """Slater-Koster model for cubic crystals.
    Implementated exactly like in the original paper.
    """
    def __init__(self, s_s_000, x_x_000, xy_xy_000, d2_d2_000, s_s_110, s_x_110, s_xy_110, s_d2_110, x_x_110, x_x_011, x_y_110, x_xy_110, x_xy_011, z_d2_011, z_d1_011, xy_xy_110, xy_xy_011, xy_xz_011, xy_d2_110, d2_d2_110, d1_d1_110, s_s_200=0, s_x_200=0, s_d2_002=0, x_x_200=0, y_y_200=0, x_xy_020=0, z_d2_002=0, xy_xy_200=0, xy_xy_002=0, d2_d2_002=0, d1_d1_002=0):
        # TODO add missing sc variables...
        self.__dict__ |= locals()
    
    @staticmethod
    def fcc(s_s_000, x_x_000, xy_xy_000, d2_d2_000, s_s_110, s_x_110, s_xy_110, s_d2_110, x_x_110, x_x_011, x_y_110, x_xy_110, x_xy_011, z_d2_011, z_d1_011, xy_xy_110, xy_xy_011, xy_xz_011, xy_d2_110, d2_d2_110, d1_d1_110, s_s_200=0, s_x_200=0, s_d2_002=0, x_x_200=0, y_y_200=0, x_xy_020=0, z_d2_002=0, xy_xy_200=0, xy_xy_002=0, d2_d2_002=0, d1_d1_002=0) -> Self:
        return SlaterKosterOriginal()

    def f(self, k):
        cos = np.cos(k).T
        sin = np.sin(k).T
        cos2 = np.cos(2*k).T
        sin2 = np.sin(2*k).T
        # TODO check if all terms are correctly symmetrized
        s_s = self.s_s_000 + 2*self.s_s_100*(cos[0]+cos[1]+cos[2]) + 4*self.s_s_110*(cos[0]*cos[1]+cos[0]*cos[2]+cos[1]*cos[2]) + 8*self.s_s_111*cos[0]*cos[1]*cos[2] + 2*self.s_s_200*(cos2[0]+cos2[1]+cos2[2])
        s_x = 2j*self.s_x_100*sin[0] + 2j*self.s_x_200*sin2[0] + 4j*self.s_x_110*(sin[0]*cos[1]+sin[0]*cos[2]) + 8j*self.s_x_111*sin[0]*cos[1]*cos[2]
        s_y = 2j*self.s_x_100*sin[1] + 2j*self.s_x_200*sin2[1] + 4j*self.s_x_110*(sin[1]*cos[0]+sin[1]*cos[2]) + 8j*self.s_x_111*cos[0]*sin[1]*cos[2]
        s_z = 2j*self.s_x_100*sin[2] + 2j*self.s_x_200*sin2[2] + 4j*self.s_x_110*(sin[2]*cos[0]+sin[2]*cos[1]) + 8j*self.s_x_111*cos[0]*cos[1]*sin[2]
        s_xy = -4*self.s_xy_110*sin[0]*sin[1] - 8*self.s_xy_111*sin[0]*sin[1]*cos[2]
        s_xz = -4*self.s_xy_110*sin[0]*sin[2] - 8*self.s_xy_111*sin[0]*cos[1]*sin[2]
        s_yz = -4*self.s_xy_110*sin[1]*sin[2] - 8*self.s_xy_111*cos[0]*sin[1]*sin[2]
        s_d1 = 3**.5*(self.s_d2_001*(cos[0]-cos[1]) + self.s_d2_002*(cos2[0]-cos2[1]) + 2*self.s_d2_110*(-cos[0]*cos[2]+cos[1]*cos[2]))
        s_d2 = self.s_d2_001*(-cos[0]-cos[1]+2*cos[2]) + self.s_d2_002*(-cos2[0]-cos2[1]+2*cos2[2]) - 2*self.s_d2_110*(-2*cos[0]*cos[1]+cos[0]*cos[2]+cos[1]*cos[2])
        x_x = self.x_x_000 + 2*self.x_x_100*cos[0] + 2*self.y_y_100*(cos[1]+cos[2]) + 2*self.x_x_200*cos2[0] + 2*self.y_y_200*(cos2[1]+cos2[2]) + 4*self.x_x_110*(cos[0]*cos[1]+cos[0]*cos[2]) + 4*self.x_x_011*cos[1]*cos[2] + 8*self.x_x_111*cos[0]*cos[1]*cos[2]
        y_y = self.x_x_000 + 2*self.x_x_100*cos[1] + 2*self.y_y_100*(cos[0]+cos[2]) + 2*self.x_x_200*cos2[1] + 2*self.y_y_200*(cos2[0]+cos2[2]) + 4*self.x_x_110*(cos[1]*cos[0]+cos[1]*cos[2]) + 4*self.x_x_011*cos[0]*cos[2] + 8*self.x_x_111*cos[0]*cos[1]*cos[2]
        z_z = self.x_x_000 + 2*self.x_x_100*cos[2] + 2*self.y_y_100*(cos[0]+cos[1]) + 2*self.x_x_200*cos2[2] + 2*self.y_y_200*(cos2[0]+cos2[1]) + 4*self.x_x_110*(cos[2]*cos[1]+cos[2]*cos[0]) + 4*self.x_x_011*cos[1]*cos[0] + 8*self.x_x_111*cos[0]*cos[1]*cos[2]
        x_y = -4*self.x_y_110*sin[0]*sin[1] - 8*self.x_y_111*sin[0]*sin[1]*cos[2]
        y_z = -4*self.x_y_110*sin[1]*sin[2] - 8*self.x_y_111*cos[0]*sin[1]*sin[2]
        z_x = -4*self.x_y_110*sin[2]*sin[0] - 8*self.x_y_111*sin[0]*cos[1]*sin[2]
        x_xy = 2j*self.x_xy_010*sin[1] + 4j*self.x_xy_110*cos[0]*sin[1] + 4j*self.x_xy_011*sin[1]*cos[2] + 8j*self.x_xy_111*cos[0]*sin[1]*cos[2]
        x_xz = 2j*self.x_xy_010*sin[2] + 4j*self.x_xy_110*cos[0]*sin[2] - 4j*self.x_xy_011*sin[2]*cos[1] + 8j*self.x_xy_111*cos[0]*cos[1]*sin[2]
        # TODO symmetric y_xy, y_yz, z_yz, z_xz
        x_yz = -8j*self.x_yz_111*sin[0]*cos[1]*cos[2]
        y_xz = -8j*self.x_yz_111*cos[0]*sin[1]*cos[2]
        z_xy = -8j*self.x_yz_111*cos[0]*cos[1]*sin[2]
        x_d1 = 3**.5*1j*(self.z_d2_001*sin[0] + self.z_d2_002*sin2[0] + 2*self.z_d2_011*(sin[0]*cos[1]+sin[0]*cos[2])) + 2j*self.z_d1_011*(sin[0]*cos[1]-sin[0]*cos[2]) + 8j*self.x_d1_111*sin[0]*cos[1]*cos[2]
        y_d1 = 3**.5*1j*(self.z_d2_001*sin[1] + self.z_d2_002*sin2[1] + 2*self.z_d2_011*(sin[1]*cos[0]+sin[1]*cos[2])) + 2j*self.z_d1_011*(sin[1]*cos[0]-sin[1]*cos[2]) + 8j*self.x_d1_111*cos[0]*sin[1]*cos[2]
        x_d2 = -1j*self.z_d2_001*sin[0] - 1j*self.z_d2_002*sin2[0] - 2j*self.z_d2_011*(sin[0]*cos[1]+sin[0]*cos[2]) + 2j*3**.5*self.z_d1_011*(sin[0]*cos[1]-sin[0]*cos[2]) - 8/3**.5*self.x_d1_111*sin[0]*cos[1]*cos[2]
        y_d2 = -1j*self.z_d2_001*sin[1] - 1j*self.z_d2_002*sin2[1] - 2j*self.z_d2_011*(sin[1]*cos[0]+sin[1]*cos[2]) + 2j*3**.5*self.z_d1_011*(sin[1]*cos[0]-sin[1]*cos[2]) - 8/3**.5*self.x_d1_111*cos[0]*sin[1]*cos[2]
        z_d2 =  2j*self.z_d2_001*sin[2] + 2j*self.z_d2_002*sin2[2] + 4j*self.z_d2_011*(cos[0]*sin[2]+cos[1]*sin[2]) + (16/3**.5)*1j*self.x_d1_111*cos[0]*sin[2]
        z_d1 = z_d2 # not described in the SK paper...
        xy_xy = self.xy_xy_000 + 2*self.xy_xy_100*(cos[0]+cos[1]) + 2*self.xy_xy_200*(cos2[0]+cos2[1]) + 2*self.xy_xy_001*cos[2] + 2*self.xy_xy_002*cos2[2] + 4*self.xy_xy_110*cos[0]*cos[1] + 4*self.xy_xy_011*(cos[0]*cos[2]+cos[1]*cos[2]) + 8*self.xy_xy_111*cos[0]*cos[1]*cos[2]
        yz_yz = self.xy_xy_000 + 2*self.xy_xy_100*(cos[1]+cos[2]) + 2*self.xy_xy_200*(cos2[1]+cos2[2]) + 2*self.xy_xy_001*cos[0] + 2*self.xy_xy_002*cos2[0] + 4*self.xy_xy_110*cos[1]*cos[2] + 4*self.xy_xy_011*(cos[1]*cos[0]+cos[2]*cos[0]) + 8*self.xy_xy_111*cos[0]*cos[1]*cos[2]
        xz_xz = self.xy_xy_000 + 2*self.xy_xy_100*(cos[0]+cos[2]) + 2*self.xy_xy_200*(cos2[0]+cos2[2]) + 2*self.xy_xy_001*cos[1] + 2*self.xy_xy_002*cos2[1] + 4*self.xy_xy_110*cos[0]*cos[2] + 4*self.xy_xy_011*(cos[0]*cos[1]+cos[2]*cos[1]) + 8*self.xy_xy_111*cos[0]*cos[1]*cos[2]
        xy_xz = -4*self.xy_xz_011*sin[1]*sin[2] - 8*self.xy_xz_111*cos[0]*sin[1]*sin[2]
        xy_yz = -4*self.xy_xz_011*sin[0]*sin[2] - 8*self.xy_xz_111*sin[0]*cos[1]*sin[2]
        xz_yz = -4*self.xy_xz_011*sin[0]*sin[1] - 8*self.xy_xz_111*sin[0]*sin[1]*cos[2]
        xy_d1 = 0 # what about third neighbors?
        xy_d2 = -4*self.xy_d2_110*sin[0]*sin[1] - 8*self.xy_d2_111*sin[0]*sin[1]*cos[2] # why did it say yx in the SK paper?
        xz_d1 = 2*3**.5*self.xy_d2_110*sin[0]*sin[2] + 4*3**.5*self.xy_d2_111*sin[0]*cos[1]*sin[2]
        yz_d1 = 2*3**.5*self.xy_d2_110*sin[1]*sin[2] + 4*3**.5*self.xy_d2_111*cos[0]*sin[1]*sin[2]
        xz_d2 = 2*self.xy_d2_110*sin[0]*sin[2] + 4*self.xy_d2_111*sin[0]*cos[1]*sin[2]
        yz_d2 = 2*self.xy_d2_110*sin[1]*sin[2] + 4*self.xy_d2_111*cos[0]*sin[1]*sin[2]
        d1_d1 = self.d2_d2_000 + 3/2*self.d2_d2_1001*(cos[0]+cos[1]) + 2*self.d1_d1_001*(cos[0]/4+cos[1]/4+cos[2]) + 2*self.d1_d1_002*(cos2[0]/4+cos2[1]/4+cos2[2]) + 3*self.d2_d2_110*(cos[0]*cos[2]+cos[1]*cos[2]) + 3*self.d1_d1_110*(cos[0]*cos[1]+cos[0]*cos[2]/4+cos[1]*cos[2]/4) + 8*self.d2_d2_111*cos[0]*cos[1]*cos[2]
        d2_d2 = self.d2_d2_000 + 2*self.d2_d2_001*(cos[0]/4+cos[1]/4+cos[2]) + 3/2*self.d1_d1_001*(cos[0]+cos[1]) + 3/2*self.d1_d1_002*(cos2[0]+cos2[1]) + 4*self.d2_d2_110*(cos[0]*cos[2]+cos[0]*cos[2]/4+cos[1]*cos[2]/4) + 3*self.d1_d1_110*(cos[0]*cos[1]+cos[1]*cos[2]) + 8*self.d2_d2_111*cos[0]*cos[1]*cos[2]
        d1_d2 = 3**.5/2*self.d2_d2_001*(-cos[0]+cos[1]) - 3**.5/2*self.d1_d1_001*(-cos[0]+cos[1]) - 3**.5/2*self.d1_d1_002*(-cos2[0]+cos2[1]) + 3**.5*self.d2_d2_110*(cos[0]*cos[2]-cos[1]*cos[2]) - 3**.5*self.d1_d1_110*(cos[0]*cos[2]-cos[1]*cos[2])
        H = np.array([
            #  s     x     y     z     yz     xz     xy     d1     d2
            [ s_s,  s_x,  s_y,  s_z,  s_yz,  s_xz,  s_xy,  s_d1,  s_d2], # s
            [ s_x,  x_x,  x_y,  z_x,  x_yz,  x_xz,  x_xy,  x_d1,  x_d2], # x
            [ s_y,  x_y,  y_y,  y_z,  x_xy,  y_xz,  x_xy,  y_d1,  y_d2], # y
            [ s_z,  z_x,  y_z,  z_z,  x_xy,  x_xy,  z_xy,  z_d1,  z_d2], # z
            [s_yz, x_yz, x_xy, x_xy, yz_yz, xz_yz, xy_yz, yz_d1, yz_d2], # yz
            [s_xz, x_xz, y_xz, x_xy, xz_yz, xz_xz, xy_xz, xz_d1, xz_d2], # xz
            [s_xy, x_xy, x_xy, z_xy, xy_yz, xy_xz, xy_xy, xy_d1, xy_d2], # xy
            [s_d1, x_d1, y_d1, z_d1, yz_d1, xz_d1, xy_d1, d1_d1, d1_d2], # d1
            [s_d2, x_d2, y_d2, z_d2, yz_d2, xz_d2, xy_d2, d1_d2, d2_d2], # d2
        ])
        for i in range(1+3+5):
            for j in range(i):
                H[i,j] = np.conj(H[i,j])
        return H