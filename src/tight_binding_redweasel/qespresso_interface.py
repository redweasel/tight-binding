"""
Simple interface for basic calculations with Quantum Espresso.
Assuming Quantum Espresso (QE) is installed and in the PATH.
This file deliberately doesn't depend on the rest of the library.
This is just the interface to QE and nothing more or less.
That makes the usage a bit less convenient, but that is
accomodated by functions in `kpaths` and the rest of the library.

# Example

For running the calculation and plotting the result with this library:
```
from tight_binding_redweasel import Symmetry, kpaths
from tight_binding_redweasel import qespresso_interface as qe
from matplotlib import pyplot as plt

# extend path to include Quantum Espresso
#import os
#qe_path = os.path.expanduser("~/path/to/q-e/bin/")
#if qe_path not in os.environ["PATH"]:
#    os.environ["PATH"] += os.pathsep + qe_path
# customize the MPI command
# qe.mpi_run = "mpirun "

# see PSLibrary at http://pseudopotentials.quantum-espresso.org/legacy_tables/ps-library for the names
qe.qe_prepare({"Cu": "Cu.pbesol-dn-kjpaw_psl.1.0.0.UPF"})

basis = [(0, 0, 0)]
types = ["Cu"]
cu_crystal = qe.QECrystal("cu", qe.fcc(3.6148), basis, types)

print(f"Density: {cu_crystal.mass_density()*1e-3:.2f} g/m³")

if not cu_crystal.read_available():
    cu_crystal.scf(8)
    cu_crystal.nscf(qe.k_grid(16), 10)

k_smpl, bands, symmetries, fermi_energy = cu_crystal.read_bands_crystal()
A_norm = cu_crystal.A / (np.max(np.abs(cu_crystal.A), axis=1, keepdims=True)*2) # normalisation for fcc and bcc
cu_interp_crystal = kpaths.interpolate(k_smpl, bands, Symmetry(symmetries), method="cubic")
cu_interp = lambda k: cu_interp_crystal(k @ A_norm)

kpaths.FCC_PATH.plot(cu_interp, label_bands="left", ylim=(2.5, 22))
plt.axhline(fermi_energy, color='k')
plt.show()
```

If a calculation has already been performed and one just wants to read in the data from the calculation,
the creation of the crystal class can be done automatically from the saved data.
The following code shows how to do that.
```
from tight_binding_redweasel import Symmetry, kpaths
from tight_binding_redweasel import qespresso_interface as qe
from matplotlib import pyplot as plt

cu_crystal = qe.from_disk("cu")
k_smpl, bands, symmetries, fermi_energy = cu_crystal.read_bands_crystal()
A_norm = cu_crystal.A / (abs(cu_crystal.A[0,0])*2) # normalisation for fcc and bcc
cu_interp_crystal = kpaths.interpolate(k_smpl, bands, Symmetry(symmetries), method="cubic")
cu_interp = lambda k: cu_interp_crystal(k @ A_norm)

kpaths.FCC_PATH.plot(cu_interp, label_bands="left", ylim=(2.5, 22))
plt.axhline(fermi_energy, color='k')
plt.show()
```
"""

import os
import re
import subprocess
from sys import platform
import numpy as np

ibrav_map = { "none": 0, "sc": 1, "fcc": 2, "bcc": 3, "hex": 4, "tri": 5, "monoclinic": 12 }

pp_files = {}

element_masses = [1.008, 4.0026, 7.0, 9.012183, 10.81, 12.011, 14.007, 15.999, 18.99840316, 20.18, 22.9897693, 24.305, 26.981538, 28.085, 30.973762, 32.07, 35.45, 39.9, 39.0983, 40.08, 44.95591, 47.867, 50.9415, 51.996, 54.93804, 55.84, 58.93319, 58.693, 63.55, 65.4, 69.723, 72.63, 74.92159, 78.97, 79.9, 83.8, 85.468, 87.62, 88.90584, 91.22, 92.90637, 95.95, 96.90636, 101.1, 102.9055, 106.42, 107.868, 112.41, 114.818, 118.71, 121.76, 127.6, 126.9045, 131.29, 132.905452, 137.33, 138.9055, 140.116, 140.90766, 144.24, 144.91276, 150.4, 151.964, 157.2, 158.92535, 162.5, 164.93033, 167.26, 168.93422, 173.05, 174.9668, 178.49, 180.9479, 183.84, 186.207, 190.2, 192.22, 195.08, 196.96657, 200.59, 204.383, 207.0, 208.9804, 208.98243, 209.98715, 222.01758, 223.01973, 226.02541, 227.02775, 232.038, 231.03588, 238.0289, 237.048172, 244.0642, 243.06138, 247.07035, 247.07031, 251.07959, 252.083, 257.09511, 258.09843, 259.101, 266.12, 267.122, 268.126, 269.128, 270.133, 269.1336, 277.154, 282.166, 282.169, 286.179, 286.182, 290.192, 290.196, 293.205, 294.211, 295.216]
element_numbers = {'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 'Br': 34, 'Kr': 35, 'Rb': 36, 'Sr': 37, 'Y': 38, 'Zr': 39, 'Nb': 40, 'Mo': 41, 'Tc': 42, 'Ru': 43, 'Rh': 44, 'Pd': 45, 'Ag': 46, 'Cd': 47, 'In': 48, 'Sn': 49, 'Sb': 50, 'Te': 51, 'I': 52, 'Xe': 53, 'Cs': 54, 'Ba': 55, 'La': 56, 'Ce': 57, 'Pr': 58, 'Nd': 59, 'Pm': 60, 'Sm': 61, 'Eu': 62, 'Gd': 63, 'Tb': 64, 'Dy': 65, 'Ho': 66, 'Er': 67, 'Tm': 68, 'Yb': 69, 'Lu': 70, 'Hf': 71, 'Ta': 72, 'W': 73, 'Re': 74, 'Os': 75, 'Ir': 76, 'Pt': 77, 'Au': 78, 'Hg': 79, 'Tl': 80, 'Pb': 81, 'Bi': 82, 'Po': 83, 'At': 84, 'Rn': 85, 'Fr': 86, 'Ra': 87, 'Ac': 88, 'Th': 89, 'Pa': 90, 'U': 91, 'Np': 92, 'Pu': 93, 'Am': 94, 'Cm': 95, 'Bk': 96, 'Cf': 97, 'Es': 98, 'Fm': 99, 'Md': 100, 'No': 101, 'Lr': 102, 'Rf': 103, 'Db': 104, 'Sg': 105, 'Bh': 106, 'Hs': 107, 'Mt': 108, 'Ds': 109, 'Rg': 110, 'Cn': 111, 'Nh': 112, 'Fl': 113, 'Mc': 114, 'Lv': 115, 'Ts': 116, 'Og': 117}
element_names = ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium', 'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine', 'Xenon', 'Cesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium', 'Promethium', 'Samarium', 'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium', 'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury', 'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine', 'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium', 'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium', 'Curium', 'Berkelium', 'Californium', 'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium', 'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium', 'Bohrium', 'Hassium', 'Meitnerium', 'Darmstadtium', 'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium', 'Moscovium', 'Livermorium', 'Tennessine', 'Oganesson']
unit_to_SI = {'angstrom': 1e-10, 'bohr': 5.2917721054482e-11}

mpi_run = "mpirun --use-hwthread-cpus "

def qe_prepare(pseudo_potential_files_dict=None):
    if not os.path.exists("./qe-data"):
        os.mkdir("./qe-data")
    if not os.path.exists("./pseudo"):
        os.mkdir("./pseudo")

    global pp_files
    if pseudo_potential_files_dict is None:
        pseudo_potential_files_dict = pp_files
    else:
        # replace instead of extend...
        # Problem with extending is, that multiple files can exist for a single element.
        pp_files = pseudo_potential_files_dict

    # TODO checkout https://github.com/dalcorso/pslibrary for automation of this process?
    for name, file in pseudo_potential_files_dict.copy().items():
        if file is None:
            # try to automatically choose the filename
            # TODO these are close, but not correct...
            # they are missing the valence tag spd
            # a correct name would be "Cu.rel-pbesol-dn-kjpaw_psl.1.0.0.UPF"
            #                                        ^
            # To get those one would need to query f"http://pseudopotentials.quantum-espresso.org/legacy_tables/ps-library/{name.lower()}"
            file = name + ".rel-pbesol-n-kjpaw_psl.1.0.0.UPF"
            pseudo_potential_files_dict[name] = file
        if not os.path.isfile("./pseudo/" + file):
            # TODO sanitize "file" -> otherwise command injection is possible
            if re.match(r"[a-zA-Z0-9\.\-]", file) is None:
                print("skipped file", file, "because it contains incompatible characters")
                continue
            res = subprocess.run("wget https://pseudopotentials.quantum-espresso.org/upf_files/" + file, cwd="./pseudo", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if "Not Found" in res.stdout.decode("utf-8"):
                print("could not find", file, "at https://pseudopotentials.quantum-espresso.org/upf_files/" + file)


# general bandstructure plotting
def plot_bands_generic(bands, x_smpl=None, *args, **kwargs):
    from matplotlib import pyplot as plt
    bands = np.asarray(bands)
    plt.gca().set_prop_cycle(None)
    for i in range(len(bands[0])):
        if x_smpl is None:
            plt.plot(bands[:,i], *args, **kwargs)
        else:
            plt.plot(x_smpl, bands[:,i], *args, **kwargs)

# create the A matrix for CELL_PARAMETERS for a fcc lattice
def sc(a):
    return np.eye(3)*a

# create the A matrix for CELL_PARAMETERS for a fcc lattice
def fcc(a):
    return np.array([[-1,0,1], [0,1,1], [-1,1,0]]).T*a/2

# create the A matrix for CELL_PARAMETERS for a bcc lattice
def bcc(a):
    return np.array([[1,1,1], [-1,1,1], [-1,-1,1]]).T*a/2

# create a hexagonal lattice with spacing a and height c
def hexagonal(a, c):
    return np.array([[1,0,0], [-1/2,3**.5/2,0], [0,0,c/a]]).T*a

# Triclinic (aP) C_i point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def triclinic(a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    assert np.pi > alpha and alpha > 0.0 and np.pi > beta and beta > 0.0 and np.pi > gamma and gamma > 0.0, f"all angles need to be in the range (0, π), but were α={alpha/np.pi:.4}π, β={beta/np.pi:.4}π, γ={gamma/np.pi:.4}π"
    assert alpha + beta + gamma < np.pi*2, f"invalid combination of angles α={alpha/np.pi:.4}π, β={beta/np.pi:.4}π, γ={gamma/np.pi:.4}π (inner angle sum must by smaller than 2π, but was {(alpha + beta + gamma)/np.pi}π)"
    bc = np.cos(beta)
    bs = np.sin(beta)
    gc = np.cos(gamma)
    gs = np.sin(gamma)
    v = (np.cos(alpha) - gc*bc) / gs
    assert v*v < bs*bs, f"invalid combination of angles α={alpha/np.pi:.4}π, β={beta/np.pi:.4}π, γ={gamma/np.pi:.4}π (one angle is bigger than the other two combined)"
    return np.array([[a, 0.0, 0.0],
                [gc * b, np.sin(gamma) * b, 0.0],
                [bc * c, v * c, (bs*bs - v*v)**.5 * c]]).T

# Monoclinic (mP) C_2h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def monoclinic(a: float, b: float, c: float, beta: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    assert np.pi > beta and beta > 0.0, f"angle needs to be in the range (0, π), but was β={beta/np.pi:.4}π"
    return np.array([[a, 0.0, 0.0],
                [0.0, b, 0.0],
                [np.cos(beta) * c, 0.0, np.sin(beta) * c]]).T

# Base centered monoclinic (mS) C_2h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def monoclinic_base_centered(a: float, b: float, c: float, beta: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    assert np.pi > beta and beta > 0.0, f"angle needs to be in the range (0, π), but was β={beta/np.pi:.4}π"
    return np.array([[0.5*a, -0.5*b, 0.0],
                [0.5*a, 0.5*b, 0.0],
                [np.cos(beta) * c, 0.0, np.sin(beta) * c]]).T

# Orthorhombic (oP) D_2h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def orthorhombic(a: float, b: float, c: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    return np.array([[a, 0.0, 0.0],
                [0.0, b, 0.0],
                [0.0, 0.0, c]]).T

# Base centered orthorhombic (oP) D_2h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def orthorhombic_base_centered(a: float, b: float, c: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    return np.array([[0.5*a, -0.5*b, 0.0],
                [0.5*a, 0.5*b, 0.0],
                [0.0, 0.0, c]]).T

# Body centered orthorhombic (oI) D_2h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def orthorhombic_body_centered(a: float, b: float, c: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    return np.array([[-0.5*a, 0.5*b, 0.5*c],
                [0.5*a, -0.5*b, 0.5*c],
                [0.5*a, 0.5*b, -0.5*c]]).T

# Face centered orthorhombic (oF) D_2h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def orthorhombic_face_centered(a: float, b: float, c: float):
    assert a > 0.0 and b > 0.0 and c > 0.0, f"all lengths a, b, c need to be positive, but were {a:.3e}, {b:.3e}, {c:.3e}"
    return np.array([[0.0, 0.5*b, 0.5*c],
                [0.5*a, 0.0, 0.5*c],
                [0.5*a, 0.5*b, 0.0]]).T

# Tetragonal (tP) D_4h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def tetragonal(a: float, c: float):
    assert a > 0.0 and c > 0.0, f"all lengths a, c need to be positive, but were {a:.3e}, {c:.3e}"
    return np.array([[a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, c]]).T

# Body centered tetragonal (tI) D_4h point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def tetragonal_body_centered(a: float, c: float):
    assert a > 0.0 and c > 0.0, f"all lengths a, c need to be positive, but were {a:.3e}, {c:.3e}"
    return np.array([[-0.5*a, 0.5*a, 0.5*c],
                [0.5*a, -0.5*a, 0.5*c],
                [0.5*a, 0.5*a, -0.5*c]]).T

# Rhombohedral (hR) D_3d point group (Schönflies notation https://en.wikipedia.org/wiki/Schoenflies_notation)
def rhombohedral(a: float, alpha: float):
    assert a > 0.0, "length a needs to be positive, but was {a:.3e}"
    # basis in which the crystal stands upright
    # -> reciprocal is also of the same type, but rotated by 180°
    assert alpha > 0.0 and alpha < 2.0*np.pi/3, f"angle needs to be in the range (0, 2π/3), but was α={alpha/np.pi:.4}π"
    v = 2.0/3.0 - 2.0/3.0*np.cos(alpha)
    h = (1.0 - v)**.5 * a
    r = v**.5 * a
    return np.array([[r, 0.0, h],
                [-0.5*r, 3**.5/2 * r, h],
                [-0.5*r, -3**.5/2 * r, h]]).T

def k_points(points, unit: str) -> str:
    """Convert a list of points with a given k-space unit, into the parameter string that Quantum Espresso wants.

    Args:
        points (arraylike(N, 3)): List of 3D k-space points.
        unit (str): One of the following units: "tpiba", "crystal", "tpiba_b", "crystal_b", "tpiba_c", "crystal_c"

    Returns:
        str: String for Quantum Espresso
    """
    k_points = f"K_POINTS {{{unit}}}\n{len(points)}\n"
    for x, y, z in points:
        k_points = k_points + f"{x} {y} {z} 1\n"
    return k_points

def k_grid(size: int, size2: int=None, size3: int=None) -> str:
    """Generate the parameter string that Quantum Espresso wants for a rectilinear grid.

    Args:
        size (int): Points along the x-direction.
        size2 (int, optional): Points along the y-direction. Defaults to "size".
        size3 (int, optional): Points along the z-direction. Defaults to "size".

    Returns:
        str: String for Quantum Espresso
    """
    if size2 is None:
        size2 = size
    if size3 is None:
        size3 = size
    return f"K_POINTS (automatic)\n{size} {size2} {size3} 0 0 0"

def print_orbitals(amplitudes, orbitals, threshold=0.05, end="\n"):
    """Given a list of orbital names and a distribution, print the contributing orbitals.

    Args:
        amplitudes (_type_): Amplitudes for each orbital. The absolute value squared is computed for the percentage output.
        orbitals (list): List of orbitals from `QECrystal.read_projections_order_spilling()`
    """
    distribution = np.abs(amplitudes)**2
    order = np.argsort(distribution)
    values = []
    for i in np.flip(order):
        if distribution[i] < threshold:
            break
        values.append(f"{orbitals[i]}: {distribution[i]:.1%}")
    if values:
        print(", ".join(values), end=end)
    else:
        print("None", end=end)

def version() -> str:
    """Get the version of Quantum Espresso that is in the PATH.

    Returns:
        str: version string of Quantum Espresso
    """
    out = subprocess.Popen('echo "" | pw.x', stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, shell=True).stdout.read().decode("utf-8")
    version = 'v' + out.split("\n")[1].split("starts")[0].split("v")[1].strip()
    return version

class QECrystal:
    def __init__(self, name: str, A, basis: list, types: list, kinetic_energy_cutoff: float=None, unit="angstrom", relativistic=False):
        """Constructor of QECrystal. Use `from_disk` instead if the data is already there.

        Args:
            name (str): Prefix/name used for all files related to this calculation.
            A (arraylike(3, 3)): Matrix of lattice vectors with the vectors as columns of the matrix.
            basis (list): Positions of the atoms in one cell in crystal coordinates [0,1[^3.
            types (list): Names of the atoms in one cell. Matching "basis".
            kinetic_energy_cutoff (float, optional): Maximal kinetic energy in the electron calculation. This defines the density of the lattice that gets used internally. Defaults to None.
            unit (str, optional): The unit of the lattice vectors "A". Can be "angstrom" or "bohr". Defaults to "angstrom".
        """
        assert unit in ("angstrom", "bohr")
        self.name = name
        self.unit = unit
        self.A = np.asarray(A)
        self.basis = np.asarray(basis)
        self.types = np.asarray(types)
        self.relativistic = relativistic
        assert len(self.types) == len(self.basis), "every basis atom needs a matching type"
        #self.ibrav = ibrav_map[symmetry]
        self.ibrav = 0 # use CELL_PARAMETERS instead!
        self.cell_scale = 1.0 # = celldm(1) from pw.x
        self.T = 0.0 # Temperature in Kelvin
        self.kinetic_energy_cutoff = kinetic_energy_cutoff
        if kinetic_energy_cutoff is None:
            self.set_cutoff_from_pseudopotentials()
        self.set_multitasking_parameters(1)

    def set_multitasking_parameters(self, nk, nd=1, ni=1, nt=1):
        """Set the multitasking parameters for the pw.x call.
        Setting nk to a divisor of the actual number of k-points is very efficient.
        The nt and nd settings need to be adjusted carefully for each problem.
        The FFT can not have too many task groups, so most processors would go into the diagonalisation.

        Args:
            nk (int): Parallel computed k-points.
            nd (int, optional): Processors for the diagonalisation. Defaults to 1.
            ni (int, optional): Parallel computed configurations (for phonons). Defaults to 1.
            nt (int, optional): Taskgroups for the FFT. Defaults to 1.
        """
        self.nk = nk
        self.nd = nd
        self.ni = ni
        self.nt = nt

    def plot_crystal(self, repeat=1, turntable=14, elevation=35):
        """3d plot of the crystal structure"""
        assert self.ibrav == 0, "only implemented for ibrav = 0"
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(projection="3d") # type: ignore
        restrict = False
        if repeat == 1:
            restrict = True
            repeat = 3
        offset = np.stack(np.reshape(np.meshgrid(*([np.arange(repeat)-(repeat-1)//2]*3)), (3, -1, 1)), axis=-1).reshape(-1, 1, 3)
        extended_basis = np.reshape(self.basis, (1, -1, 3)) + offset
        # TODO the multiple drawn lines destroy antialiasing... use NaN points instead
        cube_line = np.array([(0,0,0), (0,0,1), (0,1,1), (0,1,0), (0,0,0), (1,0,0), (1,0,1), (1,1,1), (1,1,0), (1,0,0), (1,1,0), (0,1,0), (0,1,1), (1,1,1), (1,0,1), (0,0,1)])
        deformed_cube = self.A @ cube_line.T
        ax.plot(*deformed_cube, "-k")
        # add annotations for which cube side is which lattice vector
        for i in range(3):
            ax.text(*self.A[:,i], f"$a_{i+1}$")
        limits = np.array([[0, 0], [0, 0], [0, 0]])
        for c, t in enumerate(sorted(list(set(self.types)))):
            basis_of_type = extended_basis[:,self.types == t].reshape(-1, 3)
            if restrict:
                x, y, z = basis_of_type.T
                basis_of_type = basis_of_type[(0 <= x) & (x <= 1) & (0 <= y) & (y <= 1) & (0 <= z) & (z <= 1)]
            positions = (self.A @ basis_of_type.T).T
            for i in range(len(positions)):
                ax.plot([positions[i][0]]*2, [positions[i][1]]*2, [0.0, positions[i][2]], "k:") # plot z-pole holding the atom to show depth
            ax.plot(*positions.T, f"oC{c}", label=t)
            # plot original basis atoms larger (even if they are outside of the restricted cell)
            ax.plot(*(self.A @ np.reshape(self.basis, (1, -1, 3))[:,self.types == t].reshape(-1, 3).T), f"oC{c}", markersize=12)
            # make tight viewbox
            positions = np.concatenate([positions.T, limits], axis=1)
            limits = np.concatenate([np.min(positions, axis=1, keepdims=True), np.max(positions, axis=1, keepdims=True)], axis=1)
        ax.view_init(elev=elevation, azim=turntable)
        for lim, dim in zip(limits, 'xyz'):
            getattr(ax, f'set_{dim}lim')(*lim)
        # add lines for the axes
        ax.plot(limits[0], [0]*2, [0]*2, 'k--')
        ax.plot([0]*2, limits[1], [0]*2, 'k--')
        ax.plot([0]*2, [0]*2, limits[2], 'k--')
        ax.set_xlabel(f'x [{self.unit.replace("angstrom", "Å")}]')
        ax.set_ylabel(f'y [{self.unit.replace("angstrom", "Å")}]')
        ax.set_zlabel(f'z [{self.unit.replace("angstrom", "Å")}]')
        if restrict:
            ax.set_xlim(np.min(deformed_cube[0]), np.max(deformed_cube[0]))
            ax.set_ylim(np.min(deformed_cube[1]), np.max(deformed_cube[1]))
            ax.set_zlim(np.min(deformed_cube[2]), np.max(deformed_cube[2]))
        ax.set_aspect("equal")
        ax.legend()
    
    def _mass_per_cell(self) -> float:
        nuclei_mass = 1.66053907e-27 * sum((element_masses[element_numbers[t]] for t in self.types))
        electron_mass = 9.1093837e-31 * sum((element_numbers[t] for t in self.types))
        return electron_mass + nuclei_mass # in kg

    def mass_density(self) -> float:
        """Compute the mass density based on the given crystal lattice.
        This is very useful to sanity check the input parameters.

        Returns:
            float: mass density in kg/m^3
        """
        return self._mass_per_cell() / np.linalg.det(self.lattice_vectors_in_m())

    def scale_to_density(self, density: float):
        """Scale the basis to match a given density.

        Args:
            density (float): The target density in kg/m^3
        """
        self.A *= (self.mass_density() / density)**(1/3)

    def lattice_vectors_in_m(self) -> np.ndarray:
        """
        Returns:
            ndarray(3, 3): Matrix with lattice vectors in meters.
        """
        return self.A*unit_to_SI[self.unit]

    # ----- reading results of QUANTUM ESPRESSO -----

    def read_available(self) -> bool:
        """
        Returns:
            bool: True if the final xml output of the calculation is found.
        """
        filename = f"./qe-data/{self.name}.xml"
        return os.path.exists(filename)

    def read_bandx_raw(self):
        """Read the data from Bandx.dat instead of the plottable data."""
        with open(f"{self.name}.Bandx.dat", 'r') as file:
            # this file is just a mess of numbers, no real format to them
            header = file.readline() # ignore the first line
            band_count = int(header.split("nbnd=")[1].split(",")[0])
            k_size = 3
            numbers = []
            while file:
                line = file.readline()
                if not line:
                    break
                numbers.extend([float(x) for x in line.strip().split(" ") if x.strip()])
            numbers = np.array(numbers).reshape(-1, k_size + band_count)
            k_points = numbers[:,:k_size]
            bands = numbers[:,k_size:]
        return np.array(k_points), np.array(bands)
    
    def read_bandx(self): 
        """This function extracts the high symmetry points x from the output of bandx.out.
        It also finds the output file for the band structure computed by it and returns the content of that.
        """
        sym_x = []
        datafile = None
        with open(f"{self.name}.bandx.out", 'r') as file:
            for line in file:
                if "high-symmetry point" in line:
                    sym_x.append(float(line.split()[-1]))
                if "Plottable bands (eV) written to file" in line:
                    datafile = line.split()[-1].strip()
        if datafile is None:
            raise IOError("plottable band data not found.")
        z = np.loadtxt(datafile, unpack=True) # This loads the bandx.dat.gnu file
        return np.array(sym_x), z

    def plot_bands(self):
        """Plot the bands computed by bandx"""
        from matplotlib import pyplot as plt
        sym_x, z = self.read_bandx()
        ymin, ymax = np.min(z[1]), np.max(z[1])
        # insert NaN points for every line that doesn't satisfy x2 > x1 to break up the lines
        for i in reversed(range(len(z[0]) - 1)):
            if z[0, i] >= z[0, i+1]:
                z = np.insert(z, i+1, np.nan, axis=1)
        plt.plot(z[0], z[1])
        plt.ylim(ymin, ymax)
        plt.xlim(np.nanmin(z[0]), np.nanmax(z[0]))
        plt.vlines(sym_x, ymin, ymax, "black", linestyles="dashed", lw=0.55)
        fermi_energy = self.read_bands_crystal()[3]
        plt.axhline(fermi_energy, color='r')
        plt.ylabel("Energy in eV")
        plt.title(f"Bandstructure of {self.name}")
        return sym_x

    def read_bands_crystal(self, incomplete=False):
        """Read the data that has been computed (either by scf(), nscf(), or by bands()).
        k and symmetries are in crystal coordinates. That means k_points will be in [0,1[^3.
        The symmetries are in the basis for the crystal space k space, meaning they have whole number coefficients.
        If the symmetry includes translation, the resulting symmetry matrices will be projective 4d with the translation part in the last component.
        Read in the symmetries using `Symmetry.space_group()` or `Symmetry.point_group()`.
        The lattice vectors in `self.A` are also updated by this method.

        Args:
            incomplete (bool, optional): If True, the data wil be read from `/{name}.save/data-file-schema.xml` instead of the usual `{name}.xml`. Defaults to False.

        Returns:
            tuple: k_points, bands, symmetries, fermi_energy
        """
        # read directly from the bands xml file (see https://realpython.com/python-xml-parser/)
        # read in k_points, eigenvalues and symmetries (as unnamed O(3) matrices)
        k_points = []
        bands = []
        S = []
        S_trans = []
        to_eV = 27.21138625 # from Hartree = 2Ry to 1eV
        bohr_to_angstrom = 0.52917721 # bohr_radius in Angstrom
        filename = f"./qe-data/{self.name}.save/data-file-schema.xml" if incomplete else f"./qe-data/{self.name}.xml"
        with open(filename, 'r') as file:
            from xml.dom.minidom import parse
            document = parse(file)
            #root = document.documentElement
            root = document.getElementsByTagName("qes:espresso")[0]
            assert root.getAttribute("Units") == "Hartree atomic units" # only accept these units for now
            output = root.getElementsByTagName("output")[0]
            b1 = [float(x) for x in output.getElementsByTagName("b1")[0].firstChild.nodeValue.strip().split()]
            b2 = [float(x) for x in output.getElementsByTagName("b2")[0].firstChild.nodeValue.strip().split()]
            b3 = [float(x) for x in output.getElementsByTagName("b3")[0].firstChild.nodeValue.strip().split()]
            reciprocal = np.array([b1, b2, b3]).T
            inv_reciprocal = np.linalg.inv(reciprocal)

            # This A is in bohr_radius -> convert it to Angstrom
            a1 = [float(x) for x in output.getElementsByTagName("a1")[0].firstChild.nodeValue.strip().split()]
            a2 = [float(x) for x in output.getElementsByTagName("a2")[0].firstChild.nodeValue.strip().split()]
            a3 = [float(x) for x in output.getElementsByTagName("a3")[0].firstChild.nodeValue.strip().split()]
            A = np.array([a1, a2, a3]).T
            if self.unit == "angstrom":
                A *= bohr_to_angstrom
            self.A = A # set the internal A instead of returning it
            # TODO consider returning "reciprocal" if it is too difficult to recontruct it
            # inv_reciprocal and A.T are collinear, but with what factor???
            #assert np.linalg.norm(inv_reciprocal - (A.T / np.linalg.norm(A[0]))) < 1e-7, f"reciprocal lattice doesn't match real lattice... This problem comes from Quantum Espresso. The compared matrices were\n{inv_reciprocal}\nand\n{A.T / np.linalg.norm(A[0])}"
            # now find all the data in the xml file
            symmetry_list = output.getElementsByTagName("symmetry")
            for sym in symmetry_list:
                rot = sym.getElementsByTagName("rotation")[0]
                # read content and convert to matrix
                content = rot.firstChild.nodeValue
                mat = np.array([float(x) for x in content.strip().split()]).reshape(3, 3)
                S.append(mat)
                # also read in fractional translations
                trans = sym.getElementsByTagName("fractional_translation")
                if len(trans):
                    content = trans[0].firstChild.nodeValue
                    S_trans.append(np.array([float(x) for x in content.strip().split()]))
                else:
                    S_trans.append(np.zeros(3))
            ks_energies = output.getElementsByTagName("ks_energies")
            for ks_e in ks_energies:
                k_point = ks_e.getElementsByTagName("k_point")[0]
                eigenvalues = ks_e.getElementsByTagName("eigenvalues")[0]
                k_points.append([float(x) for x in k_point.firstChild.nodeValue.strip().split()])
                bands.append([float(x) * to_eV for x in eigenvalues.firstChild.nodeValue.strip().split()])
            # transform k_points to crystal coordinates
            k_points = (inv_reciprocal @ np.array(k_points).T).T
            # transform symmetries to crystal coordinates
            # HOWEVER: they seem to already be in crystal coordinates!
            # HOWEVER: they are for the real lattice, while I only care about the symmetries for the reciprocal lattice.
            if len(S) == 0:
                # there is always one symmetry, the trivial symmetry
                S = np.eye(3)[None,...]
            # assert np.linalg.norm(np.einsum("nij,nik->njk", S, S) - np.eye(len(S[0]))) < 1e-5, "symmetries in crystal space are not orthogonal"
            assert np.linalg.norm(np.round(S) - S) < 1e-8, "symmetries in crystal space should be integers, but are not"
            assert len(S_trans) == len(S), "every symmetry needs an associated translational part if one of them has one."
            # only return the symmetries without translational part
            if not all((np.linalg.norm(t) < 1e-10 for t in S_trans)):
                # adding translational part as projective part!
                S = [np.block([[s, t[:,None]], [np.zeros((1, 3)), np.eye(1)]]) for s, t in zip(S, S_trans)]
            # transform S from the real space symmetries to the reciprocal space symmetries.
            S = np.linalg.inv(np.swapaxes(S, -1, -2))
            fermi_energy_node = output.getElementsByTagName("fermi_energy")[0]
            fermi_energy = float(fermi_energy_node.firstChild.nodeValue.strip()) * to_eV
        return np.array(k_points), np.array(bands), S, fermi_energy
    
    def read_projections_order_spilling(self):
        """Extract the names of the orbitals and their order from `{name}.projwfc.out`.
        Since the file also contains the very important spilling parameter, that is also returned.
        This is a HACK that may not work in other versions of the program than the tested version v.7.3.1.

        Returns:
            tuple: (order, spilling) where order is a list of strings with the type and index of the atom and name of the orbital.
        """
        with open(f"./{self.name}.projwfc.out", 'r') as file:
            content = file.read()
            relevant_part = content.split("Lowdin Charges:")[1].strip()
            order_str, spilling = relevant_part.split("Spilling Parameter:")
            spilling = float(spilling.split("\n")[0].strip())
            order = []
            for line in order_str.split("Atom #"):
                if not line.strip():
                    continue
                index, data = line.strip().split(":")
                index = int(index.strip())
                for key_value in data.split(","):
                    if not key_value.strip():
                        continue
                    key, _value = key_value.split("=")
                    key = key.strip()
                    if "charge" not in key and key not in ["p", "d", "f"]:
                        if self.types:
                            order.append(f"{index}{self.types[index-1]}_{key}")
                        else:
                            order.append(f"{index}_{key}")
            return order, spilling


    def read_projections(self, filename: str | None=None):
        """Read data that has been created by `projwfc.x`.
        The result contains the bandstructure and Fermi-energy in eV and the number of electrons per cell, that are not covered by pseudopotentials (as that is included in the file).
        The result also the projections on the atomic orbitals.
        That data in projections is order in the shape (k_count, band_count, spin_count, wfc_count).
        The data in overlaps is only returned if overlaps have been computed.
        In that case it has the shape (k_count, spin_count * wfc_count, spin_count * wfc_count).
        Look at `self.read_projections_order_spilling()` for the order of the orbitals.

        Args:
            filename (str, optional): Override the file path for the file `atomic_proj.xml`. Defaults to None.

        Returns:
            tuple: (k_points, bands, (projections, [overlaps]), fermi_energy, electron_count)
        """
        # read the atomic_proj.xml (see https://realpython.com/python-xml-parser/)
        # read in k_points, bands and projections (matrices) (as unnamed O(3) matrices)
        to_eV = 13.605693125 # 1Ry to 1eV
        filename = filename or f"./qe-data/{self.name}.save/atomic_proj.xml"
        with open(filename, 'r') as file:
            from xml.dom.minidom import parse
            document = parse(file)
            #root = document.documentElement
            root = document.getElementsByTagName("PROJECTIONS")[0]
            # <HEADER NUMBER_OF_BANDS="" NUMBER_OF_K-POINTS="" NUMBER_OF_SPIN_COMPONENTS="" NUMBER_OF_ATOMIC_WFC="" NUMBER_OF_ELECTRONS="" FERMI_ENERGY=""/>
            header = document.getElementsByTagName("HEADER")[0]
            fermi_energy = float(header.getAttribute('FERMI_ENERGY')) * to_eV
            band_count = int(header.getAttribute('NUMBER_OF_BANDS'))
            k_count = int(header.getAttribute('NUMBER_OF_K-POINTS'))
            spin_count = int(header.getAttribute('NUMBER_OF_SPIN_COMPONENTS'))
            wfc_count = int(header.getAttribute('NUMBER_OF_ATOMIC_WFC'))
            electron_count = float(header.getAttribute('NUMBER_OF_ELECTRONS')) # with spin even if NUMBER_OF_SPIN_COMPONENTS=1

            k_list = document.getElementsByTagName("K-POINT")
            e_list = document.getElementsByTagName("E")
            proj_list = document.getElementsByTagName("PROJS")

            k_points = np.zeros((k_count, 3))
            bands = np.zeros((k_count, band_count))
            projections = np.zeros((k_count, band_count, spin_count, wfc_count), dtype=complex)

            # now find all the data in the xml file
            for i, (k, e, proj) in enumerate(zip(k_list, e_list, proj_list)):
                k_points[i] = [float(x) for x in k.firstChild.nodeValue.strip().split()]
                bands[i] = [float(x) for x in e.firstChild.nodeValue.strip().split()]
                for wfc in proj.getElementsByTagName("ATOMIC_WFC"):
                    index = int(wfc.getAttribute("index")) - 1
                    spin_index = int(wfc.getAttribute("spin")) - 1
                    data = np.array([float(x) for x in wfc.firstChild.nodeValue.strip().split()])
                    projections[i, :, spin_index, index] = data[::2] + 1j*data[1::2]
            
            overlap_list = document.getElementsByTagName("OVPS")
            if overlap_list:
                overlaps = np.zeros((k_count, spin_count * wfc_count, spin_count * wfc_count), dtype=complex)
                for i, overlap in enumerate(overlap_list):
                    data = np.array([float(x) for x in overlap.firstChild.nodeValue.strip().split()])
                    mat = overlaps[i].ravel()
                    mat += data[::2] + data[1::2]*1j
                return np.array(k_points), np.array(bands) * to_eV, (np.array(projections), np.array(overlaps)), fermi_energy, electron_count
        
        return np.array(k_points), np.array(bands) * to_eV, np.array(projections), fermi_energy, electron_count

    def read_wannier_tb(self, filename=None):
        from . import wannier90_tb_format
        neighbors, params, r_params, _degeneracy, A = wannier90_tb_format.load_tb(f"{self.name}_tb.dat" if filename is None else filename)
        return neighbors, params, r_params

    def read_valence_electrons(self):
        """Get the number of valence electrons in the model by reading the pseudo potential files.

        Returns:
            int: number of valence electrons = number of filled bands
        """
        # Read in all used pseudo potentials and get their valence numbers.
        # Then add them up.
        pp_subset = { t: pp_files[t] for t in self.types }
        valence = dict()
        for atom_type, pp in pp_subset.items():
            with open(f"./pseudo/{pp}", 'r') as file:
                from xml.dom.minidom import parse
                document = parse(file)
                header = document.getElementsByTagName("PP_HEADER")[0]
                z_valence = float(header.getAttribute("z_valence"))
                assert z_valence == int(z_valence)
                valence[atom_type] = int(z_valence)
        return sum((valence[t] for t in self.types))

    
    def set_cutoff_from_pseudopotentials(self):
        """Set the cutoff for the kinetic energy (kinetic_energy_cutoff) from the used pseudo potentials."""
        # Read in all used pseudo potentials and get their suggested cutoffs.
        # Then take the maximum of those.
        pp_subset = { t: pp_files[t] for t in self.types }
        wfc_cutoffs = []
        for atom_type, pp in pp_subset.items():
            with open(f"./pseudo/{pp}", 'r') as file:
                from xml.dom.minidom import parse
                document = parse(file)
                header = document.getElementsByTagName("PP_HEADER")[0]
                wfc_cutoff = float(header.getAttribute("wfc_cutoff"))
                _rho_cutoff = float(header.getAttribute("rho_cutoff")) # currently unused.
                wfc_cutoffs.append(wfc_cutoff)
        self.kinetic_energy_cutoff = max(wfc_cutoffs)


    def read_wavefunctions(self):
        """Read the wavefunctions that have been computed by scf() or nscf()
        k_smpl is in the same order as in read_bands_crystal(...)
        evc is the list of the wavefunction in the format 
        Returns:
            ndarray(N_k, dim): k_smpl in crystal coordinates,
            ndarray(N_k, N_G, dim): g_smpl in crystal coordinates = integer vectors,
            ndarray(N_k, band, N_G, spin_polarisation): evc, coefficients for the plane wave construction of the bloch functions
        """
        # The format is documented in
        # https://gitlab.com/QEF/q-e/-/wikis/Developers/Format-of-data-files
        # and there was someone who already did this implementation
        # https://mattermodeling.stackexchange.com/a/9200
        # so all credit to NehZio

        order = []
        k_smpl = []
        B = None
        g_smpl_list = []
        evc_list = []
        for file in os.listdir(f"./qe-data/{self.name}.save"):
            if not (file.startswith("wfc") and file.endswith(".dat")):
                continue
            with open(f'./qe-data/{self.name}.save/{file}', 'rb') as f:
                # Moves the cursor 4 bytes to the right
                f.seek(4)

                ik = np.fromfile(f, dtype='int32', count=1)[0]
                xk = np.fromfile(f, dtype='float64', count=3)
                ispin = np.fromfile(f, dtype='int32', count=1)[0]
                gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
                scalef = np.fromfile(f, dtype='float64', count=1)[0]

                order.append(ik)
                k_smpl.append(xk)

                # Move the cursor 8 byte to the right
                f.seek(8, 1)

                ngw = np.fromfile(f, dtype='int32', count=1)[0]
                igwx = np.fromfile(f, dtype='int32', count=1)[0]
                npol = np.fromfile(f, dtype='int32', count=1)[0]
                nbnd = np.fromfile(f, dtype='int32', count=1)[0]

                # Move the cursor 8 byte to the right
                f.seek(8, 1)

                b1 = np.fromfile(f, dtype='float64', count=3)
                b2 = np.fromfile(f, dtype='float64', count=3)
                b3 = np.fromfile(f, dtype='float64', count=3)
                B_ = np.array([b1, b2, b3]).T
                assert B is None or np.linalg.norm(B - B_) < 1e-12, "different lattice for wavefunctions at different k-points"
                B = B_

                f.seek(8,1)
                
                mill = np.fromfile(f, dtype='int32', count=3*igwx)
                mill = mill.reshape((igwx, 3))
                # k+G = xk + h*b1 + k*b2 + l*b3
                # mill is in "crystal coordinates", so just return that as G
                # then
                # k+G = B @ (B_inv @ xk + mill)
                # where B_inv @ xk, mill are returned

                evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")

                f.seek(8,1)
                for i in range(nbnd):
                    evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
                    f.seek(8, 1)
                
                # TODO is this order correct???
                evc_list.append(evc.reshape(nbnd, igwx, npol))
                g_smpl_list.append(mill)
        inv_order = np.zeros(len(order), dtype=np.int64)
        inv_order[np.array(order)-1] = np.arange(len(order))
        assert B is not None
        return (np.linalg.inv(B) @ np.array([k_smpl[i] for i in inv_order]).T).T, [g_smpl_list[i] for i in inv_order], [evc_list[i] for i in inv_order]
    
    def read_charge_density(self):
        """Read the charge density from the last run of scf or nscf.

        Returns:
            ndarray(N_G, dim): g_smpl in crystal coordinates = integer vectors,
            ndarray(spin, N_G): charge density fourier coefficients.
        """
        # The format is documented in
        # https://gitlab.com/QEF/q-e/-/wikis/Developers/Format-of-data-files
        with open(f'./qe-data/{self.name}.save/charge-density.dat', 'rb') as f:
            # Moves the cursor 4 bytes to the right
            f.seek(4)

            gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
            ngm_g = np.fromfile(f, dtype='int32', count=1)[0]
            nspin = np.fromfile(f, dtype='int32', count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            b1 = np.fromfile(f, dtype='float64', count=3)
            b2 = np.fromfile(f, dtype='float64', count=3)
            b3 = np.fromfile(f, dtype='float64', count=3)

            f.seek(8,1)
            
            mill = np.fromfile(f, dtype='int32', count=3*ngm_g)
            mill = mill.reshape((ngm_g, 3))
            # k+G = xk + h*b1 + k*b2 + l*b3
            g_smpl = b1[None,:] * mill[:, 0:1] + b2[None,:] * mill[:, 1:2] + b3[None,:] * mill[:, 2:3]

            rho_g = np.zeros((nspin, ngm_g))
            for i in range(nspin):
                rho_g[i,:] = np.fromfile(f, dtype='complex128', count=ngm_g)
                f.seek(8, 1)
        return g_smpl, rho_g

    def read_dos(self):
        """Read the data from the last dos.x run.

        Example call:
        `(energy_smpl, density, states), fermi_energy = material.read_dos()`

        Returns:
            ndarray(3, N_E): tuple with energy samples, density (rho) and states (N) values.
            float: Fermi-Energy in eV
        """
        fermi_energy = None
        with open(f"{self.name}.Dos.dat", "r") as file:
            header = file.readline()
            if "EFermi" in header and header.split(' ')[-1].strip() == "eV":
                fermi_energy = float(header.split(' ')[-2])
        density_of_states = np.loadtxt(f"{self.name}.Dos.dat", unpack=True)
        return density_of_states, fermi_energy

    # ----- QUANTUM ESPRESSO Parameter construction -----

    def _crystal(self):
        cell_params = f"""CELL_PARAMETERS {self.unit}
{self.A[0,0]} {self.A[1,0]} {self.A[2,0]}
{self.A[0,1]} {self.A[1,1]} {self.A[2,1]}
{self.A[0,2]} {self.A[1,2]} {self.A[2,2]}
""" if self.ibrav == 0 else ""
        crystal = cell_params + "ATOMIC_SPECIES\n"
        for t in set(self.types):
            crystal = crystal + f" {t} {element_masses[element_numbers[t]]} {pp_files[t]}\n"
        crystal = crystal + "ATOMIC_POSITIONS crystal\n"
        for b, t in zip(self.basis, self.types):
            assert len(b) == 3
            crystal = crystal + f" {t} {b[0]} {b[1]} {b[2]}\n"
        return crystal[:-1]
    
    # ----- execution of QUANTUM ESPRESSO -----

    def _run(self, program, name):
        os.system(mpi_run + f'{program}{"" if platform == "win32" else ".x"} -nk {self.nk} -nd {self.nd} -nt {self.nt} -in {self.name}.{name}.in | tee {self.name}.{name}.out')
    
    def _run_simple(self, program, args, out_file):
        os.system(f'{program}{"" if platform == "win32" else ".x"} {args} | tee {out_file}')

    def scf(self, k_grid_size, k_grid_size2=None, k_grid_size3=None):
        """Do a self consistent field calculation (solving the Kohn-Sham equations).

        Args:
            k_grid_size (int): k points in x direction of the grid.
            k_grid_size2 (int, optional): k points in y direction of the grid. Defaults to k_grid_size.
            k_grid_size3 (int, optional): k points in z direction of the grid. Defaults to k_grid_size.
        """        
        if k_grid_size2 is None:
            k_grid_size2 = k_grid_size
        if k_grid_size3 is None:
            k_grid_size3 = k_grid_size
        with open(f"{self.name}.scf.in", "w") as file:
            # Notes:
            # The following is just for non relativistic calculations
            # for relativistic calculations, use the additional parameters
            # 
            file.write(f"""&control
    calculation='scf',
    pseudo_dir = './pseudo/',
    outdir='./qe-data/',
    prefix='{self.name}',
    tstress = .true.,
    tprnfor = .true.,
    disk_io='low',
/
&system
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},{f" celldm(1)={self.cell_scale}," if self.ibrav != 0 else ""}
    ecutwfc = {self.kinetic_energy_cutoff},
    occupations='smearing', smearing='{"marzari-vanderbilt" if self.T == 0 else "fermi-dirac"}', degauss={0.02 if self.T == 0 else self.T*6.3336231269e-6},
    {"lspinorb=.true., noncolin=.true.," if self.relativistic else ""}
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
    electron_maxstep = 100,
    mixing_beta = 0.7,
/
{self._crystal()}
{k_grid(k_grid_size, k_grid_size2, k_grid_size3)}
""")
        print(f"running the scf calculation for {self.name}")
        self._run("pw", "scf")
    
    def relax(self, relax_k_grid_size = 4, fix_volume=False):
        """Run an optimisation, which computes the scf using a fixed k_grid
        and then optimizes the cell parameters to get the lowest energy.

        Args:
            relax_k_grid_size (int, optional): Size of the k-grid used for each optimisation step. Defaults to 4.
            fix_volume (bool, optional): If True, the volume of the cell will be kept constant during optimisation. This is useful to match an experimentally determined density at a known pressure. Defaults to False.
        """        
        with open(f"{self.name}.relax.in", "w") as file:
            file.write(f"""
&control
    calculation='vc-relax',
    pseudo_dir = './pseudo/',
    outdir='./qe-data/',
    prefix='{self.name}',
    tstress = .true.,
    tprnfor = .true.,,
    disk_io='low',
/
&system
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},{f" celldm(1)={self.cell_scale}," if self.ibrav != 0 else ""}
    ecutwfc = {self.kinetic_energy_cutoff},
    occupations='smearing', smearing='{"marzari-vanderbilt" if self.T == 0 else "fermi-dirac"}', degauss={0.02 if self.T == 0 else self.T*6.3336231269e-6}
    {"lspinorb=.true., noncolin=.true.," if self.relativistic else ""}
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-5,
    mixing_beta = 0.7,
/
&ions
    ion_dynamics="bfgs",
    ion_temperature="rescaling",
    tempw={self.T},
    tolp=20,
    nraise=1,
/
&cell
    cell_dynamics="bfgs",
    press=0.001,
    press_conv_thr=0.0005,
    cell_dofree="ibrav{"shape" if fix_volume else ""}",
/
{self._crystal()}
{k_grid(relax_k_grid_size)}
""")
        print(f"running the vc-relax calculation for {self.name}")
        self._run("pw", "relax")
    
    def bands(self, band_count, k_points):
        """Calculate band structure using "calculation=bands".
        Can only be run after scf has been run.

        Args:
            k_points (str): the string given to QUANTUM ESPRESSO, which can be generated using `k_grid(...)` or `k_points(...)` or `KPath()` from `kpaths`
            band_count (int): Number of bands to compute and save.
        """
        with open(f"{self.name}.band.in", "w") as file:
            file.write(f"""
&control
    calculation='bands',
    pseudo_dir = './pseudo/',
    outdir='./qe-data/',
    prefix='{self.name}',
    disk_io='low',
/
&system
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},{f" celldm(1)={self.cell_scale}," if self.ibrav != 0 else ""}
    ecutwfc = {self.kinetic_energy_cutoff}, nbnd = {band_count},
    {"lspinorb=.true., noncolin=.true.," if self.relativistic else ""}
/
&electrons
    diagonalization='david',
/
{self._crystal()}
{str(k_points)}
""")
        print(f"running the band-structure calculation for {self.name}")
        self._run("pw", "band")

    def bandx(self):
        """Use QUANTUM ESPRESSO's bands.x to convert bands output to workable data.
        -> slow and buggy... for k-grids use `nscf` and `read_bands_crystal` for direct access instead.
        """
        with open(f"{self.name}.bandx.in", "w") as file:
            file.write(f"""
&BANDS
prefix='{self.name}'
outdir='./qe-data/'
filband='{self.name}.Bandx.dat'
/
""")
        # doesn't completely work for high resolutions...
        print(f"converting data for {self.name} to a plottable format")
        self._run("bands", "bandx")

    def dos(self):
        """Compute the fermi energy from the density of states (DoS)"""
        with open(f"./{self.name}.dos.in", "w") as file:
            file.write(f"""
&control
    calculation='nscf',
    tstress = .true.,
    tprnfor = .true.,
/
&dos
    prefix='{self.name}',
    bz_sum='tetrahedra_opt',
    outdir='./qe-data/',
    deltae =  1e-02,
    fildos = '{self.name}.Dos.dat',
/
""")
        self._run("dos", "dos")

    def nscf(self, k_points: str, band_count: int):
        """Calculate band structure using "calculation=nscf".
        Can only be run after scf has been run.

        Args:
            k_points (str): the string given to QUANTUM ESPRESSO, which can be generated using `k_grid(...)` or `k_points(...)` or `KPath()` from `kpaths`
            band_count (int): Number of bands to compute and save.
        """

        with open(f"./{self.name}.nscf.in", "w") as file:
            file.write(f"""\
&control
    calculation='nscf',
    pseudo_dir = './pseudo/',
    outdir='./qe-data/',
    prefix='{self.name}',
    tstress = true,
    tprnfor = true,
    disk_io='low',
    verbosity='high',
/
&system
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},{f" celldm(1)={self.cell_scale}," if self.ibrav != 0 else ""}
    ecutwfc = {self.kinetic_energy_cutoff},
    nbnd={band_count},
    force_symmorphic = true,
    occupations='smearing', smearing='{"marzari-vanderbilt" if self.T == 0 else "fermi-dirac"}', degauss={0.02 if self.T == 0 else self.T*6.3336231269e-6}
    {"lspinorb=.true., noncolin=.true.," if self.relativistic else ""}
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
/
{self._crystal()}
{str(k_points)}
""")
        self._run("pw", "nscf")

    def nscf_nosym(self, k_points, band_count):
        """Non self consistent field calculation with nosym=true, noinv=true for processing by further tools like wannier90."""
        with open(f"./{self.name}.nscf.in", "w") as file:
            file.write(f"""
&control
    calculation='nscf',
    pseudo_dir = './pseudo/',
    outdir='./qe-data/',
    prefix='{self.name}',
    tstress = true,
    tprnfor = true,
    disk_io='low',
    verbosity='high',
/
&system
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},{f" celldm(1)={self.cell_scale}," if self.ibrav != 0 else ""}
    ecutwfc = {self.kinetic_energy_cutoff},
    nosym = true,
    noinv = true,
    nbnd={band_count}
    occupations='smearing', smearing='{"marzari-vanderbilt" if self.T == 0 else "fermi-dirac"}', degauss={0.02 if self.T == 0 else self.T*6.3336231269e-6}
    {"lspinorb=.true., noncolin=.true.," if self.relativistic else ""}
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
/
{self._crystal()}
{str(k_points)}
""")
        self._run("pw", "nscf")

    def crystal_wannier(self):
        crystal = f"""begin unit_cell_cart
{self.unit}
{self.A[0,0]} {self.A[1,0]} {self.A[2,0]}
{self.A[0,1]} {self.A[1,1]} {self.A[2,1]}
{self.A[0,2]} {self.A[1,2]} {self.A[2,2]}
end unit_cell_cart

begin atoms_frac
"""
        for b, t in zip(self.basis, self.types):
            assert len(b) == 3
            crystal = crystal + f"{t} {b[0]} {b[1]} {b[2]}\n"
        crystal = crystal + "end atoms_frac"
        return crystal

    def dielectric_function(self, nw=500):
        # compute the dielectric function
        with open(f"{self.name}.epsilon.in", "w") as file:
            file.write(f"""
&inputpp
  prefix='{self.name}'
  outdir='./qe-data/'
  calculation = "eps"
/

&energy_grid
  smeartype = "gauss"
  intersmear = 0.2
  wmin =  0.0
  wmax = {self.kinetic_energy_cutoff}
  nw = {nw}
/
""")
        # doesn't completely work for high resolutions...
        print(f"computing dielectric function for {self.name}")
        os.system(mpi_run + f'epsilon{"" if platform == "win32" else ".x"} < {self.name}.epsilon.in | tee {self.name}.epsilon.out')
    
    def read_dielectric_function(self):
        # read the dielectric function after having computed it with `dielectric_function`
        data_r = np.loadtxt(f'epsr_{self.name}.dat', skiprows=3)
        data_i = np.loadtxt(f'epsi_{self.name}.dat', skiprows=3)
        return data_r[:,0], data_r[:,1:] + data_i[:,1:] * 1j

    def projections(self, lwrite_overlaps=False):
        """Compute the projections of the wave functions onto atomic orbitals using `projwfc.x`.
        The results can be read in using `self.read_projections()`"""
        with open(f"{self.name}.projwfc.in", "w") as file:
            file.write(f"""
&projwfc
prefix='{self.name}'
outdir='./qe-data/'
lwrite_overlaps={lwrite_overlaps}
/
""")
        # doesn't completely work for high resolutions...
        print(f"computing projections onto atomic orbitals for {self.name}")
        self._run("projwfc", "projwfc")

    def k_points_wannier(self, points):
        k_points = "begin kpoints\n"
        for x, y, z in points:
            k_points = k_points + f"{x} {y} {z}\n"
        k_points = k_points + "end kpoints"
        return k_points
    
    def k_path_wannier(self, kpath, A_norm):
        k_points = "begin kpoint_path\n"
        prev = None
        for i, name in zip(kpath.indices, kpath.names):
            # x, y, z in crystal coordinates (kpath is given in normalized reciprocal coordinates)
            x, y, z = A_norm.T @ np.asarray(kpath[i])
            this = f"{name} {x} {y} {z}"
            if prev is not None:
                k_points = k_points + prev + " " + this + "\n"
            prev = this
        k_points = k_points + "end kpoint_path"
        return k_points
    
    # prepare the wannierization parameters and check them
    # wannier_count is the number of bands in the result if disentanglement is used. Otherwise this is choosen automatically to use all bands.
    # run this after nscf_nosym()
    def prepare_wannier(self, max_disentangle_energy, max_frozen_energy, wannier_count=None, grid_size=None, iterations=100, disentangle_iterations=500, projections="random", exclude_bands=None, symmetry=False, write_tb=False, write_wannier_functions=False):
        # the difficult part is the "projections" part
        # that part is about which orbitals are used where
        # using my HamiltonianSymmetry I can accumulate all the information for it.
        # otherwise I would need to add yet another way to construct it...
        # PROBLEM: This is the difficult part! and it is just for the starting state!
        # For now, just use "random" to get it running
        #
        # Note: write_tb is REALLY slow! It took almost the entire computation in my case!
        # or maybe it was wannier_plot which was slow?
        # calling it "write_..." is misleading, as it's actually a flag "compute_..."
        k_points, bands, _, _ = self.read_bands_crystal(incomplete=True)
        if grid_size is None:
            assert symmetry, "can only infer the grid_size when no symmetry is used."
            grid_size = round(np.cbrt(len(k_points)))
            assert len(k_points) == grid_size**3, "grid from data is likely reduced by symmetry"
        else:
            k_points = np.stack(np.meshgrid(*(np.linspace(0, 1.0, grid_size, endpoint=False),)*3), axis=-1).reshape(-1, 3)
        if wannier_count is None:
            wannier_count = len(bands[0])
        with open(f"{self.name}.win", "w") as file:
            file.write(f"""
num_bands = {len(bands[0])}
num_wann = {wannier_count}
num_iter = {iterations}
conv_tol = 1.0e-10 ! = default value
conv_window = 4
trial_step = 1.0 ! line search, decrease if the wannierization doesn't converge

iprint = 2
timing_level = 2
num_dump_cycles = 10
num_print_cycles = 10

dis_win_max = {max_disentangle_energy:.3f}
!dis_win_min = 11.0
{f"dis_froz_max = {max_frozen_energy:.3f}" if not symmetry else ""}
!dis_froz_min = 11.0
dis_num_iter = {disentangle_iterations}
dis_mix_ratio   = 1.0
dis_conv_tol = 1e-6
dis_conv_window = 4

length_unit = { {'angstrom': 'Ang', 'bohr': 'Bohr'}[self.unit]}

spinors = false
!auto_projections = true !there is a warning about this not always working with pw2wannier90
!use_bloch_phases = true ! doesn't work :(

begin projections
{projections}
end projections

site_symmetry = {"true\nsymmetrize_eps=1d-9" if symmetry else "false"}
write_hr = true
write_tb = {"true" if write_tb else "false"}
write_xyz = true
translate_home_cell = false

wannier_plot = {"true" if write_wannier_functions else "false"}
!wannier_plot_supercell = 3
bands_plot = false

{f"exclude_bands={exclude_bands}" if exclude_bands else ""}

{self.crystal_wannier()}

mp_grid = {grid_size} {grid_size} {grid_size}

{self.k_points_wannier(k_points)}
""")
        # generate a list of required overlaps (written to {name}.nnkp)
        # for some reason mpirun doesn't work...
        #os.system(mpi_run + f"wannier90.x -pp {self.name}")
        self._run_simple('wannier90', f"-pp {self.name}", f"{self.name}.wout")

    # compute the overlaps of the wavefunctions from the nscf calculation, to be used by wannierization
    # use this after using prepare_wannier()
    def overlaps_for_wannier(self, use_sym=False):
        # Note: don't write the UNK files as they work really poorly with normal filesystems (there is one for each k-point!)
        with open(f"{self.name}.pw2wan.in", "w") as file:
            file.write(f"""
&inputpp 
   outdir='./qe-data/'
   prefix = '{self.name}'
   seedname = '{self.name}'
   !spin_component = 'none'
   wan_mode = 'standalone'
   write_mmn = true
   write_amn = true
   {"write_dmn = true\nread_sym = true" if use_sym else ""}
/
""")
        self._run("pw2wannier90", "pw2wan")
    
    # compute the maximally localized wave functions (MLWFs)
    # use this after using overlaps_for_wannier()
    def wannier(self):
        # written to {name}.mmn and {name}.amn
        # parallel execution doesn't work for some reason...
        #os.system(mpi_run + f"wannier90.x {self.name}")
        self._run_simple('wannier90', f"{self.name}", f"{self.name}.wout")

    # rerun wannier90 to get the bands of the last computed wannier90 tight binding on a kpath
    # bands_num_points: then the number of points along the first section of the bandstructure plot given by kpath
    # the actual plot can be done using plot_wannier_bands()
    def prepare_plot_wannier_bands(self, kpath, A_norm, wannier_count=None, grid_size=None, bands_num_points=None):
        if bands_num_points is None:
            bands_num_points = kpath.indices[1]
         # TODO read from prior wannier90 run
         # for no reason whatsoever wannier90 needs all of this data, just to evaluate its tb model.
        k_points, bands, _, _ = self.read_bands_crystal(incomplete=True)
        if grid_size is None:
            grid_size = round(np.cbrt(len(k_points)))
            assert len(k_points) == grid_size**3, "grid from data is likely reduced by symmetry"
        else:
            k_points = np.stack(np.meshgrid(*(np.linspace(0, 1.0, grid_size, endpoint=False),)*3), axis=-1).reshape(-1, 3)
        if wannier_count is None:
            wannier_count = len(bands[0])
        with open(f"{self.name}.win", "w") as file:
            file.write(f"""
num_bands = {len(bands[0])}
num_wann = {wannier_count}
restart = plot
num_iter = 0
dis_num_iter = 0

length_unit = { {'angstrom': 'Ang', 'bohr': 'Bohr'}[self.unit]}

write_hr = false
write_tb = false
write_xyz = false

wannier_plot = false
!wannier_plot_supercell = 3
bands_plot = true
bands_num_points = {bands_num_points}

{self.k_path_wannier(kpath, A_norm)}

{self.crystal_wannier()}

mp_grid = {grid_size} {grid_size} {grid_size}

{self.k_points_wannier(k_points)}
""")
        # generate a list of required overlaps (written to {name}.nnkp)
        # for some reason mpirun doesn't work...
        #os.system(mpi_run + f"wannier90.x -pp {self.name}")
        self._run_simple('wannier90', self.name, f"{self.name}.wout")

    def plot_wannier_bands(self):
        from matplotlib import pyplot as plt
        data = np.loadtxt(f"{self.name}_band.dat", unpack=True)
        plt.plot(*data, '.')

    # show Fermi surface using xcrysden
    def fermi_surface(self):
        # https://pranabdas.github.io/espresso/hands-on/fermi-surface/
        # for a non python example
        with open(f"{self.name}.fs.in", "w") as file:
            file.write(f"""
&fermi
prefix='{self.name}'
outdir='./qe-data/'
/
""")
        self._run("fs", "fs")
        os.system(f"xcrysden --bxsf {self.name}_fs.bxsf > /dev/null")


def from_disk(name: str, read_output=True, prepare_pseudopotentials=False) -> QECrystal:
    """Read all parameters for the creation of a QECrystal object from
    the xml file that was created in the last run of QE.

    Args:
        name (str): Prefix for the material. The corresponding file is located at `qe-data/{name}.xml`
        read_output (bool, optional): If True, the output section in the xml is read instead of the input section. Defaults to True.
        prepare_pseudopotentials (bool, optional): If True, the pseudopotentials for this crystal are automatically prepared i.e. downloaded.
            If this is False, the pseudopotentials can be downloaded later using qe_prepare(). Defaults to False.

    Returns:
        QECrystal: The object with all parameters filled in from a previous run.
    """
    crystal = QECrystal(name, np.eye(3), [], [], 0.0)
    basis = []
    types = []
    filename = f"./qe-data/{name}.xml"
    bohr_to_angstrom = 0.52917721 # bohr_radius in Angstrom
    with open(filename, 'r') as file:
        from xml.dom.minidom import parse
        document = parse(file)
        root = document.getElementsByTagName("qes:espresso")[0]
        parallel_info = root.getElementsByTagName("parallel_info")[0]
        nk = int(parallel_info.getElementsByTagName("npool")[0].firstChild.nodeValue)
        ni = int(parallel_info.getElementsByTagName("nprocs")[0].firstChild.nodeValue) # not sure about this one, missing online documentation...
        nd = int(parallel_info.getElementsByTagName("ndiag")[0].firstChild.nodeValue)
        nt = int(parallel_info.getElementsByTagName("ntasks")[0].firstChild.nodeValue)
        crystal.set_multitasking_parameters(nk=nk, ni=ni, nd=nd, nt=nt)
        if read_output:
            section = root.getElementsByTagName("output")[0]
        else:
            section = root.getElementsByTagName("input")[0]
        types_elem = section.getElementsByTagName("atomic_species")[0]
        positions = section.getElementsByTagName("atomic_positions")[0]
        for atom in types_elem.getElementsByTagName("species"):
            atom_type = atom.getAttribute("name")
            # ignore atom mass. That means that special isotopes get ignored here...
            _atom_mass = atom.getElementsByTagName("mass")[0].firstChild.nodeValue
            # add pseudo potential names to the dictionary
            pseudo_file = atom.getElementsByTagName("pseudo_file")[0].firstChild.nodeValue
            pp_files[atom_type] = pseudo_file
        # This A is in bohr_radius -> convert it to Angstrom
        a1 = [float(x) for x in section.getElementsByTagName("a1")[0].firstChild.nodeValue.strip().split()]
        a2 = [float(x) for x in section.getElementsByTagName("a2")[0].firstChild.nodeValue.strip().split()]
        a3 = [float(x) for x in section.getElementsByTagName("a3")[0].firstChild.nodeValue.strip().split()]
        A = np.array([a1, a2, a3]).T
        if crystal.unit == "angstrom":
            A *= bohr_to_angstrom
        crystal.A = A
        for atom in positions.getElementsByTagName("atom"):
            atom_type = atom.getAttribute("name")
            atom_pos = [float(x) for x in atom.firstChild.nodeValue.strip().split()]
            # convert atom_pos to crystal coordinates, as those are used by the class
            atom_pos = np.linalg.inv(A) @ atom_pos
            basis.append(atom_pos)
            types.append(atom_type)
        crystal.basis = np.array(basis)
        crystal.types = np.array(types)
        crystal.kinetic_energy_cutoff = float(section.getElementsByTagName("ecutwfc")[0].firstChild.nodeValue)
        #ecutrho = float(section.getElementsByTagName("ecutrho")[0].firstChild.nodeValue)
        smearing = section.getElementsByTagName("smearing")
        if smearing:
            if smearing[0].firstChild.nodeValue.strip() == "fd": # TODO check the shorthand...
                # TODO check unit!
                crystal.T = float(smearing[0].getAttribute("degauss")) / 6.3336231269e-6
    if prepare_pseudopotentials:
        qe_prepare()
    return crystal
