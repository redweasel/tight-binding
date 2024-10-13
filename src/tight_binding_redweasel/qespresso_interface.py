import os
from sys import platform
import numpy as np

# simple interface for QUANTUM ESPRESSO
# assuming quantum espresso is installed and in the PATH

ibrav_map = { "none": 0, "sc": 1, "fcc": 2, "bcc": 3, "hex": 4, "tri": 5, "monoclinic": 12 }

pp_files = {}

element_masses = [1.008, 4.0026, 7.0, 9.012183, 10.81, 12.011, 14.007, 15.999, 18.99840316, 20.18, 22.9897693, 24.305, 26.981538, 28.085, 30.973762, 32.07, 35.45, 39.9, 39.0983, 40.08, 44.95591, 47.867, 50.9415, 51.996, 54.93804, 55.84, 58.93319, 58.693, 63.55, 65.4, 69.723, 72.63, 74.92159, 78.97, 79.9, 83.8, 85.468, 87.62, 88.90584, 91.22, 92.90637, 95.95, 96.90636, 101.1, 102.9055, 106.42, 107.868, 112.41, 114.818, 118.71, 121.76, 127.6, 126.9045, 131.29, 132.905452, 137.33, 138.9055, 140.116, 140.90766, 144.24, 144.91276, 150.4, 151.964, 157.2, 158.92535, 162.5, 164.93033, 167.26, 168.93422, 173.05, 174.9668, 178.49, 180.9479, 183.84, 186.207, 190.2, 192.22, 195.08, 196.96657, 200.59, 204.383, 207.0, 208.9804, 208.98243, 209.98715, 222.01758, 223.01973, 226.02541, 227.02775, 232.038, 231.03588, 238.0289, 237.048172, 244.0642, 243.06138, 247.07035, 247.07031, 251.07959, 252.083, 257.09511, 258.09843, 259.101, 266.12, 267.122, 268.126, 269.128, 270.133, 269.1336, 277.154, 282.166, 282.169, 286.179, 286.182, 290.192, 290.196, 293.205, 294.211, 295.216]
element_numbers = {'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 'Br': 34, 'Kr': 35, 'Rb': 36, 'Sr': 37, 'Y': 38, 'Zr': 39, 'Nb': 40, 'Mo': 41, 'Tc': 42, 'Ru': 43, 'Rh': 44, 'Pd': 45, 'Ag': 46, 'Cd': 47, 'In': 48, 'Sn': 49, 'Sb': 50, 'Te': 51, 'I': 52, 'Xe': 53, 'Cs': 54, 'Ba': 55, 'La': 56, 'Ce': 57, 'Pr': 58, 'Nd': 59, 'Pm': 60, 'Sm': 61, 'Eu': 62, 'Gd': 63, 'Tb': 64, 'Dy': 65, 'Ho': 66, 'Er': 67, 'Tm': 68, 'Yb': 69, 'Lu': 70, 'Hf': 71, 'Ta': 72, 'W': 73, 'Re': 74, 'Os': 75, 'Ir': 76, 'Pt': 77, 'Au': 78, 'Hg': 79, 'Tl': 80, 'Pb': 81, 'Bi': 82, 'Po': 83, 'At': 84, 'Rn': 85, 'Fr': 86, 'Ra': 87, 'Ac': 88, 'Th': 89, 'Pa': 90, 'U': 91, 'Np': 92, 'Pu': 93, 'Am': 94, 'Cm': 95, 'Bk': 96, 'Cf': 97, 'Es': 98, 'Fm': 99, 'Md': 100, 'No': 101, 'Lr': 102, 'Rf': 103, 'Db': 104, 'Sg': 105, 'Bh': 106, 'Hs': 107, 'Mt': 108, 'Ds': 109, 'Rg': 110, 'Cn': 111, 'Nh': 112, 'Fl': 113, 'Mc': 114, 'Lv': 115, 'Ts': 116, 'Og': 117}
element_names = ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium', 'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine', 'Xenon', 'Cesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium', 'Promethium', 'Samarium', 'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium', 'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury', 'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine', 'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium', 'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium', 'Curium', 'Berkelium', 'Californium', 'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium', 'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium', 'Bohrium', 'Hassium', 'Meitnerium', 'Darmstadtium', 'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium', 'Moscovium', 'Livermorium', 'Tennessine', 'Oganesson']

mpi_run = "mpirun --use-hwthread-cpus "

def qe_prepare(pseudo_potential_files_dict):
    if not os.path.exists("./qe-data"):
        os.mkdir("./qe-data")
    if not os.path.exists("./pseudo"):
        os.mkdir("./pseudo")

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
            os.chdir("./pseudo")
            os.system("wget https://pseudopotentials.quantum-espresso.org/upf_files/" + file)
            os.chdir("..")
    global pp_files
    pp_files = pseudo_potential_files_dict

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

class QECrystal:
    # name - the prefix for the files
    # A - the lattice matrix, such that the lattice is A @ n_vec (each column is a lattice vector)
    def __init__(self, name, A, basis, types, kinetic_energy_cutoff, unit="angstrom"):
        assert unit in ("angstrom", "bohr")
        self.name = name
        self.unit = unit
        self.A = np.asarray(A)
        self.basis = np.asarray(basis)
        self.types = np.asarray(types)
        assert len(self.types) == len(self.basis), "every basis atom needs a matching type"
        #self.ibrav = ibrav_map[symmetry]
        self.ibrav = 0 # use CELL_PARAMETERS instead!
        self.cell_scale = 1.0 # = celldm(1) from pw.x
        self.T = 0.0 # Temperature in Kelvin
        self.kinetic_energy_cutoff = kinetic_energy_cutoff
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

    # 3d plot of the crystal structure
    def plot_crystal(self, repeat=1, turntable=14, elevation=35):
        assert self.ibrav == 0, "only implemented for ibrav = 0"
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
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
        # Hack: in old matplotlib versions .set_aspect("equal") doesn't work for 3D yet.
        def axisEqual3D(ax):
            extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x [{self.unit}]")
        ax.set_ylabel(f"y [{self.unit}]")
        ax.set_zlabel(f"z [{self.unit}]")
        #axisEqual3D(ax)
        ax.legend()
    
    def mass_density(self) -> float:
        """Compute the mass density based on the given crystal lattice.
        This is very useful to sanity check the input parameters.

        Returns:
            float: mass density in kg/m^3
        """
        nuclei_mass = 1.66053907e-27 * sum((element_masses[element_numbers[t]] for t in self.types))
        electron_mass = 9.1093837e-31 * sum((element_numbers[t] for t in self.types))
        return (electron_mass + nuclei_mass) / np.linalg.det(self.A*1e-10)

    # ----- reading results of QUANTUM ESPRESSO -----

    # read the data from Bandx.dat instead of the plottable data
    def read_bandx_raw(self):
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
    
    # This function extracts the high symmetry points x from the output of bandx.out
    # in also find the output file for the band structure and returns the content of that.
    def read_bandx(self): 
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

    # this is my bands plot function for the bands sorted by bandx
    def plot_bands(self):
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
        fermi_energy = self.read_bands()[4]
        plt.axhline(fermi_energy, 'r')
        plt.ylabel("Energy in eV")
        plt.title(f"Bandstructure of {self.name}")
        return sym_x

    # read the data that has been computed (either by scf(), or by bands())
    # returns k_points, weights, bands, symmetries, fermi_energy
    def read_bands(self, incomplete=False):
        # read directly from the bands xml file (see https://realpython.com/python-xml-parser/)
        # read in k_points, eigenvalues and symmetries (as unnamed O(3) matrices)
        k_points = []
        bands = []
        weights = []
        S = []
        S_trans = []
        fermi_energy = None
        to_eV = 27.21138625 # from Hartree = 2Ry to 1eV
        filename = f"./qe-data/{self.name}.save/data-file-schema.xml" if incomplete else f"./qe-data/{self.name}.xml"
        with open(filename, 'r') as file:
            from xml.dom.minidom import parse
            document = parse(file)
            #root = document.documentElement
            root = document.getElementsByTagName("qes:espresso")[0]
            assert root.getAttribute("Units") == "Hartree atomic units" # only accept these units for now
            # now find all the data in the xml file
            symmetry_list = document.getElementsByTagName("symmetry")
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
            ks_energies = document.getElementsByTagName("ks_energies")
            for ks_e in ks_energies:
                k_point = ks_e.getElementsByTagName("k_point")[0]
                eigenvalues = ks_e.getElementsByTagName("eigenvalues")[0]
                k_points.append([float(x) for x in k_point.firstChild.nodeValue.strip().split()])
                weights.append(float(k_point.getAttribute("weight")))
                bands.append([float(x) * to_eV for x in eigenvalues.firstChild.nodeValue.strip().split()])
            fermi_energy_node = document.getElementsByTagName("fermi_energy")[0]
            fermi_energy = float(fermi_energy_node.firstChild.nodeValue.strip()) * to_eV
        return np.array(k_points), np.array(weights), np.array(bands), np.array(S), fermi_energy
    
    # read the data that has been computed (either by scf(), or by bands())
    # returns k_points, bands, symmetries, fermi_energy, real_cell_matrix
    # k and symmetries are in crystal coordinates.
    # That means k_points will be in [0,1[^3
    # symmetries are the crystal space symmetries, meaning they are orthogonal and equal for real and reciprocal space.
    def read_bands_crystal(self, incomplete=False):
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
            b1 = [float(x) for x in document.getElementsByTagName("b1")[0].firstChild.nodeValue.strip().split()]
            b2 = [float(x) for x in document.getElementsByTagName("b2")[0].firstChild.nodeValue.strip().split()]
            b3 = [float(x) for x in document.getElementsByTagName("b3")[0].firstChild.nodeValue.strip().split()]
            reciprocal = np.array([b1, b2, b3]).T
            inv_reciprocal = np.linalg.inv(reciprocal)

            # This A is in bohr_radius -> convert it to Angstrom
            a1 = [float(x) for x in document.getElementsByTagName("a1")[0].firstChild.nodeValue.strip().split()]
            a2 = [float(x) for x in document.getElementsByTagName("a2")[0].firstChild.nodeValue.strip().split()]
            a3 = [float(x) for x in document.getElementsByTagName("a3")[0].firstChild.nodeValue.strip().split()]
            A = np.array([a1, a2, a3]).T
            if self.unit == "angstrom":
                A *= bohr_to_angstrom
            # TODO consider returning "reciprocal" if it is too difficult to recontruct it
            # inv_reciprocal and A.T are collinear, but with what factor???
            #assert np.linalg.norm(inv_reciprocal - (A.T / np.linalg.norm(A[0]))) < 1e-7, f"reciprocal lattice doesn't match real lattice... This problem comes from Quantum Espresso. The compared matrices were\n{inv_reciprocal}\nand\n{A.T / np.linalg.norm(A[0])}"
            # now find all the data in the xml file
            symmetry_list = document.getElementsByTagName("symmetry")
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
            ks_energies = document.getElementsByTagName("ks_energies")
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
                # ???
                S = np.eye(3)[None,...]
            # symmetries should be orthogonal in this basis. TODO sometimes they are not??? Related to fractional symmetries...
            # assert np.linalg.norm(np.einsum("nij,nik->njk", S, S) - np.eye(len(S[0]))) < 1e-5, "symmetries in crystal space are not orthogonal"
            assert np.linalg.norm(np.round(S) - S) < 1e-8, "symmetries in crystal space are not integers"
            assert len(S_trans) == len(S), "every symmetry needs an associated translational part if one of them has one."
            # only return the symmetries without translational part
            if not all((np.linalg.norm(t) < 1e-10 for t in S_trans)):
                # adding translational part as projective part!
                S = [np.block([[s, t[:,None]], [np.zeros((1, 3)), np.eye(1)]]) for s, t in zip(S, S_trans)]
            # transform S from the real space symmetries to the reciprocal space symmetries.
            S = np.swapaxes(S, -1, -2)
            S = np.linalg.inv(S)
            fermi_energy_node = document.getElementsByTagName("fermi_energy")[0]
            fermi_energy = float(fermi_energy_node.firstChild.nodeValue.strip()) * to_eV
        return np.array(k_points), np.array(bands), S, fermi_energy, A
    
    # read the data that has been computed (either by scf(), or by bands())
    # returns k_points, weights, bands, symmetries, fermi_energy
    def read_projections(self, filename=None):
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
                return np.array(k_points), np.array(bands) * to_eV, np.array(projections), np.array(overlaps), fermi_energy, electron_count
        
        return np.array(k_points), np.array(bands) * to_eV, np.array(projections), fermi_energy, electron_count

    def read_wannier_tb(self, filename=None):
        from . import wannier90_tb_format as tb_fmt
        neighbors, params, r_params, degeneracy, A = tb_fmt.load_tb(f"{self.name}_tb.dat" if filename is None else filename)
        return neighbors, params, r_params

    def read_wavefunctions(self):
        """Read the wavefunctions that have been computed by scf() or nscf()
        k_smpl is in the same order as in read_bands(...)
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

    def crystal(self):
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
    
    def k_points(self, points, unit):
        k_points = f"K_POINTS {{{unit}}}\n{len(points)}\n"
        for x, y, z in points:
            k_points = k_points + f"{x} {y} {z} 1\n"
        return k_points

    def k_grid(self, size, size2=None, size3=None):
        if size2 is None:
            size2 = size
        if size3 is None:
            size3 = size
        return f"K_POINTS (automatic)\n{size} {size2} {size3} 0 0 0"

    # ----- execution of QUANTUM ESPRESSO -----

    def scf(self, k_grid_size, k_grid_size2=None, k_grid_size3=None):
        if k_grid_size2 is None:
            k_grid_size2 = k_grid_size
        if k_grid_size3 is None:
            k_grid_size3 = k_grid_size
        with open(f"{self.name}.scf.in", "w") as file:
            # Notes:
            # The following is just for non relativistic calculations
            # for relativistic calculations, use the additional parameters
            # lspinorb=.true., noncolin=.true.
            file.write(f"""
&control
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
    occupations='smearing', smearing='{"marzari-vanderbilt" if self.T == 0 else "fermi-dirac"}', degauss={0.02 if self.T == 0 else self.T*6.3336231269e-6}
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
    electron_maxstep = 100,
    mixing_beta = 0.7,
/
{self.crystal()}
{self.k_grid(k_grid_size, k_grid_size2, k_grid_size3)}
""")
        print(f"running the scf calculation for {self.name}")
        os.system(mpi_run + f'pw{"" if platform == "win32" else ".x"} -nk {self.nk} -nd {self.nd} -nt {self.nt} < {self.name}.scf.in | tee {self.name}.scf.out')
    
    def relax(self, relax_k_grid_size = 4, fix_volume=False):
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
{self.crystal()}
{self.k_grid(relax_k_grid_size)}
""")
        print(f"running the vc-relax calculation for {self.name}")
        os.system(mpi_run + f'pw{"" if platform == "win32" else ".x"} -nk {self.nk} -nd {self.nd} -nt {self.nt} < {self.name}.relax.in | tee {self.name}.relax.out')
    
    # calculate band structure
    # k_points is the string given to QUANTUM ESPRESSO, which can be generated using
    # self.k_grid(...) or self.k_points(...) or KPath() from kpaths.py
    # can only be run after scf has been run.
    def bands(self, band_count, k_points):
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
/
&electrons
    diagonalization='david',
/
{self.crystal()}
{str(k_points)}
""")
        print(f"running the band-structure calculation for {self.name}")
        os.system(mpi_run + f'pw{"" if platform == "win32" else ".x"} -nk {self.nk} -nd {self.nd} -nt {self.nt} < {self.name}.band.in | tee {self.name}.band.out')

    # use QUANTUM ESPRESSO's bands.x to convert bands output to workable data.
    # -> slow and buggy... for k-grids use my function for direct access instead
    def bandx(self):
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
        os.system(mpi_run + f'bands{"" if platform == "win32" else ".x"} < {self.name}.bandx.in | tee {self.name}.bandx.out')

    # compute the fermi energy from the density of states (dos)
    def dos(self):
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
        os.system(mpi_run + f'dos{"" if platform == "win32" else ".x"} < {self.name}.dos.in > {self.name}.dos.out')

    # calculate band structure
    # k_points is the string given to QUANTUM ESPRESSO, which can be generated using
    # self.k_grid(...) or self.k_points(...) or KPath() from kpaths.py
    # can only be run after scf has been run.
    def nscf(self, k_points, band_count):
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
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
/
{self.crystal()}
{str(k_points)}
""")
        os.system(mpi_run + f'pw{"" if platform == "win32" else ".x"} -nk {self.nk} -nd {self.nd} -nt {self.nt} < {self.name}.nscf.in | tee {self.name}.nscf.out')

    # non self consistent field calculation with nosym=true, noinv=true for processing by further tools
    def nscf_nosym(self, k_points, band_count):
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
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
/
{self.crystal()}
{str(k_points)}
""")
        os.system(mpi_run + f'pw{"" if platform == "win32" else ".x"} -nk {self.nk} -nd {self.nd} -nt {self.nt} < {self.name}.nscf.in | tee {self.name}.nscf.out')

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

    def epsilon(self, nw=500):
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
    
    def read_epsilon(self):
        data_r = np.loadtxt(f'epsr_{self.name}.dat', skiprows=3)
        data_i = np.loadtxt(f'epsi_{self.name}.dat', skiprows=3)
        return data_r[:,0], data_r[:,1:] + data_i[:,1:] * 1j

    def k_points_wannier(self, points):
        k_points = "begin kpoints\n"
        for x, y, z in points:
            k_points = k_points + f"{x} {y} {z}\n"
        k_points = k_points + "end kpoints"
        return k_points
    
    # prepare the wannierization parameters and check them
    # wannier_count is the number of bands in the result if disentanglement is used. Otherwise this is choosen automatically to use all bands.
    # run this after nscf_nosym()
    def prepare_wannier(self, wannier_count=None, grid_size=None, iterations=100, projections="random"):
        # the difficult part is the "projections" part
        # that part is about which orbitals are used where
        # using my HamiltonianSymmetry I can accumulate all the information for it.
        # otherwise I would need to add yet another way to construct it...
        # PROBLEM: This is the difficult part! and it is just for the starting state!
        # For now, just use "random" to get it running
        #
        # TODO make the kpath for the plotting chooseable using the kpath module, or
        # don't use bands_plot and implement that part myself using the tb model.
        k_points, weights, bands, symmetries, fermi_energy = self.read_bands(incomplete=True)
        if grid_size is None:
            grid_size = round(np.cbrt(len(k_points)))
        else:
            k_points = np.stack(np.meshgrid(*(np.linspace(0, 1.0, grid_size, endpoint=False),)*3), axis=-1).reshape(-1, 3)
        #assert len(k_points) == grid_size**3, "grid from data is reduced by symmetry, which is strangely not allowed for this step"
        if wannier_count is None:
            wannier_count = len(bands[0])
        with open(f"{self.name}.win", "w") as file:
            file.write(f"""
num_bands = {len(bands[0])}
num_wann = {wannier_count}
num_iter = {iterations}
conv_tol = 1.0e-10 ! = default value
conv_window = 4
trial_step = 3.0 ! line search, increase if the wannierization doesn't converge

iprint = 2
num_dump_cycles = 10
num_print_cycles = 10

dis_win_max = 18.0
dis_win_min = 11.0
!dis_froz_max = 13.4
!dis_froz_min = 11.0

spinors = false
!auto_projections = true !there is a warning about this not always working with pw2wannier90
!use_bloch_phases = true ! doesn't work :(

begin projections
{projections}
end projections

site_symmetry = true
!write_hr = true
write_tb = true
write_xyz = true

wannier_plot = true
wannier_plot_supercell = 3
bands_plot = true

begin kpoint_path
L 0.50000 0.50000 0.5000 G 0.00000 0.00000 0.0000
G 0.00000 0.00000 0.0000 X 0.50000 0.00000 0.5000
end kpoint_path

!exclude_bands=

{self.crystal_wannier()}

mp_grid = {grid_size} {grid_size} {grid_size}

{self.k_points_wannier(k_points)}
""")
        # generate a list of required overlaps (written to {name}.nnkp)
        # for some reason mpirun doesn't work...
        #os.system(mpi_run + f"wannier90.x -pp {self.name}")
        os.system(f'wannier90{"" if platform == "win32" else ".x"} -pp {self.name}')
    
    # use QUANTUM ESPRESSO's bands.x to convert bands output to workable data.
    # -> slow and buggy... for k-grids use my function for direct access instead
    def projections(self, lwrite_overlaps=False):
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
        os.system(mpi_run + f'projwfc{"" if platform == "win32" else ".x"} < {self.name}.projwfc.in | tee {self.name}.projwfc.out')


    # compute the overlaps of the wavefunctions from the nscf calculation, to be used by wannierization
    # use this after using prepare_wannier()
    def overlaps_for_wannier(self):
        with open(f"{self.name}.pw2wan.in", "w") as file:
            file.write(f"""
&inputpp 
   outdir='./qe-data/'
   prefix = '{self.name}'
   seedname = '{self.name}'
   !spin_component = 'none'
   !write_mmn = true
   !write_amn = true
   !write_unk = true ! not compatible with irr_bz
   !write_dmn = true ! not compatible with irr_bz
   !wan_mode = 'standalone'
   irr_bz = true
/
""")
        os.system(mpi_run + f'pw2wannier90{"" if platform == "win32" else ".x"} -in {self.name}.pw2wan.in | tee {self.name}.pw2wan.out')
    
    # compute the maximally localized wave functions (MLWFs)
    # use this after using overlaps_for_wannier()
    def wannier(self):
        # written to {name}.mmn and {name}.amn
        # parallel execution doesn't work for some reason...
        #os.system(mpi_run + f"wannier90.x {self.name}")
        os.system(f'wannier90{"" if platform == "win32" else ".x"} {self.name}')

    def plot_wannier(self):
        pass

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

        os.system(mpi_run + f'fs{"" if platform == "win32" else ".x"} -in {self.name}.fs.in > {self.name}.fs.out')

        os.system(f"xcrysden --bxsf {self.name}_fs.bxsf > /dev/null")


