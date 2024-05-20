import os
import numpy as np

# simple interface for QUANTUM ESPRESSO
# assuming quantum espresso is installed on linux

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
    for file in pseudo_potential_files_dict.values():
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
        #self.ibrav = ibrav_map[symmetry]
        self.ibrav = 0 # use CELL_PARAMETERS instead
        self.kinetic_energy_cutoff = kinetic_energy_cutoff

    # 3d plot of the crystal structure
    def plot_crystal(self, repeat=1, turntable=20, elevation=35):
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        offset = np.stack(np.reshape(np.meshgrid(*([np.arange(repeat)]*3)), (3, -1, 1)), axis=-1).reshape(-1, 1, 3)
        extended_basis = np.reshape(self.basis, (1, -1, 3)) + offset
        # TODO the multiple drawn lines destroy antialiasing... use NaN points instead
        cube_line = np.array([(0,0,0), (0,0,1), (0,1,1), (0,1,0), (0,0,0), (1,0,0), (1,0,1), (1,1,1), (1,1,0), (1,0,0), (1,1,0), (0,1,0), (0,1,1), (1,1,1), (1,0,1), (0,0,1)])
        deformed_cube = self.A @ cube_line.T
        ax.plot(*deformed_cube, "-k")
        ax.set_prop_cycle(None)
        for t in list(set(self.types)):
            ax.plot(*(self.A @ extended_basis[:, self.types == t].reshape(-1, 3).T), "o", label=t)
        ax.set_aspect("equal")
        ax.view_init(elev=elevation, azim=turntable)
        ax.legend()
        return fig, ax

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

    # read the data that has been computed (either by scf(), or by bands())
    # returns k_points, weights, bands, symmetries, fermi_energy
    def read_bands(self, incomplete=False):
        # read directly from the bands xml file (see https://realpython.com/python-xml-parser/)
        # read in k_points, eigenvalues and symmetries (as unnamed O(3) matrices)
        k_points = []
        bands = []
        weights = []
        S = []
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
    
    def read_wannier_tb(self, filename=None):
        import wannier90_tb_format as tb_fmt
        neighbors, params, r_params, degeneracy, A = tb_fmt.load_tb(f"{self.name}_tb.dat" if filename is None else filename)
        return neighbors, params, r_params


    # read the wavefunctions that have been computed by scf()
    def read_wavefunctions(self):
        # TODO reverse engineer https://gitlab.com/QEF/q-e/blob/31603626cc6bba390574e89c262a1e16d913a8a9/Modules/qexml.f90#L2052
        # or use wannier90 to convert it to a readable format
        raise NotImplementedError()
    
    # returns an array of density of states with the columns (column major)
    # Energy, Density of States, Integrated Density of States.
    # also returns the Fermi-energy
    def read_dos(self):
        fermi_energy = None
        with open(f"{self.name}.Dos.dat", "r") as file:
            header = file.readline()
            if "EFermi" in header and header.split(' ')[-1].strip() == "eV":
                fermi_energy = float(header.split(' ')[-2])
        density_of_states = np.loadtxt(f"{self.name}.Dos.dat", unpack=True)
        return density_of_states, fermi_energy

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

    # ----- QUANTUM ESPRESSO Parameter construction -----

    def crystal(self):
        cell_params = f"""CELL_PARAMETERS {self.unit}
{self.A[0,0]} {self.A[1,0]} {self.A[2,0]}
{self.A[0,1]} {self.A[1,1]} {self.A[2,1]}
{self.A[0,2]} {self.A[1,2]} {self.A[2,2]}
"""
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
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},
    ecutwfc = {self.kinetic_energy_cutoff},
    occupations='smearing', smearing='marzari-vanderbilt', degauss=0.02
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
        os.system(mpi_run + f"pw.x -nk 1 -nd 1 -nb 1 -nt 1 < {self.name}.scf.in | tee {self.name}.scf.out")
    
    def relax(self):
        relax_k_grid_size = 4
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
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},
    ecutwfc = {self.kinetic_energy_cutoff},
    occupations='smearing', smearing='marzari-vanderbilt', degauss=0.02
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-5,
    mixing_beta = 0.7,
/
&ions
    ion_dynamics="bfgs",
    ion_temperature="rescaling",
    tempw=400,
    tolp=50,
    nraise=1,
/
&cell
    cell_dynamics="bfgs",
    cell_dofree="ibrav",
/
{self.crystal()}
{self.k_grid(relax_k_grid_size)}
""")
        print(f"running the vc-relax calculation for {self.name}")
        os.system(mpi_run + f"pw.x -nk 1 -nd 1 -nb 1 -nt 1 < {self.name}.relax.in | tee {self.name}.relax.out")
    
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
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},
    ecutwfc = {self.kinetic_energy_cutoff}, nbnd = {band_count},
/
&electrons
    diagonalization='david',
/
{self.crystal()}
{str(k_points)}
""")
        print(f"running the band-structure calculation for {self.name}")
        os.system(mpi_run + f"pw.x -nk 1 -nd 1 -nb 1 -nt 1 < {self.name}.band.in | tee {self.name}.band.out")

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
        os.system(mpi_run + f"bands.x < {self.name}.bandx.in | tee {self.name}.bandx.out")

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
    outdir='./qe-data/',
    deltae =  1e-02,
    fildos = '{self.name}.Dos.dat',
/
""")
        os.system(mpi_run + f"dos.x < {self.name}.dos.in > {self.name}.dos.out")

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
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},
    ecutwfc = {self.kinetic_energy_cutoff},
    nbnd={band_count}
    occupations='smearing', smearing='marzari-vanderbilt', degauss=0.01
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
/
{self.crystal()}
{str(k_points)}
""")
        os.system(mpi_run + f"pw.x -nk 1 -nd 1 -nb 1 -nt 1 < {self.name}.nscf.in | tee {self.name}.nscf.out")

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
    ibrav = {self.ibrav}, nat={len(self.basis)}, ntyp= {len(set(self.types))},
    ecutwfc = {self.kinetic_energy_cutoff},
    nosym = true,
    noinv = true,
    nbnd={band_count}
    occupations='smearing', smearing='marzari-vanderbilt', degauss=0.01
/
&electrons
    diagonalization='david',
    conv_thr = 1.0e-8,
/
{self.crystal()}
{str(k_points)}
""")
        os.system(mpi_run + f"pw.x -nk 1 -nd 1 -nb 1 -nt 1 < {self.name}.nscf.in | tee {self.name}.nscf.out")

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
!auto_projections = true !there is a warning about this not always working with pw2wannier90.x
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
        os.system(f"wannier90.x -pp {self.name}")
    
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
        os.system(mpi_run + f"pw2wannier90.x -in {self.name}.pw2wan.in | tee {self.name}.pw2wan.out")
    
    # compute the maximally localized wave functions (MLWFs)
    # use this after using overlaps_for_wannier()
    def wannier(self):
        # written to {name}.mmn and {name}.amn
        # parallel execution doesn't work for some reason...
        #os.system(mpi_run + f"wannier90.x {self.name}")
        os.system(f"wannier90.x {self.name}")

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

        os.system(mpi_run + f"fs.x -in {self.name}.fs.in > {self.name}.fs.out")

        os.system(f"xcrysden --bxsf {self.name}_fs.bxsf > /dev/null")


