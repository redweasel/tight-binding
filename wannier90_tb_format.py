import numpy as np

# This file makes an interface for my calculations with wannier90
# However! wannier90 has a very arcane style of file
# It's essentially a custom format for each purpose.
# They didn't care whatsoever about reusable code.
# For my codebase and exchange with my other code in other languagues I will be using JSON

# read the wannier90 ..._tb.dat format
def load_tb(filename):
    with open(filename, "r") as file:
        file.readline() # just the creation date
        # read the lattice vectors in angstrom units
        A = np.array([[float(x.strip()) for x in file.readline().strip().split()] for i in range(3)])
        # read number of bands and the number of points in the wigner seitz gridcell
        band_count = int(file.readline().strip())
        point_count = int(file.readline().strip())
        # now read the degeneracy list with length point_count
        degeneracy = ""
        while True:
            line = file.readline()
            if len(line.strip()) > 0:
                degeneracy = degeneracy + line.strip() + " "
            else:
                break
        degeneracy = [int(s) for s in degeneracy.split() if len(s)]
        assert point_count == len(degeneracy)
        # now read the hamiltonian matrices
        neighbors = []
        params = []
        r_params = []
        index = -1
        for line in file:
            if not line.strip():
                continue # skip empty lines
            parts = line.strip().split()
            if len(parts) == 3:
                # start new matrix or return to existing one
                pos = tuple([int(x) for x in parts])
                neg_pos = tuple([-x for x in pos])
                if pos not in neighbors:
                    if pos != neg_pos and neg_pos in neighbors:
                        # skip the entry if the adjungated matrix is already in neighbors
                        index = -1
                        continue
                    index = len(neighbors)
                    neighbors.append(pos)
                    params.append(np.zeros((band_count, band_count), dtype=np.complex128))
                    r_params.append(np.zeros((band_count, band_count, 3), dtype=np.complex128))
                else:
                    index = neighbors.index(pos)
            elif len(parts) == 4:
                if index < 0:
                    continue # skip this entry
                # new matrix entry (i,j) x+iy
                i,j = int(parts[0]), int(parts[1])
                x,y = float(parts[2]), float(parts[3])
                params[index][i-1,j-1] = x + y*1j
            elif len(parts) == 8:
                if index < 0:
                    continue # skip this entry
                # new matrix element <i| vec r|j>
                i,j = int(parts[0]), int(parts[1])
                x = float(parts[2]) + 1j*float(parts[3])
                y = float(parts[4]) + 1j*float(parts[5])
                z = float(parts[6]) + 1j*float(parts[7])
                r_params[index][i-1,j-1,0] = x
                r_params[index][i-1,j-1,1] = y
                r_params[index][i-1,j-1,2] = z
            else:
                raise ValueError(f"File has a format error in the tight bindng matrices in line \"{line.strip()}\"")
    # sort the neighbors by length
    neighbors = np.array(neighbors)
    params = np.array(params)
    r_params = np.array(r_params)
    order = np.argsort(np.linalg.norm(neighbors, axis=-1))
    return neighbors[order], params[order], r_params[order], degeneracy, A

# save the wannier90 ..._hr.dat format for testing purposes
def save_hr(filename, neighbors, params):
    raise NotImplementedError()
