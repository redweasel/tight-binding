# this file contains functions used by DoS
# for approximating the zero temperature state (density).
# The functions here are mostly geometrical functions, that
# compute cuts of polyhedra with planes to approximate the
# volume integral for the amount of states below a given energy.
import numpy as np

# base class for smearing methods. All documentation is here
class SmearingMethod:
    """Volume computations for implicit functions defined by a grid of data points."""

    def volume(self, value: float):
        """Compute the volume (0 to 1) of the data that has a value less than the given value.

        Args:
            value (float): maximal value of the data to be considered inside the volume.

        Returns:
            ndarray[N_k]: the volume of the implicit function for each cell, that has a smaller value than "value"
        """
        raise NotImplementedError()
    
    def dvolume(self, value: float):
        """Compute the gradient wrt "value" of the volume of the data that has a value less than the given value.

        Args:
            value (float): maximal value of the data to be considered inside the volume.

        Returns:
            ndarray[N_k]: the derivative of `volume` wrt value.
        """
        raise NotImplementedError()
    
    def volume_dvolume(self, value: float):
        """Compute both the volume and gradient of the volume. See `volume` and `dvolume`.

        Args:
            value (float): maximal value of the data to be considered inside the volume.

        Returns:
            (ndarray[N_k], ndarray[N_k]): `volume`, `dvolume`
        """
        return self.volume(value), self.dvolume(value)
    
    def samples(self, value: float):
        """Compute points on the surface of the implicit function and compute integration weights.

        Args:
            value (float): maximal value of the data to be considered inside the volume.

        Returns:
            (ndarray[N, 3], ndarray[N, 3], ndarray[N]): sample positions in reciprocal space, approximate model gradients at the samplepoints in reciprocal space, sample weights (= area)
        """
        raise NotImplementedError()


# huge epsilon needed to make the axis aligned case work ok
EPSILON = 1e-4

# cheap approximation of volume in cuboid using cube cuts
def cube_cut_volume(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    # simplify using abs (also on a0 to reduce numerical cancelation and get more 0 multiplications)
    a0_sign = a0 < 0.0
    a0, ax, ay, az = np.abs(a0), np.abs(ax), np.abs(ay), np.abs(az) # copy arrays!
    # cube cuts = sum of right angle tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-40
    ax /= norm; ay /= norm; az /= norm; a0 /= norm
    ax = np.maximum(ax, EPSILON); ay = np.maximum(ay, EPSILON); az = np.maximum(az, EPSILON)
    a1 = (-ax + ay + az) / 2
    a2 = ( ax - ay + az) / 2
    a3 = ( ax + ay - az) / 2
    volume = np.maximum(0, ( ax + ay + az)/2 - a0)**3
    volume -= np.maximum(0, a1 - a0)**3
    volume -= np.maximum(0, a2 - a0)**3
    volume += np.maximum(0, -np.minimum(np.minimum(a1, a2), a3) - a0)**3
    volume -= np.maximum(0, a3 - a0)**3
    # (ax,ay,az) is normalized, so if a0 > 3**.5/2, then the cube will be either fully in or out
    volume = np.where(a0 < 3**.5/2, np.minimum(1.0, volume / (6 * ax * ay * az)), 0.0)
    return np.where(a0_sign, volume, 1.0 - volume)

# cheap approximation using cube cuts (direct derivative of cube_cut_volume, not the actual area!)
def cube_cut_dvolume(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    a0, ax, ay, az = np.abs(a0), np.abs(ax), np.abs(ay), np.abs(az) # copy arrays!
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-40
    ax /= norm; ay /= norm; az /= norm; a0 /= norm
    ax = np.maximum(ax, EPSILON); ay = np.maximum(ay, EPSILON); az = np.maximum(az, EPSILON)
    a1 = (-ax + ay + az) / 2
    a2 = ( ax - ay + az) / 2
    a3 = ( ax + ay - az) / 2
    area = np.maximum(0, ( ax + ay + az)/2 - a0)**2
    area -= np.maximum(0, a1 - a0)**2
    area -= np.maximum(0, a2 - a0)**2
    area += np.maximum(0, -np.minimum(np.minimum(a1, a2), a3) - a0)**2
    area -= np.maximum(0, a3 - a0)**2
    axyzn = ax * ay * az * norm
    # (ax,ay,az) is normalized, so if a0 > 3**.5/2, then the cube will be either fully in or out
    return np.where(a0 < 3**.5/2, area / (2 * axyzn), 0.0)

# cheap approximation of volume and area in cuboid using cube cuts
def cube_cut_volume_dvolume(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    # cube cuts = sum of tetrahedrons
    a0_sign = a0 < 0.0
    a0, ax, ay, az = np.abs(a0), np.abs(ax), np.abs(ay), np.abs(az) # copy arrays!
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-40
    ax /= norm; ay /= norm; az /= norm; a0 /= norm
    ax = np.maximum(ax, EPSILON); ay = np.maximum(ay, EPSILON); az = np.maximum(az, EPSILON)
    a1 = (-ax + ay + az) / 2
    a2 = ( ax - ay + az) / 2
    a3 = ( ax + ay - az) / 2
    v0 = np.maximum(0, ( ax + ay + az)/2 - a0)
    v1 = np.maximum(0, a1 - a0)
    v2 = np.maximum(0, a2 - a0)
    v3 = np.maximum(0, a3 - a0)
    v4 = np.maximum(0, -np.minimum(np.minimum(a1, a2), a3) - a0)
    volume = v0**3 - v1**3 - v2**3 + v4**3 - v3**3
    area = v0**2 - v1**2 - v2**2 + v4**2 - v3**2
    # (ax,ay,az) is normalized, so if a0 > 3**.5/2, then the cube will be either fully in or out
    axyz = ax * ay * az
    volume = np.where(a0 < 3**.5/2, volume / (6 * axyz), 0.0)
    dvolume = np.where(a0 < 3**.5/2, area / (2 * axyz * norm), 0.0)
    return np.where(a0_sign, volume, 1.0 - volume), dvolume

# center of mass of the surface of a cube cut
def cube_cut_area_com(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    sx, sy, sz = np.sign(ax), np.sign(ay), np.sign(az)
    ax, ay, az = np.abs(ax), np.abs(ay), np.abs(az) # copy arrays!
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-40
    ax /= norm; ay /= norm; az /= norm; a0 = a0 / norm
    ax = np.maximum(ax, EPSILON); ay = np.maximum(ay, EPSILON); az = np.maximum(az, EPSILON)
    v0 = np.maximum(0, ( ax + ay + az)/2 + a0)
    v1 = np.maximum(0, (-ax + ay + az)/2 + a0)
    v2 = np.maximum(0, ( ax - ay + az)/2 + a0)
    v3 = np.maximum(0, ( ax + ay - az)/2 + a0)
    v4 = np.maximum(0, ( ax - ay - az)/2 + a0)
    v5 = np.maximum(0, (-ax + ay - az)/2 + a0)
    v6 = np.maximum(0, (-ax - ay + az)/2 + a0)
    v7 = np.maximum(0, (-ax - ay - az)/2 + a0)
    area = np.where(np.abs(a0) < 3**.5/2, v0**2 - v1**2 - v2**2 + v4**2 - v3**2 + v5**2 + v6**2 - v7**2, 0)
    shape = (1,) * len(np.shape(a0)) + (3,)
    com = np.reshape((0, 0, 0), shape)
    d = (1/3) / np.stack((ax, ay, az), axis=-1)
    com=com+v0[...,None]**2 * (np.reshape((-1, -1, -1), shape)/2 + d * v0[...,None])
    com -=  v1[...,None]**2 * (np.reshape(( 1, -1, -1), shape)/2 + d * v1[...,None])
    com -=  v2[...,None]**2 * (np.reshape((-1,  1, -1), shape)/2 + d * v2[...,None])
    com +=  v4[...,None]**2 * (np.reshape((-1,  1,  1), shape)/2 + d * v4[...,None])
    com -=  v3[...,None]**2 * (np.reshape((-1, -1,  1), shape)/2 + d * v3[...,None])
    com +=  v5[...,None]**2 * (np.reshape(( 1, -1,  1), shape)/2 + d * v5[...,None])
    com +=  v6[...,None]**2 * (np.reshape(( 1,  1, -1), shape)/2 + d * v6[...,None])
    com -=  v7[...,None]**2 * (np.reshape(( 1,  1,  1), shape)/2 + d * v7[...,None])
    com *= np.stack((sx, sy, sz), axis=-1)
    com = com / np.where(area[...,None] != 0, area[...,None], np.inf)
    axyz = ax * ay * az
    area = np.where(np.abs(axyz) != 0, area / (2 * axyz), 0)
    return area, com


class CubesSmearing(SmearingMethod):
    """
    Smearing method based on the volume of cubes. This is similar to the tetrahedron method,
    however instead of using tetrahedrons, where the volume of each cut tetrahedron can be computed exactly,
    this method approximates the energy bands linearly in a cube and uses that to cut the cube.
    It is 8x faster than the tetrahedron method and has similar precision.
    """
    def __init__(self, positions, A, values, grads=None, wrap=False):
        self.B = np.linalg.inv(A).T
        # k_smpl are in reciprocal space coordinates.
        # self.centers should be in crystal coordinates.
        self.centers = positions @ A
        # determine stepsizes from centers (should be a rectilinear grid)
        self.step_sizes = np.array([
            self.centers[1,0,0][0] - self.centers[0,0,0][0],
            self.centers[0,1,0][1] - self.centers[0,0,0][1],
            self.centers[0,0,1][2] - self.centers[0,0,0][2],
        ])
        if grads is None:
            self.centers += self.step_sizes * 0.5
            if not wrap:
                self.centers = self.centers[:-1,:-1,:-1]
            # preprocessing everything here, halves the computation time in my test cases
            a = [values, np.roll(values, -1, axis=0)]
            a.extend([np.roll(vertex, -1, axis=1) for vertex in a])
            a.extend([np.roll(vertex, -1, axis=2) for vertex in a])
            a0 = (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]) / 8
            ax = (-a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]) / 4
            ay = (-a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]) / 4
            az = (-a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]) / 4
            #axy = (a[0] - a[1] - a[2] + a[3] + a[4] - a[5] - a[6] + a[7]) / 2
            #ayz = (a[0] + a[1] - a[2] - a[3] - a[4] - a[5] + a[6] + a[7]) / 2
            #axz = (a[0] - a[1] + a[2] - a[3] - a[4] + a[5] - a[6] + a[7]) / 2
            #axyz = (-a[0] + a[1] + a[2] - a[3] + a[4] - a[5] - a[6] + a[7])
            # TODO one could take the second derivative along the gradient into account here!
            #  -> just a correction to a0 depending on the surface value
            # TODO move the normalisation and other calculations that only involve ax, ay, and az into here -> add "norm" as returned cache
            if wrap:
                self.cubes = a0, ax, ay, az
            else:
                self.cubes = a0[:-1,:-1,:-1], ax[:-1,:-1,:-1], ay[:-1,:-1,:-1], az[:-1,:-1,:-1]
        else:
            # gradients are in k_space, -> convert to crystal space!
            # crystal space is [0,1]^3 so unit volume.
            self.cubes = values, *np.einsum("...i,ij->j...", grads, self.B) * self.step_sizes[:,None,None,None]
            # The results here are a bit worse, however
            # I think that is not because of the gradients,
            # but because the averaging of the values (without gradients)
            # had the effect of implicitly considering second order effects.
            # TODO reevaluate the above statement
            if not wrap:
                raise NotImplementedError("Non-wrapping with gradients requires cube weights, which are currently not implemented.")
        L = self.B @ np.diag(self.step_sizes)
        normal = np.reshape(self.cubes[1:], (3, -1)).T
        basis = np.zeros((len(normal), 3, 3)) + np.eye(3)
        basis[:,:,0] = normal
        basis = np.linalg.qr(basis).Q
        basis2 = np.zeros((len(normal), 3, 3)) + np.eye(3)
        basis2[:,:,0] = normal @ np.linalg.inv(L)
        basis2 = np.linalg.qr(basis2).Q
        self.scale = np.abs(np.linalg.det(np.swapaxes(basis2[:,:,1:], -1, -2) @ L @ basis[:,:,1:]))

    def volume(self, value: float):
        return cube_cut_volume(value - self.cubes[0], *self.cubes[1:])

    def dvolume(self, value: float):
        return cube_cut_dvolume(value - self.cubes[0], *self.cubes[1:])
    
    def volume_dvolume(self, value: float):
        volume, dvolume = cube_cut_volume_dvolume(value - self.cubes[0], *self.cubes[1:])
        return volume, dvolume

    def samples(self, value: float):
        area, x = cube_cut_area_com(value - self.cubes[0], *self.cubes[1:])
        # transform x into the the cubes
        x *= self.step_sizes
        x += self.centers
        x = x.reshape(-1, 3)
        w = np.ravel(area)
        select = w > 1e-4
        w = w[select]
        x = x[select]
        # transform crystal to reciprocal coordinates
        x = x @ self.B.T
        # transform the areas into the correct coordinates
        # -> areas transform with the scaling (det) in the subspace perpendicular to the normal
        # TODO move all the scale computation to __init__ as the normals are already known there
        w *= self.scale[select]
        # TODO divide w by the correct number from the number of cubes in the model
        # TODO how does the area change with scaling? This seems bad!
        # crystal space gradients
        grads = np.reshape(self.cubes[1:], (3, -1)).T[select] / self.step_sizes
        # transform to reciprocal space gradients
        grads = grads @ np.linalg.inv(self.B)
        return x, grads, w


# unit tetrahedron made from (0,0,0), (1,0,0), (0,1,0), (0,0,1)
def unit_tetra_cut_volume(a0, ax, ay, az):
    # normalisation for stability only!
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-50
    ax = ax / norm; ay = ay / norm; az = az / norm; a0 = a0 / norm
    ax, ay, az = np.sort((ax, ay, az), axis=0) # ax is smallest, az is biggest
    # switch to dual problem to cut the growing parts in half
    dual = ay + a0 < 0
    a0, ax, ay, az = np.where(dual, (-a0, -az, -ay, -ax), (a0, ax, ay, az))
    # the following makes the calculation avoid division by 0, but introduces an error of magnitude 1e-8 in the special cases
    ax = np.where(np.abs(ax) < 4e-9, -4e-9, ax)
    ay = np.where(np.abs(ay) < 1e-9, 1e-9, ay)
    az = np.where(np.abs(az) < 2e-9, 2e-9, az)
    volume = np.maximum(0, -a0)**3 / (ax * ay * az)
    volume -= np.maximum(0, -ax - a0)**3 / (ax * (ay - ax + 1e-50) * (az - ax + 1e-50))
    return np.where(dual, volume, 1 - volume) / 6

# unit tetrahedron made from (0,0,0), (1,0,0), (0,1,0), (0,0,1)
def unit_tetra_cut_dvolume(a0, ax, ay, az):
    # normalisation for stability only!
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-50
    ax = ax / norm; ay = ay / norm; az = az / norm; a0 = a0 / norm
    ax, ay, az = np.sort((ax, ay, az), axis=0) # ax is smallest, az is biggest
    # switch to dual problem to cut the growing parts in half
    dual = ay + a0 < 0
    a0, ax, ay, az = np.where(dual, (-a0, -az, -ay, -ax), (a0, ax, ay, az))
    # the following makes the calculation avoid division by 0, but introduces an error of magnitude 1e-8 in the special cases
    ax = np.where(np.abs(ax) < 4e-9, -4e-9, ax)
    ay = np.where(np.abs(ay) < 1e-9, 1e-9, ay)
    az = np.where(np.abs(az) < 2e-9, 2e-9, az)
    dvolume = np.maximum(0, -a0)**2 / (ax * ay * az)
    dvolume -= np.maximum(0, -ax - a0)**2 / (ax * (ay - ax + 1e-50) * (az - ax + 1e-50))
    return dvolume / 2 / norm

# arbitrary tetrahedron cut with full volume 1/6
def tetra_cut_volume(v0, v1, v2, v3):
    a0 = v0
    ax = v1 - v0
    ay = v2 - v0
    az = v3 - v0
    return unit_tetra_cut_volume(a0, ax, ay, az)

# derivative of (wrt total value) of arbitrary tetrahedron cut with full volume 1/6
def tetra_cut_dvolume(v0, v1, v2, v3):
    a0 = v0
    ax = v1 - v0
    ay = v2 - v0
    az = v3 - v0
    return unit_tetra_cut_dvolume(a0, ax, ay, az)

# cube cut with tetrahedrons, which share the diagonal a[0], a[7]
def cube_tetra_cut_volume(a):
    assert len(a) == 8
    tetras  = tetra_cut_volume(a[0], a[7], a[0b001], a[0b011])
    tetras += tetra_cut_volume(a[0], a[7], a[0b011], a[0b010])
    tetras += tetra_cut_volume(a[0], a[7], a[0b010], a[0b110])
    tetras += tetra_cut_volume(a[0], a[7], a[0b110], a[0b100])
    tetras += tetra_cut_volume(a[0], a[7], a[0b100], a[0b101])
    tetras += tetra_cut_volume(a[0], a[7], a[0b101], a[0b001])
    return tetras

# cube cut with tetrahedrons, which share the diagonal a[0], a[7]
def cube_tetra_cut_dvolume(a):
    assert len(a) == 8
    tetras  = tetra_cut_dvolume(a[0], a[7], a[0b001], a[0b011])
    tetras += tetra_cut_dvolume(a[0], a[7], a[0b011], a[0b010])
    tetras += tetra_cut_dvolume(a[0], a[7], a[0b010], a[0b110])
    tetras += tetra_cut_dvolume(a[0], a[7], a[0b110], a[0b100])
    tetras += tetra_cut_dvolume(a[0], a[7], a[0b100], a[0b101])
    tetras += tetra_cut_dvolume(a[0], a[7], a[0b101], a[0b001])
    return tetras

class TetraSmearing(SmearingMethod):
    """
    Classical tetrahedron method from literature (O. Jepsen and O. K. Andersen, Solid State Commun. 9,
    1763 (1971))
    """
    def __init__(self, values, wrap, B):
        a = [values, np.roll(values, -1, axis=0)]
        a.extend([np.roll(vertex, -1, axis=1) for vertex in a])
        a.extend([np.roll(vertex, -1, axis=2) for vertex in a])
        # TODO align the cubes correctly, such that the diagonal is the longest one or fits the symmetry.
        # -> use B, the reciprocal lattice matrix for that
        # TODO precompute the values for unit_tetra_cut_volume as none of that computation changes a0
        # -> 6x memory usage, ~3x performance
        if wrap:
            self.tetras = np.array(a)
        else:
            self.tetras = np.array(a)[:,:-1,:-1,:-1]

    def volume(self, value: float):
        return cube_tetra_cut_volume(value - self.tetras)

    def dvolume(self, value: float):
        return cube_tetra_cut_dvolume(value - self.tetras)

    def volume_dvolume(self, value: float):
        shifted_tetras = value - self.tetras
        volume = cube_tetra_cut_volume(shifted_tetras)
        dvolume = cube_tetra_cut_dvolume(shifted_tetras)
        return volume, dvolume

# cut a sphere with volume 1 and therefore radius r=0.62035=cbrt(3/4pi)
def unit_sphere_cut(a0, ax, ay, az):
    norm = (ax**2 + ay**2 + az**2)**0.5 + 1e-50
    # plane distance from sphere center
    r = 0.620350490899
    d = np.maximum(np.minimum(a0 / (norm * r), 1.0), -1.0)
    dp = 1 + d
    volume = dp*dp * (2 - d) * (1/4)
    dvolume = dp * (1 - d) * (3/4 / r) / norm
    return volume, dvolume

class SphereSmearing(SmearingMethod):
    """A simple approximation for smearing.
    This can be used with computed analytic gradients or with interpolated gradients.
    With interpolated gradients, this method produces results VERY close to the cubic smearing, but much faster (~4x).
    However, derivatives of the density of states have much larger errors with this method.
    """
    def __init__(self, positions, B, values, grads=None, wrap=False):
        self.positions = positions
        if grads is None:
            # get the gradients like for cubes from the data itself.
            # the spheres are considered to be arranged as a cubic lattice in crystal space.
            a = [values, np.roll(values, -1, axis=0)]
            a.extend([np.roll(vertex, -1, axis=1) for vertex in a])
            a.extend([np.roll(vertex, -1, axis=2) for vertex in a])
            a0 = (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]) / 8
            ax = (-a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]) / 4
            ay = (-a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]) / 4
            az = (-a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]) / 4
            if wrap:
                self.values = a0
                self.grads = ax, ay, az
            else:
                self.values = a0[:-1,:-1,:-1]
                self.grads = ax[:-1,:-1,:-1], ay[:-1,:-1,:-1], az[:-1,:-1,:-1]
        else:
            # gradients are in k_space, we want them to remain there,
            # as the lattice is assumed to be well spaced in reciprocal space.
            # that means the total volume is not 1, but np.linalg.det(B).
            volume = np.linalg.det(B)
            cell_volume = volume / len(positions[...,0].flat)
            # to counteract that, one can scale the gradients
            self.grads = np.moveaxis(grads, -1, 0) * cell_volume**(1/3)
            self.values = values
            # this works well, however it doesn't produce better results than the interpolated one.
            # the interpolated version usually overestimates the fermi-energy, while this usually underestimates it.
            # -> a mix between the two methods will probably be superior!
            # TODO test!
            if not wrap:
                raise NotImplementedError("Non-wrapping with gradients requires cube weights, which are currently not implemented.")

    def volume_dvolume(self, value: float):
        volume, dvolume = unit_sphere_cut(value - self.values, *self.grads)
        return volume, dvolume

    def volume(self, value: float):
        return self.volume_dvolume(value)[0]
    
    def dvolume(self, value: float):
        return self.volume_dvolume(value)[1]


