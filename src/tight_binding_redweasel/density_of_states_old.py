# all sorts of functions related to the density of states
# and the fermi energy at finite temperatures

import numpy as np

k_B = 8.61733326214518e-5 # eV/K

# cheap approximation of volume in cuboid using cube cuts
def cube_cut_volume(a):
    # approximate the integral with correct first order behavior
    a0 = (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]) / 8
    ax = np.abs(-a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]) / 4
    ay = np.abs(-a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]) / 4
    az = np.abs(-a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]) / 4
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5
    ax /= norm; ay /= norm; az /= norm; a0 /= norm
    volume = 0
    volume += np.maximum(0, ( ax + ay + az)/2 + a0)**3
    volume -= np.maximum(0, (-ax + ay + az)/2 + a0)**3
    volume -= np.maximum(0, ( ax - ay + az)/2 + a0)**3
    volume -= np.maximum(0, ( ax + ay - az)/2 + a0)**3
    volume += np.maximum(0, ( ax - ay - az)/2 + a0)**3
    volume += np.maximum(0, (-ax + ay - az)/2 + a0)**3
    volume += np.maximum(0, (-ax - ay + az)/2 + a0)**3
    volume -= np.maximum(0, (-ax - ay - az)/2 + a0)**3
    return volume / 6 / (ax * ay * az)

# cheap approximation using cube cuts (direct derivative of cube_cut_volume, not the actual area)
def cube_cut_area(a):
    # approximate the integral with correct first order behavior
    a0 = (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]) / 8
    ax = np.abs(-a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]) / 4
    ay = np.abs(-a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]) / 4
    az = np.abs(-a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]) / 4
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5
    ax /= norm; ay /= norm; az /= norm; a0 /= norm
    area = 0
    area += np.maximum(0, ( ax + ay + az)/2 + a0)**2
    area -= np.maximum(0, (-ax + ay + az)/2 + a0)**2
    area -= np.maximum(0, ( ax - ay + az)/2 + a0)**2
    area -= np.maximum(0, ( ax + ay - az)/2 + a0)**2
    area += np.maximum(0, ( ax - ay - az)/2 + a0)**2
    area += np.maximum(0, (-ax + ay - az)/2 + a0)**2
    area += np.maximum(0, (-ax - ay + az)/2 + a0)**2
    area -= np.maximum(0, (-ax - ay - az)/2 + a0)**2
    return area / 2 / (ax * ay * az) / norm

# gauss integration using the derivative of the fermi function as weight function
def gauss_5_df(f, mu, beta):
    x = np.array([-8.211650879369324585, -3.054894371595123559, 0.0, 3.054894371595123559, 8.211650879369324585])
    w = np.array([0.0018831678927720540279, 0.16265409664449248517, 0.6709254709254709216, 0.16265409664449248517, 0.0018831678927720540279])
    x_smpl = x / beta + mu
    return np.sum(f(x_smpl) * w) / beta

# gauss integration using the half-fermi function 1/(1+e^abs(x)) as weight function
def gauss_6_f(f, mu, beta):
    x = np.array([-10.612971636582431145, -4.7544317516063152596, -1.1799810705877835648, 1.1799810705877835648, 4.7544317516063152596, 10.612971636582431145])
    w = np.array([0.00013527426019966680491, 0.02778699826589691238, 0.47207772747390341905, 0.47207772747390341905, 0.02778699826589691238, 0.00013527426019966680491])
    x_smpl = x / beta + mu
    return np.sum(f(x_smpl) * w) / beta * 2*np.log(2)

# finds the inverse of strictly monotonic functions on the full range of real numbers
# is unstable at points where the derivative of f is very small (-> strictly monotonic)
# converges faster than secant_bisect
# use the more precise start estimate for b, because it is kept in the first step
def secant(y, f, a, b, tol=1e-16, max_i=100):
    fa = f(a) - y
    fb = f(b) - y
    for i in range(max_i):
        x = (a * fb - b * fa) / (fb - fa)
        if a == x or b == x:
            return x
        val = f(x) - y
        a = b
        fa = fb
        b = x
        fb = val
        if abs(val) < tol or fa == fb:
            return x
    # didn't finish but here is the best available solution
    return (a * fb - b * fa) / (fb - fa)

class DensityOfStates:
    def __init__(self, model, N=24):
        x_ = np.linspace(-.5, .5, N, endpoint=False)
        y_ = np.linspace(-.5, .5, N, endpoint=False)
        z_ = np.linspace(-.5, .5, N, endpoint=False)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        self.k_smpl = np.stack([x, y, z], axis=-1)
        shape = np.shape(self.k_smpl)
        bands = model.bands(np.reshape(self.k_smpl, (-1, 3)))
        self.model = model
        self.bands = np.reshape(bands, shape[:-1]+(-1,))
        self.bands_range = [(np.min(self.bands[...,i]), np.max(self.bands[...,i])) for i in range(self.bands.shape[-1])]
    
    def states_below(self, energy):
        states = 0.0
        for i, band_range in enumerate(self.bands_range):
            if (band_range[0] - energy) * (band_range[1] - energy) < 0:
                band = energy - self.bands[...,i] # positive parts are considered for volume
                # TODO prepare the coefficients that are computed in cube_cut_volume and just shift one of them
                cubes = [band, np.roll(band, 1, axis=0)]
                cubes.extend([np.roll(vertex, 1, axis=1) for vertex in cubes])
                cubes.extend([np.roll(vertex, 1, axis=2) for vertex in cubes])
                states += np.mean(cube_cut_volume(cubes))
            elif band_range[1] <= energy:
                states += 1.0 # completely full
        return states
    
    # returns states_below, density
    def states_density(self, energy):
        states = 0.0
        density = 0.0
        for i, band_range in enumerate(self.bands_range):
            if (band_range[0] - energy) * (band_range[1] - energy) < 0:
                band = energy - self.bands[...,i] # positive parts are considered for volume
                cubes = [band, np.roll(band, 1, axis=0)]
                cubes.extend([np.roll(vertex, 1, axis=1) for vertex in cubes])
                cubes.extend([np.roll(vertex, 1, axis=2) for vertex in cubes])
                states += np.mean(cube_cut_volume(cubes))
                density += np.mean(cube_cut_area(cubes))
            elif band_range[1] <= energy:
                states += 1.0 # completely full
        return states, density
    
    # returns density
    def density(self, energy):
        density = 0.0
        for i, band_range in enumerate(self.bands_range):
            if (band_range[0] - energy) * (band_range[1] - energy) < 0:
                band = energy - self.bands[...,i] # positive parts are considered for volume
                cubes = [band, np.roll(band, 1, axis=0)]
                cubes.extend([np.roll(vertex, 1, axis=1) for vertex in cubes])
                cubes.extend([np.roll(vertex, 1, axis=2) for vertex in cubes])
                density += np.mean(cube_cut_area(cubes))
        return density
    
    # get the indices of the bands, that cut the given energy
    def cut_band_indices(self, energy):
        indices = []
        for i, band_range in enumerate(self.bands_range):
            if (band_range[0] - energy) * (band_range[1] - energy) < 0:
                indices.append(i)
        return indices
    
    # compute the full density of states curve
    # returns energy_smpl, states, density
    def full_curve(self, N=10):
        # compute energy samples based on where the bands start and end
        energy_smpl = []
        for (min_e, max_e) in self.bands_range:
            energy_smpl.extend(list(np.linspace(min_e, max_e, N)))
        energy_smpl = sorted(energy_smpl)
        states = []
        density = []
        for energy in energy_smpl:
            states_, density_ = self.states_density(energy)
            states.append(states_)
            density.append(density_)
        return energy_smpl, states, density
    
    # for a given number of electrons per cell (float in general)
    # compute the fermi energy (correct for metals and isolators)
    def fermi_energy(self, electrons, tol=1e-8, maxsteps=30):
        # first approximation from integer number of electrons
        assert electrons >= 0 and electrons <= len(self.bands_range)
        e_int = round(electrons)
        if e_int > 0 and e_int < len(self.bands_range):
            max_below = self.bands_range[e_int-1][1]
            min_above = self.bands_range[e_int][0]
            fermi_energy = (max_below + min_above) / 2
            if max_below < min_above:
                return fermi_energy # isolator
        elif e_int == 0:
            fermi_energy = self.bands_range[0][0]
        else:
            assert e_int == len(self.bands_range)
            fermi_energy = self.bands_range[-1][1]
        for _ in range(maxsteps):
            states, density = self.states_density(fermi_energy)
            # TODO handle density = 0 using bisection
            fermi_energy -= (states - electrons) / density
            if abs((states - electrons) / density) <= tol:
                return fermi_energy
        raise ValueError(f"root search didn't converge in {maxsteps} steps. {abs((states - electrons) / density)} > {tol}.")

    # returns the approximate bandgap if the model describes an isolator, 0 if it describes a metal
    def bandgap(self, electrons):
        # first approximation from integer number of electrons
        assert electrons >= 0 and electrons <= len(self.bands_range)
        e_int = round(electrons)
        assert e_int > 0 and e_int < len(self.bands_range)
        max_below = self.bands_range[e_int-1][1]
        min_above = self.bands_range[e_int][0]
        if max_below < min_above:
            # TODO improve this calculation with self.model.bands_grads(...)
            return min_above - max_below # isolator
        return 0.0 # metal

    # for a given number of electrons per cell (float in general)
    # and a temperature T in Kelvin, compute the chemical potential mu
    # (units are correct if the bands from the model are in eV)
    def chemical_potential(self, electrons, T, fermi_energy=None, tol=1e-8, maxsteps=30):
        assert electrons >= 0 and electrons <= len(self.bands_range)
        assert T >= 0
        if fermi_energy is None:
            mu = self.fermi_energy(electrons, tol=tol, maxsteps=maxsteps)
        else:
            mu = fermi_energy
        if T == 0:
            return mu
        beta = 1 / (k_B * T) # 1/eV
        # now satisfy the equation N(mu) + gauss_5_f(density, mu, beta) = electrons
        # to do that, use secant root solving and hope for convergence
        def null_func(mu):
            states = self.states_below(mu) - electrons
            # TODO this doesn't work for isolators...
            states += gauss_6_f(lambda e: np.array([self.density(e_) for e_ in e]) * np.sign(e - mu), mu, beta)
            return states
        return secant(0, null_func, mu, mu + 0.01, tol, maxsteps)

    # compute points on the fermi surface
    # returns points, weights
    def fermi_surface_samples(self):
        return None