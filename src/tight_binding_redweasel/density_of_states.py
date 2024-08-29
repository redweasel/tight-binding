# all sorts of functions related to the density of states
# and the fermi energy at finite temperatures

import numpy as np

k_B = 8.61733326214518e-5 # eV/K

def cubes_preprocessing(band, wrap):
    a = [band, np.roll(band, -1, axis=0)]
    a.extend([np.roll(vertex, -1, axis=1) for vertex in a])
    a.extend([np.roll(vertex, -1, axis=2) for vertex in a])
    a0 = (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]) / 8
    ax = -(-a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]) / 4
    ay = -(-a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]) / 4
    az = -(-a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]) / 4
    #axy = (a[0] - a[1] - a[2] + a[3] + a[4] - a[5] - a[6] + a[7]) / 2
    #ayz = (a[0] + a[1] - a[2] - a[3] - a[4] - a[5] + a[6] + a[7]) / 2
    #axz = (a[0] - a[1] + a[2] - a[3] - a[4] + a[5] - a[6] + a[7]) / 2
    #axyz = (-a[0] + a[1] + a[2] - a[3] + a[4] - a[5] - a[6] + a[7])
    if wrap:
        return a0, ax, ay, az
    else:
        return a0[:-1,:-1,:-1], ax[:-1,:-1,:-1], ay[:-1,:-1,:-1], az[:-1,:-1,:-1]

# cheap approximation of volume in cuboid using cube cuts
def cube_cut_volume(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    # TODO get rid of these added constants (they are added to make the axial case work without if's)
    ax, ay, az = np.abs(ax) + 1e-6, np.abs(ay) + 1e-6, np.abs(az) + 1e-6
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5
    ax /= norm; ay /= norm; az /= norm; a0 = a0 / norm
    volume = 0
    volume += np.maximum(0, ( ax + ay + az)/2 + a0)**3
    volume -= np.maximum(0, (-ax + ay + az)/2 + a0)**3
    volume -= np.maximum(0, ( ax - ay + az)/2 + a0)**3
    volume -= np.maximum(0, ( ax + ay - az)/2 + a0)**3
    volume += np.maximum(0, ( ax - ay - az)/2 + a0)**3
    volume += np.maximum(0, (-ax + ay - az)/2 + a0)**3
    volume += np.maximum(0, (-ax - ay + az)/2 + a0)**3
    volume -= np.maximum(0, (-ax - ay - az)/2 + a0)**3
    axyz = ax * ay * az
    return np.where(np.abs(norm*axyz) > 1e-8, volume / 6 / axyz, a0 > 0)

# cheap approximation using cube cuts (direct derivative of cube_cut_volume, not the actual area!)
def cube_cut_dvolume(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    ax, ay, az = np.abs(ax) + 1e-6, np.abs(ay) + 1e-6, np.abs(az) + 1e-6
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5
    ax /= norm; ay /= norm; az /= norm; a0 = a0 / norm
    area = 0
    area += np.maximum(0, ( ax + ay + az)/2 + a0)**2
    area -= np.maximum(0, (-ax + ay + az)/2 + a0)**2
    area -= np.maximum(0, ( ax - ay + az)/2 + a0)**2
    area -= np.maximum(0, ( ax + ay - az)/2 + a0)**2
    area += np.maximum(0, ( ax - ay - az)/2 + a0)**2
    area += np.maximum(0, (-ax + ay - az)/2 + a0)**2
    area += np.maximum(0, (-ax - ay + az)/2 + a0)**2
    area -= np.maximum(0, (-ax - ay - az)/2 + a0)**2
    axyzn = ax * ay * az * norm
    return np.where(np.abs(axyzn) > 1e-8, area / (2 * axyzn), 0)

# cheap approximation of volume and area in cuboid using cube cuts
def cube_cut_volume_area(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    ax, ay, az = np.abs(ax) + 1e-6, np.abs(ay) + 1e-6, np.abs(az) + 1e-6
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5
    ax /= norm; ay /= norm; az /= norm; a0 = a0 / norm
    v0 = np.maximum(0, ( ax + ay + az)/2 + a0)
    v1 = np.maximum(0, (-ax + ay + az)/2 + a0)
    v2 = np.maximum(0, ( ax - ay + az)/2 + a0)
    v3 = np.maximum(0, ( ax + ay - az)/2 + a0)
    v4 = np.maximum(0, ( ax - ay - az)/2 + a0)
    v5 = np.maximum(0, (-ax + ay - az)/2 + a0)
    v6 = np.maximum(0, (-ax - ay + az)/2 + a0)
    v7 = np.maximum(0, (-ax - ay - az)/2 + a0)
    volume = v0**3 - v1**3 - v2**3 - v3**3 + v4**3 + v5**3 + v6**3 - v7**3
    area = v0**2 - v1**2 - v2**2 - v3**2 + v4**2 + v5**2 + v6**2 - v7**2
    axyz = ax * ay * az
    return np.where(np.abs(norm*axyz) > 1e-8, volume / (6 * axyz), a0 > 0), np.where(np.abs(norm*axyz) > 1e-8, area / (2 * axyz * norm), 0)

# center of mass of the surface of a cube cut
def cube_cut_area_com(a0, ax, ay, az):
    # approximate the trilinear integral with correct first order behavior
    sx, sy, sz = np.sign(ax), np.sign(ay), np.sign(az)
    ax, ay, az = np.abs(ax) + 1e-6, np.abs(ay) + 1e-6, np.abs(az) + 1e-6
    # cube cuts = sum of tetrahedrons
    norm = (ax**2 + ay**2 + az**2)**0.5
    ax /= norm; ay /= norm; az /= norm; a0 = a0 / norm
    v0 = np.maximum(0, ( ax + ay + az)/2 + a0)
    v1 = np.maximum(0, (-ax + ay + az)/2 + a0)
    v2 = np.maximum(0, ( ax - ay + az)/2 + a0)
    v3 = np.maximum(0, ( ax + ay - az)/2 + a0)
    v4 = np.maximum(0, ( ax - ay - az)/2 + a0)
    v5 = np.maximum(0, (-ax + ay - az)/2 + a0)
    v6 = np.maximum(0, (-ax - ay + az)/2 + a0)
    v7 = np.maximum(0, (-ax - ay - az)/2 + a0)
    area = v0**2 - v1**2 - v2**2 - v3**2 + v4**2 + v5**2 + v6**2 - v7**2
    shape = (1,) * len(np.shape(a0)) + (3,)
    com = np.reshape((0, 0, 0), shape)
    d = (1/3) / np.stack((ax, ay, az), axis=-1)
    com=com+v0[...,None]**2 * (np.reshape((-1, -1, -1), shape)/2 + d * v0[...,None])
    com -=  v1[...,None]**2 * (np.reshape(( 1, -1, -1), shape)/2 + d * v1[...,None])
    com -=  v2[...,None]**2 * (np.reshape((-1,  1, -1), shape)/2 + d * v2[...,None])
    com -=  v3[...,None]**2 * (np.reshape((-1, -1,  1), shape)/2 + d * v3[...,None])
    com +=  v4[...,None]**2 * (np.reshape((-1,  1,  1), shape)/2 + d * v4[...,None])
    com +=  v5[...,None]**2 * (np.reshape(( 1, -1,  1), shape)/2 + d * v5[...,None])
    com +=  v6[...,None]**2 * (np.reshape(( 1,  1, -1), shape)/2 + d * v6[...,None])
    com -=  v7[...,None]**2 * (np.reshape(( 1,  1,  1), shape)/2 + d * v7[...,None])
    com *= np.stack((sx, sy, sz), axis=-1)
    com = np.where(area[...,None] != 0, com / area[...,None], np.reshape((0, 0, 0), shape))
    axyz = ax * ay * az
    area = np.where(np.abs(axyz) != 0, area / (2 * axyz), 0)
    return area, com

# gauss integration using the derivative of the fermi function as weight function (integration over const 1 is always 1)
def gauss_5_df(f, mu, beta):
    x = np.array([-8.211650879369324585, -3.054894371595123559, 0.0, 3.054894371595123559, 8.211650879369324585])
    w = np.array([0.0018831678927720540279, 0.16265409664449248517, 0.6709254709254709216, 0.16265409664449248517, 0.0018831678927720540279])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0)

# gauss integration using the derivative of the fermi function as weight function (integration over const 1 is always 1)
def gauss_7_df(f, mu, beta):
    x = np.array([-13.208619179276130495, -6.822258565704534483, -2.74015863439980345, 0.0, 2.74015863439980345, 6.822258565704534483, 13.208619179276130495])
    w = np.array([1.5006783863250833195e-05, 0.0054715468031045289346, 0.18481163647966422636, 0.61940361986673598773, 0.18481163647966422636, 0.0054715468031045289346, 1.5006783863250833195e-05])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0)

# gauss integration using the half-fermi function 1/(1+e^abs(x)) as weight function
def gauss_4_f(f, mu, beta):
    x = np.array([-5.7755299052817408167, -1.3141165029468411252, 1.3141165029468411252, 5.7755299052817408167])
    w = np.array([0.013822390808990555472, 0.48617760919100944106, 0.48617760919100944106, 0.013822390808990555472])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0) / beta * 2*np.log(2)

# gauss integration using the half-fermi function 1/(1+e^abs(x)) as weight function
def gauss_6_f(f, mu, beta):
    x = np.array([-10.612971636582431145, -4.7544317516063152596, -1.1799810705877835648, 1.1799810705877835648, 4.7544317516063152596, 10.612971636582431145])
    w = np.array([0.00013527426019966680491, 0.02778699826589691238, 0.47207772747390341905, 0.47207772747390341905, 0.02778699826589691238, 0.00013527426019966680491])
    x_smpl = x / beta + mu
    f_smpl = f(x_smpl)
    return np.sum(f_smpl * w.reshape((-1,) + (1,) * len(np.shape(f_smpl)[1:])), axis=0) / beta * 2*np.log(2)

# explicitly compute the value of the convolution of the states
# function with the derivative of the fermi function at a given point.
# takes a piecewise linear representation of the states function
# and returns the exact convolution.
def convolve_df(x, energy_smpl, states, beta, extrapolation='flat', extrapolation_point=None):
    def f(e):
        return 1 / (1 + np.exp(beta*e))
    #def F(e):
    #    return -np.log1p(np.exp(-beta*e)) / beta
    def F_diff(e0, e1):
        #return F(e1) - F(e0)
        return np.log(1/(1+np.exp(-beta*e1)) + 1/(np.exp(beta*e0) + np.exp(beta*(e0-e1)))) / beta
    def segment(x, x0, x1, y0, y1):
        return y1*f(x-x1) - y0*f(x-x0) + (y1-y0)/(x1-x0)*F_diff(x-x0, x-x1)
    s = np.sum(segment(x, np.roll(energy_smpl, 1), energy_smpl, np.roll(states, 1), states)[1:])
    # add segments/other functions as extrapolation on both side to get correct high temperature behavior
    if extrapolation == 'flat':
        if extrapolation_point is None:
            s += f(x - energy_smpl[0]) * states[0]
            s += f(energy_smpl[-1] - x) * states[-1]
        else:
            # assume N(mu) starts at 0 and ends at N(extrapolation_point[0]) = extrapolation_point[1]
            s += f(extrapolation_point[0] - x) * extrapolation_point[1]
    # TODO add an option for the free electron gas as extrapolation
    return s

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

import scipy
def _Li2(x):
    return scipy.special.spence(1-x)
# 0 at x=0
def _int_df(x):
    return 0.5 * np.tanh(x/2)
# 0 at x=±inf
def _int_xdf(x):
    x = -np.abs(x)
    ex = np.exp(x) # small
    return (x*ex)/(1 + ex) - np.log1p(ex)
# 0 at x=0
def _int_xxdf(x):
    # use symmetry to avoid cancellation
    sign = np.sign(x)
    x = -np.abs(x)
    ex = np.exp(x) # small
    f = (x*x*ex)/(1 + ex) - 2*x*np.log1p(ex)-(2*_Li2(-ex) + np.pi**2/6)
    return -sign * f
# integrate (ax^2+bx+c)(-df/de) from x0 to x1 analytically
def int_poly(x0, x1, a, b, c):
    return a * (_int_xxdf(x1) - _int_xxdf(x0)) + b * (_int_xdf(x1) - _int_xdf(x0)) + c * (_int_df(x1) - _int_df(x0))

def naive_fermi_energy(bands, electrons):
    i = round(np.prod(np.shape(bands)[:-1]) * electrons)
    return np.mean(np.sort(np.ravel(bands))[i:i+2])

def naive_energy(bands, T, mu):
    e = np.ravel(bands)
    return np.mean(e / (1 + np.exp((e - mu) / (k_B * T)))) * np.shape(bands)[-1]

class DensityOfStates:
    def __init__(self, model, N=24, ranges=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)), wrap=True):
        if wrap:
            xyz = [np.linspace(*r, N, endpoint=False) + 1/2/N*(r[1] - r[0]) for r in ranges]
        else:
            xyz = [np.linspace(*r, N+1) for r in ranges]
        self.k_smpl = np.stack(np.meshgrid(*xyz, indexing='ij'), axis=-1)
        self.wrap = wrap
        shape = np.shape(self.k_smpl)
        bands = model(np.reshape(self.k_smpl, (-1, 3)))
        self.model = model
        self.bands = np.reshape(bands, shape[:-1]+(-1,))
        # TODO compute better band ranges using a Newton step to find the actual extrema in k
        self.bands_range = [(np.min(self.bands[...,i]), np.max(self.bands[...,i])) for i in range(self.bands.shape[-1])]
        # preprocessing the cubes halves he computation time
        self.cubes = [cubes_preprocessing(self.bands[...,i], wrap=wrap) for i in range(len(self.bands_range))]
    
    def model_bandcount(self):
        return len(self.bands_range)

    def states_below(self, energy):
        states = 0.0
        for i, band_range in enumerate(self.bands_range):
            if band_range[0] < energy < band_range[1]:
                states += np.mean(cube_cut_volume(energy - self.cubes[i][0], *self.cubes[i][1:]))
            elif band_range[1] <= energy:
                states += 1.0 # completely full
        return states
    
    # returns states_below, density
    def states_density(self, energy):
        states = 0.0
        density = 0.0
        for i, band_range in enumerate(self.bands_range):
            if band_range[0] < energy < band_range[1]:
                volume, area = cube_cut_volume_area(energy - self.cubes[i][0], *self.cubes[i][1:])
                states += np.mean(volume)
                density += np.mean(area)
            elif band_range[1] <= energy:
                states += 1.0 # completely full
        return states, density
    
    # returns density
    def density(self, energy):
        density = 0.0
        for i, band_range in enumerate(self.bands_range):
            if band_range[0] < energy < band_range[1]:
                density += np.mean(cube_cut_dvolume(energy - self.cubes[i][0], *self.cubes[i][1:]))
        return density
    
    # returns density for a specific band
    def density_band(self, energy, i):
        if self.bands_range[i][0] < energy < self.bands_range[i][1]:
            return np.mean(cube_cut_dvolume(energy - self.cubes[i][0], *self.cubes[i][1:]))
        return 0.0
    
    # returns density but split into the band contributions
    def density_bands(self, energy):
        return [self.density_band(energy, i) for i in range(self.model_bandcount())]
    
    # get the indices of the bands, that cut the given energy
    def cut_band_indices(self, energy):
        indices = []
        for i, band_range in enumerate(self.bands_range):
            if band_range[0] < energy < band_range[1]:
                indices.append(i)
        return indices
    
    # compute the full density of states curve
    # returns energy_smpl, states, density
    def full_curve(self, N=10, T=0):
        # compute energy samples based on where the bands start and end
        energy_smpl = []
        for (min_e, max_e) in self.bands_range:
            energy_smpl.extend(list(np.linspace(min_e, max_e, N)))
        energy_smpl = sorted(list(set(energy_smpl)))
        states = []
        density = []
        for energy in energy_smpl:
            states_, density_ = self.states_density(energy)
            states.append(states_)
            density.append(density_)
        if T != 0:
            # smooth the curve
            beta = 1 / (k_B * T) # 1/eV
            energy_smpl_T0, states_T0, density_T0 = energy_smpl, states, density
            delta = 5.0 / beta
            energy_smpl = np.array(list(np.linspace(energy_smpl[0] - delta, energy_smpl[0], N, endpoint=False)) + list(energy_smpl) + list(reversed(list(np.linspace(energy_smpl[-1] + delta, energy_smpl[-1], N, endpoint=False)))))
            states = [convolve_df(x, energy_smpl_T0, states_T0, beta) for x in energy_smpl]
            density = [convolve_df(x, energy_smpl_T0, density_T0, beta) for x in energy_smpl]
        return np.array(energy_smpl), np.array(states), np.array(density)
    
    # for a given number of electrons per cell (float in general)
    # compute the fermi energy (correct for metals and isolators)
    def fermi_energy(self, electrons, tol=1e-8, maxsteps=30):
        # first approximation from integer number of electrons
        assert electrons >= 0 and electrons <= len(self.bands_range)
        e_int = round(electrons)
        if e_int > 0 and e_int < len(self.bands_range):
            max_below = self.bands_range[e_int-1][1]
            min_above = self.bands_range[e_int][0]
            if e_int == electrons:
                fermi_energy = (max_below + min_above) / 2
                if max_below < min_above:
                    return fermi_energy # isolator (can only happen at integer electrons)
            elif e_int > electrons:
                fermi_energy = max_below
            else:
                fermi_energy = min_above
        elif e_int == 0:
            fermi_energy = self.bands_range[0][0]
            if electrons == 0:
                return fermi_energy
        else:
            assert e_int == len(self.bands_range)
            fermi_energy = self.bands_range[-1][1]
            if electrons == e_int:
                return fermi_energy
        for _ in range(maxsteps):
            states, density = self.states_density(fermi_energy)
            # TODO handle density = 0 using bisection (isolators are already handles above...)
            fermi_energy -= (states - electrons) / density
            if abs((states - electrons) / density) / fermi_energy <= tol:
                return fermi_energy
        return fermi_energy
        #raise ValueError(f"root search didn't converge in {maxsteps} steps. {abs((states - electrons) / density) / fermi_energy} > {tol}.")

    # returns the approximate bandgap if the model describes an isolator, 0 if it describes a metal
    def bandgap(self, electrons):
        # first approximation from integer number of electrons
        assert electrons >= 0 and electrons <= len(self.bands_range)
        e_int = round(electrons)
        if e_int != electrons:
            return 0.0 # only integers can make isolators
        assert e_int > 0 and e_int < len(self.bands_range)
        max_below = self.bands_range[e_int-1][1]
        min_above = self.bands_range[e_int][0]
        if max_below < min_above:
            # TODO improve this calculation with self.model.bands_grads(...)
            return min_above - max_below # isolator
        return 0.0 # metal
    
    # returns the minimum and maximum energy in the used band structure
    def energy_range(self):
        return self.bands_range[0][0], self.bands_range[-1][1]

    def chemical_potential(self, electrons, T_smpl, N=30, tol=1e-8, maxsteps=30):
        fermi_energy = self.fermi_energy(electrons, tol=tol, maxsteps=maxsteps)
        # use a good distribution for energy_smpl,
        # such that this works for small T!
        # the distribution needs to distribute the error terms
        # to be equal on all summands in convolve_df
        # To optimize the performance, all T > ... can be performed with the same distribution linear distribution
        e0, e1 = self.energy_range()
        # any temperature bigger than this can be sampled linear without precision loss
        T_lin = 0.25 * 0.25 * 2/k_B*max(abs(e0-fermi_energy), abs(e1-fermi_energy))
        energy_smpl_lin = np.linspace(e0, e1, N)
        T_smpl = np.asarray(T_smpl).reshape(-1)
        if np.max(T_smpl) >= T_lin:
            states_lin = [self.states_below(energy) for energy in energy_smpl_lin]
        beta = -10000.0 # invalid value
        res = []
        for T in T_smpl:
            if T <= 0:
                res.append(fermi_energy)
            elif T >= T_lin:
                beta = 1 / (k_B * T) # 1/eV
                res.append(secant(electrons, lambda x: convolve_df(x, energy_smpl_lin, states_lin, beta, extrapolation='flat'), fermi_energy, fermi_energy + 0.1, tol, maxsteps))
            else:
                new_beta = 1 / (k_B * T) # 1/eV
                if abs(beta - new_beta) / new_beta > 5e-2:
                    # recalculate distribution for precision
                    beta = 0.25 / (k_B * T) # 1/eV
                    energy_smpl = np.log(1 / (1e-16 + np.linspace(1/(1 + np.exp(beta*(e0-fermi_energy))), 1/(1 + np.exp(beta*(e1-fermi_energy))), N)) - 1 + 1e-16) / beta + fermi_energy
                    beta = new_beta
                    states = [self.states_below(energy) for energy in energy_smpl]
                # keep distribution if it doesn't cause too big errors (performance)
                beta = new_beta
                res.append(secant(electrons, lambda x: convolve_df(x, energy_smpl, states, beta, extrapolation='flat', extrapolation_point=(e1, self.model_bandcount())), fermi_energy, fermi_energy + 1/beta, tol, maxsteps))
        return np.array(res)
    
    # TODO currently WRONG
    def energy(self, T, electrons, mu=None, N=30):
        if mu is None:
            mu = self.chemical_potential(electrons, [T], N=N)
        # energy is integral e f(e) rho(e) de
        #         = integral (f + e df/de) N(e) de
        # like usual assume N to be piecewise linear and do the integral
        # = integral (f + e df/de) (a e + b) de
        # = integral (a e + b) f + (a e^2 + b e) df/de de
        # = [(a e^2/2 + b e) f] + integral -(a e^2/2 + b e) df/de + (a e^2 + b e) df/de de
        # = [(a e^2/2 + b e) f] + integral -a e^2/2 df/de de
        # Note the shift by mu can be done by
        # integral (e+mu) f(e) rho(e+mu) de = mu*q + integral e f(e) rho(e) de
        # where q is the number electrons per cell
        def accumulate(T, energy_smpl, states):
            def segment(T, x0, x1, y0, y1):
                dx = x1 - x0
                dy = y1 - y0
                beta = 1/(k_B * T)
                a = dy/dx
                b = y0 - x0 * a
                # TODO ERROR
                simple_term = (a*x1*x1/2 + b*x1) / (1 + np.exp((x1-mu)*beta)) - (a*x0*x0/2 + b*x0) / (1 + np.exp((x0-mu)*beta))
                return simple_term + a/2 * int_poly((x0-mu)*beta, (x1-mu)*beta, 1/beta/beta, 0, 0)/beta
            s = np.sum(segment(T, np.roll(energy_smpl, 1), energy_smpl, np.roll(states, 1), states)[1:])
            # can add segments/other functions as extrapolation on both side to get correct high temperature behavior
            # however if they are flat, they have no contribution.
            return s
        # calculate custom distribution for precision/cost balance
        beta = 0.25 / (k_B * T) # 1/eV
        e0, e1 = self.energy_range()
        energy_smpl = np.log(1 / (1e-16 + np.linspace(1/(1 + np.exp(beta*(e0-mu))), 1/(1 + np.exp(beta*(e1-mu))), N)) - 1 + 1e-16) / beta + mu
        states = [self.states_below(energy) for energy in energy_smpl]
        return accumulate(T, energy_smpl, states)

    # volumetric heat capacity in eV/K
    # TODO currently WRONG
    def heat_capacity(self, T, mu, N=30):
        # The heat capacity is du/dT where u is the energy per unit cell
        # u = integral e*rho(e)*f(e) de = -integral N(e)*(f(e)+e df/de(e)) de
        # du/dT = -integral N(e)*(-(e-mu)/T*df/de(e)+e/T*(-df/de(e) - (e-mu)*d2f/de2(e))) de
        # this integral has multiple parts. assume N(e) to be piecewise linear, then there are the following integrals
        #  (I) integral e^n df/de(e) with n=0,1,2
        # (II) integral e^n d2f/de2(e) with n=1,2,3
        # now solve these analytically.
        # integral e^n d2f/de2(e) = [e^n df/de(e)] - integral n e^(n-1) df/de(e) -> reduction to type (I)
        # for (I) the solutions are already implemented for n=0,1 in convolve_df
        # the solution for n=2 includes the Li_2(x) (dilogarithm, can be used from scipy)
        def accumulate(T, energy_smpl, states):
            def segment(T, x0, x1, y0, y1):
                dx = x1 - x0
                dy = y1 - y0
                beta = 1/(k_B * T)
                a, b, c = 1, -mu, 0
                simple_term = ... # TODO!
                return simple_term/T + dy/dx/T * int_poly(x0*beta, x1*beta, a/beta/beta, b/beta, c)/beta
            s = np.sum(segment(T, np.roll(energy_smpl, 1), energy_smpl, np.roll(states, 1), states)[1:])
            # can add segments/other functions as extrapolation on both side to get correct high temperature behavior
            # however if they are flat, they have no contribution.
            return s
        # calculate custom distribution for precision/cost balance
        beta = 0.25 / (k_B * T) # 1/eV
        e0, e1 = self.energy_range()
        energy_smpl = np.log(1 / (1e-16 + np.linspace(1/(1 + np.exp(beta*(e0-mu))), 1/(1 + np.exp(beta*(e1-mu))), N)) - 1 + 1e-16) / beta + mu
        states = [self.states_below(energy) for energy in energy_smpl]
        return accumulate(T, energy_smpl, states)

    # compute points on the (fermi) surface at the given (fermi) energy.
    # if the improved keyword argument is True, the results will be much more precise
    # at the cost of an integration over the fermi surface.
    # returns points, weights
    def fermi_surface_samples(self, energy, improved=True, normalize=None):
        assert normalize in [None, "band", "total"], "normalize needs to be None, 'band' or 'total'"
        # use cube_cut_area_com() to get points, then
        # use the approximate gradients to do a newton step
        # to get the points even closer to the fermi surface.
        weights = []
        band_indices = []
        points = []
        # TODO this only works for cubes! What about cuboids? Parallelograms?
        size = np.linalg.norm(self.k_smpl[1,0,0] - self.k_smpl[0,0,0])
        centers = (self.k_smpl + np.roll(self.k_smpl, shift=(-1, -1, -1), axis=(0, 1, 2))) / 2
        if not self.wrap:
            centers = centers[:-1,:-1,:-1]
        total_w_sum = 0
        for i in self.cut_band_indices(energy):
            w, x = cube_cut_area_com(self.cubes[i][0] - energy, *self.cubes[i][1:])
            # transform x into the the cubes
            x *= size
            x += centers
            x = x.reshape(-1, 3)
            w = np.ravel(w)
            select = w > 1e-4
            w = w[select]
            x = x[select]
            if improved:
                bands = self.model(x)[:,i:i+1] - energy
                x += size * np.reshape(self.cubes[i][1:], (3, -1)).T[select] * (bands / (1e-20 + np.linalg.norm(np.reshape(self.cubes[i][1:], (3, -1)), axis=0)[select][...,None]**2))
            w_sum = np.sum(w)
            total_w_sum += w_sum
            if normalize == 'band':
                w /= w_sum
            points.extend(x)
            weights.extend(w)
            band_indices.extend([i] * len(w))
        assert len(points) == len(weights)
        w = np.array(weights)
        if normalize == 'total':
            w /= total_w_sum
        elif normalize is None:
            w /= len(centers[...,0].flat) * size # correct area measure weights
        total_area = total_w_sum / (len(centers[...,0].flat) * size)
        return np.array(points).reshape(-1, 3), np.array(band_indices), w, total_area