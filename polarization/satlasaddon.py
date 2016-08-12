"""
.. module:: polar
    :platform: Windows
    :synopsis: Implementation of a class for optical
     pumping simulations.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
import numpy as np
from satlas.profiles import Voigt, Lorentzian
from satlas.basemodel import BaseModel
from scipy import integrate
from sympy.physics.wigner import wigner_6j, wigner_3j
import scipy.constants as csts
from satlas import lmfit
import itertools

W6J = wigner_6j
W3J = wigner_3j

__all__ = ['Polar']

# Define constants
C = csts.physical_constants['speed of light in vacuum'][0]  # Speed of light, m/s
H = csts.physical_constants['Planck constant'][0]  # Planck's constant, Js
PI = np.pi  # pi...
GL = 1.0  # Orbital g-factor
GS = 2.00232  # Spin g-factor
MUB = csts.physical_constants['Bohr magneton'][0]  # Bohr magneton
EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6  # eV to MHz conversion factor


#######################
# CALCULATION OBJECTS #
#######################

class BxRho_Voigt(Voigt):
    def __init__(self, A=None, fwhmG=None, mu=None, laser=None, fwhmL=None):
        self._fwhmG = fwhmG
        self._fwhmL = fwhmL
        self._laser = laser
        self._A = A
        self._lorentzian = 0
        super(BxRho_Voigt, self).__init__(mu=mu, fwhm=[fwhmG, fwhmL], ampIsArea=True, amp=1.0)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value
        self.set_factor()

    @property
    def gaussian(self):
        return self._gaussian

    @gaussian.setter
    def gaussian(self, value):
        self._gaussian = value
        self.fwhm = [value, self.fwhmL]

    @property
    def lorentzian(self):
        return self._lorentzian

    @lorentzian.setter
    def lorentzian(self, value):
        self._lorentzian = value
        self.fwhm = [self.fwhmG, value]

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value * 1e6
        self.set_factor()

    @property
    def laser(self):
        return self._laser

    @laser.setter
    def laser(self, value):
        self._laser = value
        self.set_factor()

    def set_factor(self):
        self._factor = self.A * self.laser * C * C / (8 * PI * H * self.mu * self.mu)

    def __call__(self, x):
        return super(BxRho_Voigt, self).__call__(x*1e6) * self._factor / (x*1e6)

class BxRho_Lorentzian(Lorentzian):
    def __init__(self, A=None, mu=None, laser=None, fwhm=None):
        self._laser = laser
        self._A = A
        super(BxRho_Lorentzian, self).__init__(mu=mu, fwhm=fwhm, ampIsArea=True, amp=1.0)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value
        self.set_factor()

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value * 1e6
        self.set_factor()

    @property
    def laser(self):
        return self._laser

    @laser.setter
    def laser(self, value):
        self._laser = value
        self.set_factor()

    def set_factor(self):
        self._factor = self.A * self.laser * C * C / (8 * PI * H * self.mu * self.mu)

    def __call__(self, x):
        return super(BxRho_Lorentzian, self).__call__(x * 1e6) * self._factor / (x * 1e6)

##############
# MAIN CLASS #
##############
class RateModel(BaseModel):
    def __init__(self, I, J, L, ABC, centroids, energies, A_array, scale=1.0, shape='Voigt', laser_intensity=80, laser_mode=None, interaction_time=1e-6, fwhmG=0.1, fwhmL=None, background=0, field=0, fixed_frequencies=None, frequency_mode='fixed'):
        super(RateModel, self).__init__()
        self.I = I
        self.J = J
        self.L = L
        self.A_array = A_array
        self.shape = shape

        try:
            lasers = len(laser_intensity)
        except:
            laser_intensity = [laser_intensity]
            laser_mode = [laser_mode]
        if fixed_frequencies is not None:
            self.fixed_frequencies = fixed_frequencies
        else:
            self.fixed_frequencies = []
        self.vary_freqs = len(laser_intensity) - len(self.fixed_frequencies)
        self.frequency_mode = frequency_mode

        self.laser_intensity = laser_intensity
        self.mode = laser_mode

        self._calculate_F_levels()
        self._set_energies(energies)
        self._calculate_energy_coefficients()

        self._params = self._populate_params(laser_intensity, ABC, centroids, shape, scale, fwhmG, fwhmL, interaction_time, background, field)
        self._set_population()
        self._calculate_A_partial()
        self._calculate_energy_changes()
        self._create_D_matrix()

        self.params = self._params

    @property
    def params(self):
        return self._check_variation(self._params)

    @params.setter
    def params(self, params):
        self._params = self._check_variation(params)
        self._calculate_energy_changes()
        A = np.zeros((self.level_counts_cs[-1], self.level_counts_cs[-1]))
        for key in self.transition_indices:
            for x, y in self.transition_indices[key]:
                A[x, y] = params[key].value
        A = A * self.partial_A
        A = np.transpose(A) - np.eye(A.shape[0]) * A.sum(axis=1)
        self.A_array_used = A
        self.decay_matrix = np.abs(self.A_array_used * np.eye(self.A_array_used.shape[0]))
        self._edit_D_matrix()

    def _set_energies(self, energies):
        N = self.level_counts.sum()
        # Pre-allocate the energy and population vectors.
        E = np.zeros(N)
        Nlevcs = self.level_counts.cumsum()
        for i, (n, ncs) in enumerate(zip(self.level_counts, Nlevcs)):
            E[ncs - n:ncs] = energies[i]
        self.energies = E * EV_TO_MHZ

    def _populate_params(self, laser_intensity, ABC, centroids, shape, scale, fwhmG, FWHML, interaction_time, background, field):
        p = lmfit.Parameters()
        for i, val in enumerate(laser_intensity):
            p.add('Laser_intensity_' + str(i), value=val, min=0, max=None)
        for i, j in enumerate(self.Jlist):
            p.add('A_level_' + str(i), value=ABC[i][0])
            p.add('B_level_' + str(i), value=ABC[i][1])
            p.add('C_level_' + str(i), value=ABC[i][2])
            if not i == len(self.Jlist)-1:
                p.add('Centroid_level_' + str(i), value=centroids[i])
            else:
                p.add('Centroid_level_' + str(i), value=0, vary=False)
        for i, _ in enumerate(self.level_counts):
            for j, _ in enumerate(self.level_counts):
                if i < j and np.isfinite(self.A_array[i, j]):
                    p.add('Transition_strength_' + str(i) + '_to_' + str(j), value=self.A_array[i, j], min=0, vary=False)
                    fwhmL = self.A_array[i, j]/(2*PI)*1e-6 if FWHML is None else FWHML
                    p.add('FWHML_' + str(i) + '_to_' + str(j), value=fwhmL, min=0)
                    if shape.lower() == 'voigt':
                        par_lor_name = 'FWHML_' + str(i) + '_to_' + str(j)
                        par_gauss_name = 'FWHMG_' + str(i) + '_to_' + str(j)
                        expr = '0.5346*{0}+(0.2166*{0}**2+{1}**2)**0.5'
                        p.add('FWHMG_' + str(i) + '_to_' + str(j), value=fwhmG, min=0.0001)
                        p.add('TotalFWHM_' + str(i) + '_to_' + str(j), value=0, vary=False, expr=expr.format(par_lor_name, par_gauss_name))
        p.add('Scale', value=scale)
        p.add('Interaction_time', value=interaction_time, min=0)
        p.add('Background', value=background)
        p.add('Field', value=field)
        return self._check_variation(p)

    def _check_variation(self, p):
        for key in self._vary:
            if key in p:
                p[key].vary = self._vary[key]

        for i, j in enumerate(self.Jlist):
            if j[0] < 1.5 or self.I < 1.5:
                p['C_level_' + str(i)].value = 0
                p['C_level_' + str(i)].vary = False
                if j[0] < 1.0 or self.I < 1.0:
                    p['B_level_' + str(i)].value = 0
                    p['B_level_' + str(i)].vary = False
                    if j[0] < 0.5 or self.I < 0.5:
                        p['A_level_' + str(i)].value = 0
                        p['A_level_' + str(i)].vary = False
        return p

    def _calculate_F_levels(self):
        I = self.I
        J = self.J
        L = self.L
        self.Flist = []
        self.MFlist = []
        self.Jlist = []
        self.Llist = []
        dummyJ = np.array([])
        dummyF = np.array([])
        dummyFz = np.array([])
        dummy = np.array([])
        dummyL = np.array([])
        for i, (j, l) in enumerate(zip(J, L)):
            F = np.arange(np.abs(j - I), j + I + 1)  # Values of F

            Flen = (2 * F + 1).astype('int')  # Lengths of F_z
            starts = np.cumsum(np.append([0], Flen[:-1]))  # Index for different F states

            # Pre-allocate
            f = np.zeros(int((2 * F + 1).sum()))  # F-states
            mz = np.zeros(int((2 * F + 1).sum()))  # F_z-states

            # Fill the pre-allocated arrays
            for i, (entry, start) in enumerate(zip(Flen, starts)):
                mz[start:start + entry] = np.arange(-F[i], F[i] + 1)
                f[start:start + entry] = F[i]
            self.Flist.append(f)
            self.MFlist.append(mz)
            self.Jlist.append([j]*len(f))
            self.Llist.append([l]*len(f))
            dummyF = np.append(dummyF, f)
            dummyFz = np.append(dummyFz, mz)
            dummyJ = np.append(dummyJ, np.ones(len(f))*j)
            dummyL = np.append(dummyL, np.ones(len(f))*l)
            dummy = np.append(dummy, np.array([len(f)]))
        self.F = dummyF
        self.Mf = dummyFz
        self.J = dummyJ
        self.L = dummyL
        self.level_counts = dummy.astype('int')
        self.level_counts_cs = self.level_counts.cumsum()

    def _calculate_energy_coefficients(self):
        S = 0.5
        L = self.L
        # Since I, J and F do not change, these factors can be calculated once
        # and then stored.
        I, J, F = self.I, self.J, self.F
        C = (F*(F+1) - I*(I+1) - J*(J + 1)) * (J/J) if I > 0 else 0 * J  #*(J/J) is a dirty trick to avoid checking for J=0
        D = (3*C*(C+1) - 4*I*(I+1)*J*(J+1)) / (2*I*(2*I-1)*J*(2*J-1))
        E = (10*(0.5*C)**3 + 20*(0.5*C)**2 + C*(-3*I*(I+1)*J*(J+1) + I*(I+1) + J*(J+1) + 3) - 5*I*(I+1)*J*(J+1)) / (I*(I-1)*(2*I-1)*J*(J-1)*(2*J-1))
        C = np.where(np.isfinite(C), 0.5 * C, 0)
        D = np.where(np.isfinite(D), 0.25 * D, 0)
        E = np.where(np.isfinite(E), E, 0)

        gJ = GL * (J * (J + 1) + L * (L + 1) - S * (S + 1)) / \
                (2 * J * (J + 1)) + GS * (J * (J + 1) - L * (L + 1) + S *
                                          (S + 1)) / (2 * J * (J + 1))
        gJ = np.where(np.isfinite(gJ), gJ, 0)
        gF = gJ * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / \
                    (2 * F * (F + 1))
        gF = np.where(np.isfinite(gF), -gF, 0)
        self.A_coeff, self.B_coeff, self.C_coeff, self.field_coeff = C, D, E, gF * MUB * self.Mf * ((10 ** (-6)) / H)

    def _calculate_energy_changes(self):
        field = self._params['Field'].value
        A = np.zeros(self.level_counts_cs[-1])
        B = np.zeros(self.level_counts_cs[-1])
        C = np.zeros(self.level_counts_cs[-1])
        centr = np.zeros(self.level_counts_cs[-1])
        for i, (ncs, n) in enumerate(zip(self.level_counts_cs, self.level_counts)):
            A[ncs-n:ncs] = self._params['A_level_' + str(i)].value
            B[ncs-n:ncs] = self._params['B_level_' + str(i)].value
            C[ncs-n:ncs] = self._params['C_level_' + str(i)].value
            centr[ncs-n:ncs] = self._params['Centroid_level_' + str(i)].value
        self.energy_change = centr + self.A_coeff * A + self.B_coeff * B + self.C_coeff * C + self.field_coeff * field

    def _set_population(self, level=-1):
        try:
            levels = len(level)
        except:
            levels = 1
            level = [level]
        total_number = sum(self.level_counts[level])
        P = np.zeros(self.level_counts_cs[-1])
        for lev in level:
            N = self.level_counts_cs[lev]
            P[N - self.level_counts[lev]:N] = 1.0 / total_number
        self.P = P

    def _calculate_A_partial(self):
        I = self.I
        J = self.Jlist
        F = self.Flist
        Mf = self.MFlist
        N = self.level_counts_cs[-1]
        self.partial_A = np.zeros((N, N))
        self.transition_indices = {}
        for i, _ in enumerate(self.level_counts):
            for j, _ in enumerate(self.level_counts):
                if i < j and not np.isclose(self.A_array[i, j], 0):
                    indices_ex = []
                    indices_gr = []
                    for k, (Jex, Fe, Mze) in enumerate(zip(J[i], F[i], Mf[i])):
                        for l, (Jgr, Fg, Mzg) in enumerate(zip(J[j], F[j], Mf[j])):
                            A = float((2 * Jex + 1) * (2 * Fe + 1) * (2 * Fg + 1))
                            W3 = W3J(Fg, 1.0, Fe, -Mzg, Mzg - Mze, Mze)
                            W6 = W6J(Jgr, Fg, I, Fe, Jex, 1.0)
                            A = A * (W3 ** 2)
                            A = A * (W6 ** 2)
                            x = self.level_counts_cs[i] - self.level_counts[i] + k
                            y = self.level_counts_cs[j] - self.level_counts[j] + l
                            self.partial_A[x, y] = A
                            indices_ex.append(x)
                            indices_gr.append(y)
                    self.transition_indices['Transition_strength_' + str(i) + '_to_' + str(j)] = list(zip(indices_ex, indices_gr))

    def _create_D_matrix(self):
        N = self.level_counts_cs[-1]
        D = np.zeros((N, N, len(self.laser_intensity)), dtype='object')
        bxrho = BxRho_Voigt if self.shape.lower() == 'voigt' else BxRho_Lorentzian

        self.indices = []
        for laser_index, laser in enumerate(self.laser_intensity):
            for i, j in itertools.combinations(range(len(self.level_counts)), 2):
                for k, (fe, mze) in enumerate(zip(self.Flist[i], self.MFlist[i])):
                    for l, (fg, mzg) in enumerate(zip(self.Flist[j], self.MFlist[j])):
                        x = self.level_counts_cs[i] - self.level_counts[i] + k
                        y = self.level_counts_cs[j] - self.level_counts[j] + l
                        if np.isclose(self.A_array[i, j], 0) or np.isclose(self.partial_A[x, y], 0):
                            continue
                        frac = 1.0 if self.mode[laser_index] == (mze - mzg) else 0
                        if frac == 0:
                            pass
                        else:
                            intensity = self._params['Laser_intensity_' + str(laser_index)].value
                            A = self._params['Transition_strength_' + str(i) + '_to_' + str(j)].value
                            mu = (self.energies[k] + self.energy_change[k]) - (self.energies[l] + self.energy_change[l])
                            kwargs = {'A': A, 'mu': mu, 'laser': intensity}
                            if self.shape.lower() == 'voigt':
                                kwargs['fwhmG'] = self._params['FWHMG_' + str(i) + '_to_' + str(j)].value * 1e6
                                kwargs['fwhmL'] = self._params['FWHML_' + str(i) + '_to_' + str(j)].value * 1e6
                            else:
                                kwargs['fwhm'] = self._params['FWHML_' + str(i) + '_to_' + str(j)].value * 1e6
                            D[x, y, laser_index] = bxrho(**kwargs)
                            self.indices.append((x, y, laser_index, i, j))

        self.D = D

    def _edit_D_matrix(self):
        self.locations = []
        self.transitions = []
        for x, y, laser_index, i, j in self.indices:
            intensity = self._params['Laser_intensity_' + str(laser_index)].value
            A = self.A_array_used[y, x]
            mu = (self.energies[x] + self.energy_change[x]) - (self.energies[y] + self.energy_change[y])
            self.D[x, y, laser_index].mu = mu
            self.D[x, y, laser_index].A = A
            if self.shape.lower() == 'voigt':
                self.D[x, y, laser_index].gaussian = self._params['FWHMG_' + str(i) + '_to_' + str(j)].value * 1e6
                self.D[x, y, laser_index].lorentzian = self._params['FWHML_' + str(i) + '_to_' + str(j)].value * 1e6
            else:
                self.D[x, y, laser_index].fwhm = self._params['FWHML_' + str(i) + '_to_' + str(j)].value * 1e6
            self.D[x, y, laser_index].laser = intensity
            self.locations.append(mu)
            self.transitions.append((self.F[x], self.F[y]))
        self.locations, indices = np.unique(np.array(self.locations), return_index=True)
        self.transitions = np.array(self.transitions)[indices]

    def _evaluate_matrices(self, f):
        D = np.zeros(self.D.shape)
        for i, j, laser_index, _, _ in self.indices:
            if laser_index < self.vary_freqs:
                freq = f
            else:
                freq = self.fixed_frequencies[laser_index - self.vary_freqs]
                if self.frequency_mode.lower() == 'offset':
                    freq += f

            D[i, j, laser_index] = self.D[i, j, laser_index](freq)

        D = D.sum(axis=2)
        D = np.transpose(D) + D
        D = D - np.eye(D.shape[0]) * D.sum(axis=1)
        self.M = self.A_array_used + D
        self.decay_matrix = np.abs(np.diag(np.diag(self.M)))

    def _rhsint(self, y, t):
        """Define the system of ODE's for use in the odeint method from SciPy.
        Note that the input is (y, t)."""
        return np.dot(self.M, y)

    def _process_population(self, y):
        raise NotImplementedError('Function should be implemented in child classes!')

    def __call__(self, x):
        try:
            response = np.zeros(len(x))
            for i, f in enumerate(x):
                self._evaluate_matrices(f)
                dt = self._params['Interaction_time'].value / 400
                y = integrate.odeint(self._rhsint, self.P, np.arange(0, self._params['Interaction_time'].value, dt))[-1, :]
                response[i] = self._process_population(y)
        except:
            self._evaluate_matrices(x)
            dt = self._params['Interaction_time'].value / 400
            y = integrate.odeint(self._rhsint, self.P, np.arange(0, self._params['Interaction_time'].value, dt))[-1, :]
            response = self._process_population(y)
        return self._params['Scale'].value * response + self._params['Background'].value

class RateModelDecay(RateModel):
    def _process_population(self, y):
        return np.dot(self.decay_matrix, y).sum()

class RateModelPolar(RateModel):
    def __init__(self, *args, **kwargs):
        super(RateModelPolar, self).__init__(*args, **kwargs)
        self._convertFMftoMIMJ()

    def _convertFMftoMIMJ(self):
        self.MIlist = []
        self.MJlist = []
        self.MI = np.array([])
        self.MJ = np.array([])
        for i, (J, F, Mf) in enumerate(zip(self.Jlist, self.Flist, self.MFlist)):
            I = self.I
            A = self._params['A_level_' + str(i)].value
            J = J[0]

            # Create the array of possible F-values.
            f = np.arange(np.abs(I - J), I + J + 1)

            # Create grids of MI and MJ
            I = np.arange(-I, I + 1)
            J = np.arange(-J, J + 1)
            I, J = np.meshgrid(I, J)

            # Calculate the total projection
            mf = I + J

            # Create an equal-size matrix with the correct
            # F-numbers in each place, depending on the sign of A
            M = np.zeros(I.shape)
            for i, val in enumerate(reversed(f)):
                if np.sign(A) == 1:
                    if i != 0:
                        M[0:-i, i] = val
                        M[-i - 1, i:] = val
                    else:
                        M[:, 0] = val
                        M[-1, :] = val
                else:
                    M[i, 0:- 1 - i] = val
                    M[i:, - 1 - i] = val

            f_select = []
            m_select = []
            for f, m in zip(F, Mf):
                f_select.append(np.isclose(M, f))
                m_select.append(np.isclose(mf, m))
            MI = []
            MJ = []
            for f, mf in zip(f_select, m_select):
                MI.append(I[np.bitwise_and(f, mf)][0])
                MJ.append(J[np.bitwise_and(f, mf)][0])
            self.MIlist.append(MI)
            self.MJlist.append(MJ)
            self.MI = np.append(self.MI, np.array(MI))
            self.MJ = np.append(self.MJ, np.array(MJ))

    def _process_population(self, y):
        return np.dot(self.MI, y) / self.I
