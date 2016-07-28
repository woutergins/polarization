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

class BxRho_Voigt(Voigt)
    def __init__(self, A=None, fwhmG=None, mu=None, laser=None):
        self._fwhmG = fwhmG
        self._laser = laser
        self._A = A
        super(BxRho_Voigt, self).__init__(mu=mu, fwhm=[fwhmG, 1/(A*2*PI)], ampIsArea=True, amp=1.0)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value
        temp = self._fwhm
        temp[1] = 1/(value * 2 * PI)
        self.fwhm = temp
        self.set_factor()

    @property
    def fwhmG(self):
        return self._fwhmG

    @fwhmG.setter
    def fwhmG(self, value):
        self._fwhmG = value
        temp = self._fwhm
        temp[0] = value
        self.fwhm = temp

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
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
        return super(BxRho_Voigt, self).__call__(x) * self._factor / x

class BxRho_Lorentzian(Lorentzian)
    def __init__(self, A=None, mu=None, laser=None):
        self._laser = laser
        self._A = A
        super(BxRho_Lorentzian, self).__init__(mu=mu, fwhm=1/(A*2*PI), ampIsArea=True, amp=1.0)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value
        self.fwhm = 1 / (value * 2 * PI)
        self.set_factor()

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
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
        return super(BxRho_Lorentzian, self).__call__(x) * self._factor / x

class BxRho(object):

    r"""BxRho calculation object.

    Parameters
    ----------
    ground : float
        Ground state energy in MHz. Used to calculate :math:`\nu_0`.
    excited : float
        Excited state energy in MHz. Used to calculate :math:`\nu_0`.
    A : float
        A :sub:`eg` value in s :sup:`-1`.
    G : float
        Inverted lifetime in s :sup:`-1`.
    Il : float
        Laser intensity in W/m :sup:`2`."""

    def __init__(self, ground, excited, A, G, Il):
        self.Aeg = A
        dE = np.abs(excited - ground)
        self.f0 = dE * 10 ** 6
        self.G = G
        self.SetFactor(np.abs(Il))
        self.L = Lorentzian(amp=1.0, fwhm=G / (2 * PI), mu=self.f0,
                            ampIsArea=True)

    def SetFactor(self, Il):
        self.factor = self.Aeg * Il * C * C / (8 * PI * H *
                                               self.f0 * self.f0)

    def __str__(self):
        s = '<BxRho object: A=%f, f0=%f, G=%f>' % (self.Aeg, self.f0, self.G)
        return s

    def __call__(self, x):
        r"""Response of BxRho for given frequency value in MHz.

        Parameters
        ----------
        x : array_like
            Frequencies/frequency for which the calculation has
            to be performed.

        Note
        ----
        The formula used is:

        .. math::
            B_{if}\rho\left(\nu\right) = \frac{A_{if} I c^2}{8\pi h^2 \nu_0^2
            \nu}\mathcal{L}\left(\nu;\nu_0,\frac{\Gamma}{4\pi}\right)."""
        return self.factor * self.L(x * 10 ** 6) / (x * 10 ** 6)


##############
# MAIN CLASS #
##############
class RateModel(BaseModel):
    def __init__(self, I, J, ABC, centroids, energies, A_array, scale=1.0, shape='Voigt', laser_intensity=80, laser_mode=None, interaction_time=1e-6, fwhmG=0.1, background=0):
        super(RateModel, self).__init__()
        self.I = I
        self.J = J
        self.A_array = A_array
        self.laser_intensity = laser_intensity
        self.shape = shape

        self._calculate_F_levels()
        self._set_energies(energies)
        self._calculate_energy_coefficients()

        self._params = self._populate_params(laser_intensity, ABC, centroids, shape, scale, fwhmG, interaction_time)
        self._set_population()
        self._calculate_A_partial()
        self._create_D_matrix()

        self.params = self._params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
        self._calculate_energy_change()
        A = np.zeros(self.level_counts_cumsum[-1])
        for key in self.transition_indices:
            for x, y in self.transition_indices[key]:
                A[x, y] = params[key].value
        A = np.transpose(A) - np.eye(A.shape[0]) * A.sum(axis=1)
        self.A_array_used = A * self.partial_A

    def _set_energies(self, energies):
        N = self.level_counts.sum()
        # Pre-allocate the energy and population vectors.
        E = np.zeros(N)
        Nlevcs = self.level_counts.cumsum()
        for i, (n, ncs) in enumerate(zip(self.level_counts, Nlevcs)):
            E[ncs - n:ncs] = energies[i]
        self.energies = E * EV_TO_MHZ

    def _populate_params(self, laser_intensity, ABC, centroids, shape, scale, fwhmG, interaction_time, background):
        p = lmfit.Parameters()
        p.add('Laser_intensity_[W/m^2]', value=laser_intensity, min=0, max=None)
        for i, j in enumerate(J):
            p.add('A_level_' + str(i), value=ABC[i][0])
            p.add('B_level_' + str(i), value=ABC[i][1])
            p.add('C_level_' + str(i), value=ABC[i][2])
            if not i == len(J)-1:
                p.add('Centroid_level_' + str(i), value=centroids[i])
        for i, _ in enumerate(self.level_counts):
            for j, _ in enumerate(self.level_counts):
                if i < j and np.isfinite(A_array[i, j]):
                    p.add('Transition_strength_' + str(i) + '_to_' + str(j), value=A_array[i, j], min=0)
        p.add('Scale', value=scale)
        p.add('Interaction_time', value=interaction_time, min=0)
        if shape == 'Voigt':
            p.add('FWHMG', value=fwhmG, min=0.0001)
        p.add('Background', value=background)
        return p

    def _calculate_F_levels(self):
        I = self.I
        J = self.J
        self.Flist = []
        self.MFlist = []
        self.Jlist = []
        dummyJ = np.array([])
        dummyF = np.array([])
        dummyFz = np.array([])
        dummy = np.array([])
        for i, j in enumerate(J):
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
            self.Jlist.append([len(f)]*j)
            dummyF = np.append(dummyF, f)
            dummyFz = np.append(dummyFz, mz)
            dummyJ = np.append(dummyJ, np.ones(len(f))*j)
            dummy = np.append(dummy, np.array([len(f)]))
        self.F = dummyF
        self.Mf = dummyFz
        self.J = dummyJ
        self.level_counts = dummy
        self.level_counts_cs = self.level_counts.cumsum()

    def _calculate_energy_coefficients(self):
        # Since I, J and F do not change, these factors can be calculated once
        # and then stored.
        I, J, F = self.I, self.J, self.F
        C = (F*(F+1) - I*(I+1) - J*(J + 1)) * (J/J) if I > 0 else 0 * J  #*(J/J) is a dirty trick to avoid checking for J=0
        D = (3*C*(C+1) - 4*I*(I+1)*J*(J+1)) / (2*I*(2*I-1)*J*(2*J-1))
        E = (10*(0.5*C)**3 + 20*(0.5*C)**2 + C*(-3*I*(I+1)*J*(J+1) + I*(I+1) + J*(J+1) + 3) - 5*I*(I+1)*J*(J+1)) / (I*(I-1)*(2*I-1)*J*(J-1)*(2*J-1))
        C = np.where(np.isfinite(C), 0.5 * C, 0)
        D = np.where(np.isfinite(D), 0.25 * D, 0)
        E = np.where(np.isfinite(E), E, 0)
        self.A_coeff, self.B_coeff, self.C_coeff = C, D, E

    def _calculate_energy_changes(self):
        A = np.zeros(self.level_counts.cumsum[-1])
        B = np.zeros(self.level_counts.cumsum[-1])
        C = np.zeros(self.level_counts.cumsum[-1])
        centr = np.zeros(self.level_counts.cumsum[-1])
        for i, (ncs, n) in enumerate(self.level_counts_cumsum, self.level_counts):
            A[ncs-n:ncs] = self._params['A_level_' + str(i)].value
            B[ncs-n:ncs] = self._params['B_level_' + str(i)].value
            C[ncs-n:ncs] = self._params['C_level_' + str(i)].value
            centr[ncs-n:ncs] = self._params['Centroid_level_' + str(i)].value
        self.energy_change = centr + self.C * A + self.D * B + self.E * C

    def _set_population(self):
        N = self.level_counts.cumsum[-1]
        P = np.zeros(N)
        P[N - self.level_counts[-1]:] = 1.0 / self.level_counts[-1]

    def _calculate_A_partial(self):
        I = self.I
        J = self.Jlist
        F = self.Flist
        Mf = self.MFlist
        self.partial_A = np.zeros(len(self.F))
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
                            x = self.level_counts_cumsum[i] - self.level_counts[i] + k
                            y = self.level_counts_cumsum[j] - self.level_counts[j] + l
                            self.partial_A[x, y] = A
                            indices_ex.append(x)
                            indices_gr.append(y)
                    self.transition_indices['Transition_strength_' + str(i) + '_to_' + str(j)] = list(zip(indices_ex, indices_gr))
        self.partial_A = np.transpose(self.partial_A) - np.eye(self.partial_A.shape[0]) * self.partial_A.sum(axis=1)

    def _create_D_matrix(self):
        N = self.level_counts_cumsum[-1]
        D = np.zeros((N, N), dtype='object')
        bxrho = BxRho_Voigt if self.shape.lower() == 'voigt' else BxRho_Lorentzian

        for i, left in enumerate(self.level_counts):
            for j, right in enumerate(self.level_counts):
                if i < j and not np.isclose(self.A_array[i, j], 0):
                    for k, (fe, mze) in enumerate(zip(self.Flist[i], self.MFlist[i])):
                        for l, (fg, mzg) in enumerate(zip(self.Flist[j], self.MFlist[j])):
                            x = self.level_counts_cumsum[i] - self.level_counts[i] + k
                            y = self.level_counts_cumsum[j] - self.level_counts[j] + l

                            frac = 1.0 if self.mode == (mze - mzg) else 0
                            if frac == 0:
                                pass
                            else:
                                intensity = self._params['Laser_intensity'].value
                                A = self.A_array[i, j]
                                mu = (self.energies[k] + self.energy_change[k]) - (self.energies[l] - self.energy_change[l])
                                kwargs = {'A': A, 'mu': mu, 'laser': intensity}
                                if self.shape.lower() == 'voigt':
                                    kwargs['fwhmG'] = self._params['FWHMG'].value
                                D[x, y] = bxrho(**kwargs)
        self.D = D

    def _edit_D_matrix(self):
        for i, left in enumerate(self.level_counts):
            for j, right in enumerate(self.level_counts):
                if i < j and not np.isclose(self.A_array[i, j], 0):
                    for k, (fe, mze) in enumerate(zip(self.Flist[i], self.MFlist[i])):
                        for l, (fg, mzg) in enumerate(zip(self.Flist[j], self.MFlist[j])):\
                            if self.mode == (mze - mzg):
                                x = self.level_counts_cumsum[i] - self.level_counts[i] + k
                                y = self.level_counts_cumsum[j] - self.level_counts[j] + l

                                intensity = self._params['Laser_intensity'].value
                                A = self.A_array_used[x, y]
                                mu = (self.energies[k] + self.energy_change[k]) - (self.energies[l] - self.energy_change[l])
                                self.D[x, y].mu = mu
                                self.D[x, y].A = A
                                self.D[x, y].laser = intensity
                                if self.shape.lower() == 'voigt':
                                    self.D[x, y].fwhmG = self._params['FWHMG'].value

    def _evaluate_matrices(self, f):
        D = np.zeros(self.D.shape)
        for i, left in enumerate(self.level_counts):
            for j, right in enumerate(self.level_counts):
                if i < j and not np.isclose(self.A_array[i, j], 0):
                    for k, (fe, mze) in enumerate(zip(self.Flist[i], self.MFlist[i])):
                        for l, (fg, mzg) in enumerate(zip(self.Flist[j], self.MFlist[j])):\
                            if self.mode == (mze - mzg):
                                D[x, y] = self.D[x, y](f)
        D = np.transpose(D) + D - np.eye(D.shape[0]) * D.sum(axis=1)
        self.M = self.A_array_used + D

    def _rhsint(self, y, t):
        """Define the system of ODE's for use in the odeint method from SciPy.
        Note that the input is (y, t)."""
        return np.dot(self.M, y)

    def __call__(self, x):
        try:
            response = np.array(len(x))
            for f in x:
                self._evaluate_matrices(f)
                dt = self._params['Interaction_time'].value / 400
                y = integrate.odeint(self._rhsint, self.P, np.arange(0, self._params['Interaction_time'].value, dt))
                pop1 = np.zeros(len(self.level_counts))
                pop2 = np.zeros(len(self.level_counts))
                for i, (n, ncs) in enumerate(zip(self.level_counts, self.level_counts_cumsum)):
                    pop1[i] = y[-2, ncs-n:ncs].sum()
                    pop2[i] = y[-1, ncs-n:ncs].sum()
                decays =

class Polar(object):

    r"""Class for calculating the polarization for optical pumping between
    several supplied levels and a given laser intensity and frequency.
    Assumes no decay below the ground state.

    Parameters
    ----------
    levels : list of :class:`Level` objects
        Ordered from highest to lowest energy.
    laser : float
        Laser intensity, float in W/m :sup:`2`. If supplied as a list of
        floats, that many lasers will be simulated.
    mode : {-1, 0, 1}
        Mode of the optical pumping; 1 = :math:`\sigma^+`,
        -1 = :math:`\sigma^-` , 0 = linear polarization. In the case
        of multiple lasers, a ist of modes has to be supplied.
    spin : integer or half-integer
        Nuclear spin.
    field : float
        Strength of the magnetic field in T.
    lifetimes : array_like
        Lifetimes of the excited states in s. NxN array, with N the length of
        the :attr:`levels` parameter. Ordering must match the supplied list of
        :class:`Level` objects.
    time : float
        Interaction time in seconds.

    Other parameters
    ----------------
    steps : integer, optional
        The number of timesteps, defaults to 400.
    relaxationtime: float, optional
        The length of the relaxation after the interaction with the laser.
        Models purely decay of the nuclei, without laser interaction. If set
        to 0 (default) this is not done.
    frac : float, optional
        Fraction of contamination of other laser modes. If set to 0 (default),
        no contamination of other laser modes is incorporated. Has to be
        between 0 and 1.
    integrator : string, optional
        Selects the integrator to use for solving the differential equations.
        Defaults to `odeint`, which is the LSODA solver. Possible other
        values are `vode`, `zvode`, `lsoda`, `dopri5` and `dop853`. See the
        SciPy documentation for an overview of these integrators.
    time_dependence : callable, optional
        If the frequency supplied is time-dependent, the time_dependence
        gives the Doppler shift at each time. If given, the D-matrix is
        calculated again for each timestep, and the used frequency is the given
        frequency multiplied with the result of this callable for the time.

    Returns
    -------
    Polar
        Callable object, returns the polarization in percent for a given
        frequency and population of the different levels in percent."""

    def __init__(self, levels, laser, mode, spin, field, lifetimes, time, steps=400, relaxationtime=0, frac=0, integrator='odeint', time_dependence=None):
        super(Polar, self).__init__()

        # Set all parameters for preparation of the A and D matrices
        self.levels = levels
        self.laser = laser
        try:
            self.n = len(self.laser)
        except TypeError:
            self.laser = [self.laser]
            self.n = 1
        try:
            self.mode = [m if m in [-1, 0, 1] else 1 for m in mode]
        except TypeError:
            self.mode = [mode if mode in [1, 0, -1] else 1]

        self.pos = []
        self.spin = spin
        self.field = field
        self.lifetime = lifetimes
        self.frac = frac
        # Prepare the A and D matrix.
        self._prepare()

        # Set all the parameters for the integration.
        self.steps = steps

        self.tof = time
        self.dt = self.tof / self.steps

        self.rTof = relaxationtime
        self.rdt = self.rTof / self.steps

        integrators = ['odeint', 'vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
        self.integrator = integrator if integrator in integrators else 'odeint'

        self.time_dependence = time_dependence if callable(time_dependence) else None
        self.frequency_shift = self.time_dependence is not None

    def _convF(self, I, J):
        """Convenience function for the calculation of F and F_z."""
        F = np.arange(np.abs(J - I), J + I + 1)  # Values of F

        Flen = (2 * F + 1).astype('int')  # Lengths of F_z
        starts = np.cumsum(np.append([0], Flen[:-1]))  # Index for different F states

        # Pre-allocate
        f = np.zeros(int((2 * F + 1).sum()))  # F-states
        mz = np.zeros(int((2 * F + 1).sum()))  # F_z-states

        # Fill the pre-allocated arrays
        for i, (entry, start) in enumerate(zip(Flen, starts)):
            mz[start:start + entry] = np.arange(-F[i], F[i] + 1)
            f[start:start + entry] = F[i]

        return f, mz

    def _energy(self, level, F, m_z):
        """Calculate the energy of a hyperfine level in a magnetic field.
        Returns the energy in MHz."""
        # Gather the quantum numbers
        I = self.spin

        L = level.L
        S = level.S
        J = level.J

        A = level.A
        B = level.B

        # Field information
        field = self.field

        # Hyperfine interaction part
        cA = F * (F + 1) - I * (I + 1) - J * (J + 1)
        if I > 0.5 and J > 0.5:
            cB = (3 * cA * (cA + 1) -
                  4 * I * (I + 1) * J * (J + 1)) / \
                (2 * I * (2 * I - 1) * J * (2 * J - 1))
        else:
            cB = 0.0

        # Interaction with magnetic field
        if not np.isclose(J, 0.0):
            gJ = GL * (J * (J + 1) + L * (L + 1) - S * (S + 1)) / \
                (2 * J * (J + 1)) + GS * (J * (J + 1) - L * (L + 1) + S *
                                          (S + 1)) / (2 * J * (J + 1))
        else:
            gJ = 0.0
        gF = np.zeros(len(F))
        for i, f in enumerate(F):
            if not np.isclose(f, 0):
                gF[i] = gJ * (f * (f + 1) + J * (J + 1) - I * (I + 1)) / \
                    (2 * f * (f + 1))
            else:
                gF[i] = 0
        val = A * cA / 2.0
        val += B * cB / 4.0
        val += gF * MUB * field * m_z * ((10 ** (-6)) / H)
        val += (level.energy * EV_TO_MHZ)  # Convert to MHz
        return val

    def _A(self, excited, ground, Fe, Fg, Mze, Mzg, lifetime):
        """Calculate the partial Einstein A coefficient.

        Parameters
        ----------
        excited: Level
            Level object for the ground state.
        ground: Level
            Level object for the excited state.
        Fe: integer or half-integer
            Selected F quantum number in the excited state.
        Fg: integer or half-integer
            Selected F quantum number in the ground state.
        Mze: integer or half-integer
            Selected projection of F in the excited state.
        Mzg: integer or half-integer
            Selected projection of F in the ground state.
        lifetime: float
            Mean lifetime of the excited state in seconds.

        Returns
        -------
        Aeg: float
            Partial Einstein A coefficient, in s:sup:`-1`."""

        I = self.spin
        Jex = excited.J
        Jgr = ground.J
        Fe = float(Fe)
        Fg = float(Fg)

        A = float((2 * Jex + 1) * (2 * Fe + 1) * (2 * Fg + 1))
        W3 = W3J(Fg, 1.0, Fe, -Mzg, Mzg - Mze, Mze)
        W6 = W6J(Jgr, Fg, I, Fe, Jex, 1.0)
        A = A * (W3 ** 2)
        A = A * (W6 ** 2)
        A = A / lifetime
        return A

    def _convertFMftoMIMJ(self, level, F, Mf):
        """Convert the (F, Mf) quantum numbers to (MI, MJ) quantum numbers
        using the projection method.

        Parameters
        ----------
        level: Level
            Level object for the relevant level.
        F, Mf: integer or half-integers
            Quantum numbers ot be converted.

        Returns
        -------
        (Mi, Mj): tuple
            If a single (F, Mf) couple is given, the converted quantum numbers
            are returned as a tuple of floats. Otherwise, they are returned as
            a tuple of arrays."""
        A = level.A
        I = self.spin
        J = level.J

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

        # If (F, Mf) contains multiple values, deduce the correct quantum
        # numbers for each value. If not, only calculate the single value.
        try:
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
            MI = np.array(MI)
            MJ = np.array(MJ)
        except:
            f_select = np.isclose(M, F)
            m_select = np.isclose(mf, Mf)

            MI = I[np.bitwise_and(f_select, m_select)][0]
            MJ = J[np.bitwise_and(f_select, m_select)][0]

        return MI, MJ

    def _prepare(self):
        """When creating the Polar-object, calculate as much (frequency
        independent) values as possible. Create BxRho-objects to facilitate
        (and speed up) the frequency-based calculations."""
        # Prepare the quantum numbers.
        J = np.array([lev.J for lev in self.levels])
        I = self.spin

        # Count the number of magnetic states in the multiplets.
        Nlev = ((2 * J + 1) * (2 * I + 1)).astype('int')

        # Total number of states.
        N = Nlev.sum()
        self.Nlev = Nlev

        # Pre-allocate the energy and population vectors.
        E = np.zeros(N)
        P = np.zeros(N)
        P[N - Nlev[-1]:] = 1.0 / Nlev[-1]

        # The transition matrices are NxN.
        A = np.zeros((N, N))
        # D has to contain BxRho-objects, so dtype='objects'.
        D = np.zeros((self.n, N, N), dtype='object')

        # Create the different F-state labels for the levels.
        # These are seperate lists for each level.
        F = []
        Mf = []
        for j in J:
            f, mf = self._convF(I, j)
            F.append(f)
            Mf.append(mf)
        F = np.array(F)
        Mf = np.array(Mf)
        self.F = F

        # F_z component of the different states.
        sM = np.array([])
        for sm in np.array(Mf.flatten()):
            sM = np.append(sM, sm)

        # Calculate I_z from sM.
        self.MI = np.array([])
        for lev, f, mf in zip(self.levels, F, Mf):
            mi = self._convertFMftoMIMJ(lev, f, mf)[0]
            self.MI = np.append(self.MI, mi)

        # Fill the energy vector.
        Nlevcs = Nlev.cumsum()
        for n, ncs, l, f, mf in zip(Nlev, Nlevcs, self.levels, F, Mf):
            E[ncs - n:ncs] = self._energy(l, f, mf)
        self.Nlevcs = Nlevcs

        # Fill the A and D arrays in the correct places.

        #####################################
        # ASSUMES ENERGY-ORDERED LEVELS!!!! #
        #####################################
        # Loop over couples of levels.
        for i, left in enumerate(self.levels):
            for j, right in enumerate(self.levels):
                # Second level has to be lower in energy, so i < j
                # since the levels are energy-ordered.
                # This if-condition could be substituted by
                # looping over the correct levels as the 'right' levels,
                # but the index number has to be adjusted to something
                # unintuitive. Since this check does not take a long time
                # to perform, this has been left in.
                # Also check if the lifetime is something finite.
                # If not, skip the calculations.
                if i < j and not self.lifetime[i, j] == np.inf:
                    # Loop over the magnetic substates in each level.
                    for k, (fe, mze) in enumerate(zip(F[i], Mf[i])):
                        for l, (fg, mzg) in enumerate(zip(F[j], Mf[j])):
                            # Select the correct indices for the matrices.
                            x = Nlevcs[i] - Nlev[i] + k
                            y = Nlevcs[j] - Nlev[j] + l
                            # Calculate the Einstein A-coefficient.
                            A[x, y] = self._A(left, right,
                                              fe, fg, mze, mzg,
                                              self.lifetime[i, j])
                            if not np.isclose(A[x, y], 0):
                                if [fg, fe, E[x]-E[y]] not in self.pos:
                                    self.pos.append([fg, fe, E[x]-E[y]])
                            # Fill the D-matrix for each laser.
                            # The lasers use the first dimension of the array!
                            for z in range(self.n):
                                frac = 1.0 if self.mode[z] == (
                                    mze - mzg) else self.frac
                                if frac == 0:
                                    pass
                                else:
                                    intensity = frac * self.laser[z]
                                    tau = 1.0 / self.lifetime[i, j]
                                    if mze - mzg == 1:  # sigma plus
                                        D[z, x, y] = BxRho(E[x], E[y],
                                                           A[x, y],
                                                           tau,
                                                           intensity)
                                    elif mze - mzg == -1:  # sigma minus
                                        D[z, x, y] = BxRho(E[x], E[y],
                                                           A[x, y],
                                                           tau,
                                                           intensity)
                                    # linear polarization
                                    elif mze - mzg == 0:
                                        D[z, x, y] = BxRho(E[x], E[y],
                                                           A[x, y],
                                                           tau,
                                                           intensity)
                                    else:
                                        pass

        # Copy the needed arrays to self.
        A = np.transpose(A) - np.eye(A.shape[0]) * A.sum(axis=1)
        self.A = A
        self.D = D
        self.P = P

    def changeInitialPopulation(self, P):
        """
        Change the initial population to the supplied one.

        Parameters
        ----------
        P : list of population values.
            This list can be either for each magnetic substate,
            or for each fine-structure level. The latter case
            assumes equal distribution among the magnetic substates
            of each fine-structure level.
            Can be supplied as integer (e.g. 10 particles, 40 particles, ...)
            or as floats (0.10, 0.40). The input is automatically normalized.
        """
        # In case the length of the supplied population is equal
        # to the total number of magnetic states,
        # just copy the array.
        if len(P) == len(self.P):
            P = np.array(P, dtype='float')
            P = P / np.sum(P)
            self.P = P
        # If the supplied population is equal in length to the
        # number of fine-structure levels,
        # distribute the population of each fine-structure level
        # equally among all the magnetic substates.
        elif len(P) == len(self.levels):
            # Count the number of magnetic substates.
            J = np.array([lev.J for lev in self.levels])
            I = self.spin
            Nlev = ((2 * J + 1) * (2 * I + 1)).astype('int')
            # Normalize the input.
            P = np.array(P, dtype='float')
            P = P / np.sum(P)
            # Go over each level, append to a temporary variable
            # a list of the population in each magnetic sublevel.
            dinkie = np.array([])
            for i, n in enumerate(Nlev):
                dinkie = np.append(dinkie, [P[i] / n] * n)
            self.P = dinkie
        # If someone supplied wrong dimensions, raise (hell).
        else:
            st = 'Dimension mismatch: %i is not %i or %i' % (
                len(P), len(self.P), len(self.levels))
            raise ValueError(st)

    def _initializeM(self, f):
        """Creates the matrix M associated with a given laser frequency f,
        in MHz."""
        # Calculate BxRho(f).
        if not len(f) == self.n:
            mess = 'Not enough frequencies! Want %i, got %i' % (self.n, len(f))
            raise IndexError(mess)
        D = util.callNDArray(self.D, f)
        D = D.sum(axis=0)

        # Prepare D.
        D = np.transpose(D) + D
        D -= np.eye(D.shape[0]) * D.sum(axis=1)

        # Create the M matrix.
        self.M = self.A + D

    def _rhs(self, t, y, f):
        """Define the system of ODE's for use in the ode-object from SciPy.
        Note that the input is (t, y)."""
        if self.frequency_shift:
            self._initializeM(f * self.time_dependence(t))
        return np.dot(self.M, y)

    def _rhsint(self, y, t, f):
        """Define the system of ODE's for use in the odeint method from SciPy.
        Note that the input is (y, t)."""
        if self.frequency_shift:
            self._initializeM(f * self.time_dependence(t))
        return np.dot(self.M, y)

    def _produce(self, F):
        """For a given frequency F, generate the matrices for the system
        of ODE's and use a given integration method to solve the system.
        Convert the result to a polarization, and process and organise the
        output.

        The integration method selected as a standard is the LSODA method.
        If this performs unexpectedly, see the comments in the source code."""
        # Initializes the matrix M.
        F = np.array(F)
        self._initializeM(F)

        ##############################
        # LSODA USER FRIENDLY METHOD #
        ##############################
        # Use the user-friendlier odeint to solve the differential
        # equation using the LSODA solver. For this, the function arguments
        # have to be (y, t), while for the ode command, the order is (t, y).
        # We are only interested in the result for the final timestep,
        # the last row of the result so [-1, :] is saved.
        if self.integrator == 'odeint':
            y = integrate.odeint(self._rhsint, self.P,
                                 np.arange(0, self.tof, self.dt),
                                 args=(F,))[-1, :]
            if not np.isclose(self.rTof, 0):
                dinkieD = self.D
                self.D = np.zeros(self.D.shape)
                self._initializeM(F)
                y = integrate.odeint(self._rhsint, y,
                                     np.arange(0, self.rTof, self.rdt),
                                     args=(F,))[-1, :]
                self.D = dinkieD

        #######################
        # MORE GENERAL METHOD #
        #######################
        # Different integrator methods are available, if the standard
        # method gives bad results, the Runge-Kutta methods are
        # recommended for testing.
        # The different integrators are:
        # 'vode': Real-valued Variable-coefficient Ordinary Differential
        #         Equation solver.
        # 'zvode': Complex-valued Variable-coefficient Ordinary Differential
        #          Equation solver.
        # 'lsoda': Livermore Solver for Ordinary Differential Equations.
        # 'dopri5': Explicit Runge-Kutta method of order (4)5.
        # 'dop853': Explicit Runge-Kutta method or order 8(5, 3)
        # For more detailed parameters available for each integrator, see
        # the SciPy documentation.
        else:
            r = integrate.ode(self._rhs).set_integrator(self.integrator).set_f_params(F)
            r.set_initial_value(self.P)
            while r.t < self.tof and r.successful():
                y = r.integrate(r.t + self.dt)
            if not np.isclose(self.rTof, 0):
                dinkieD = self.D
                self.D = np.zeros(self.D.shape)
                self._initializeM(F)
                r = integrate.ode(self._rhs).set_integrator(self.integrator).set_f_params(F)
                r.set_initial_value(y)
                while r.t < self.rTof and r.successful():
                    y = r.integrate(r.t + self.rdt)
                self.D = dinkieD

        ######################
        # PROCESS THE RESULT #
        ######################
        # Convert the population to a polarization percentage.
        pol = 100 * (np.dot(self.MI, y) / self.spin)
        # Convert to atomic population.
        pop = []
        for n, ncs in zip(self.Nlev, self.Nlevcs):
            pop.append(y[ncs - n:ncs].sum())
        pop = np.array(pop)
        y = 100.0 * (pop / pop.sum())
        # y = np.array(y)
        y = 100.0 * (y / y.sum())
        y = np.append(pol, y)
        return y

    def evolution(self, f):
        """Given a single frequency f or range of frequency, return the time
        evolution of the polarization and population.

        Parameters
        ----------
        f : float
            Laser frequency in MHz

        Returns
        -------
        resp : NumPy array
            Array containing the time, polarization and population of the
            fine structure levels in function of time.
            If f """

        # Initializes the matrix M.
        f = np.array([f])
        self._initializeM(f)

        ##############################
        # LSODA USER FRIENDLY METHOD #
        ##############################
        # Use the user-friendlier odeint to solve the differential
        # equation using the LSODA solver. For this, the function arguments
        # have to be (y, t), while for the ode command, the order is (t, y).
        # We are only interested in the result for the final timestep,
        # the last row of the result so [-1, :] is saved.
        if self.integrator == 'odeint':
            y = integrate.odeint(self._rhsint, self.P,
                                 np.arange(0, self.tof, self.dt),
                                 args=(f,))
            if not np.isclose(self.rTof, 0):
                dinkieD = self.D
                self.D = np.zeros(self.D.shape)
                self._initializeM(np.array([f]))
                yRel = integrate.odeint(self._rhsint, y,
                                        np.arange(0, self.rTof, self.rdt),
                                        args=(f,))
                self.D = dinkieD
                y = np.vstack((y, yRel))

        #######################
        # MORE GENERAL METHOD #
        #######################
        # Different integrator methods are available, if the standard
        # method gives bad results, the Runge-Kutta methods are
        # recommended for testing.
        # The different integrators are:
        # 'vode': Real-valued Variable-coefficient Ordinary Differential
        #         Equation solver.
        # 'zvode': Complex-valued Variable-coefficient Ordinary Differential
        #          Equation solver.
        # 'lsoda': Livermore Solver for Ordinary Differential Equations.
        # 'dopri5': Explicit Runge-Kutta method of order (4)5.
        # 'dop853': Explicit Runge-Kutta method or order 8(5, 3)
        # For more detailed parameters available for each integrator, see
        # the SciPy documentation.
        else:
            r = integrate.ode(self._rhs).set_integrator(self.integrator).set_f_params(f)
            r.set_initial_value(self.P)
            shape = (1 + np.bitwise_not(np.isclose(self.rTof, 0)) * self.steps,
                     len(self.P))
            y = np.zeros(shape)
            y[0, :] = self.P
            i = 1
            while r.t < self.tof and r.successful():
                y[i, :] = r.integrate(r.t + self.dt)
                i += 1
            if not np.isclose(self.rTof, 0):
                dinkieD = self.D
                self.D = np.zeros(self.D.shape)
                self._initializeM(f)
                r = integrate.ode(self._rhs).set_integrator(self.integrator).set_f_params(f)
                r.set_initial_value(y)
                while r.t < self.rTof and r.successful():
                    y[i, :] = r.integrate(r.t + self.rdt)
                    i += 1
                self.D = dinkieD

        ######################
        # PROCESS THE RESULT #
        ######################
        # Convert the population to a polarization percentage.
        def pol(a):
            return 100 * (np.dot(self.MI, a) / self.spin)

        pol = np.apply_along_axis(pol, 1, y).reshape((-1, 1))
        # Convert to atomic population.
        pop = np.zeros((self.steps, self.Nlev.sum()))
        for i, (n, ncs) in enumerate(zip(self.Nlev, self.Nlevcs)):
            pop[:, i] = y[:, ncs - n:ncs].sum(axis=1)
        y = 100.0 * (pop / pop.sum())
        y = np.hstack((pol, y))
        time = np.linspace(0, self.tof + self.rTof, y.shape[0])
        return time, y

    def __call__(self, *f):
        """Given frequency is supplied in MHz.
        A list of frequencies has to be supplied for each laser,
        so in the case of one laser,::

            p = Polar(...)
            p(10)

        is not valid, but::

            p = Polar(...)
            p([10])

        is.
        For each *scan* of a laser frequency, i.e. an entry in the list which
        is array like, the output gains an extra dimension, with the first N
        dimensions detailing the frequencies. The first entry in the last
        dimension is the polarization achieved, while the
        subsequent entries contain the population in the level ordered by
        decreasing energy (highest first, lowest last).

        Parameters
        ----------
        f : list of (list of) floats
            The entries of the frequencies for which the laser(s) are set.

        Returns
        -------
        resp : NumPy array
            Array containing the polarization and population of the fine
            structure levels."""
        assert self.n == len(f)
        try:
            # If a range of f-values is given, the try block will call
            # self(F) for each F in f, and save all results, then return them
            # print(len(f[0]) > 0)
            assert len(f[0]) > 0
            if self.n >= 2:
                if isinstance(f[1], np.ndarray):
                    end = 2
                else:
                    end = 1
                t = f[end:]
                f = np.meshgrid(*f[0:end])
                for x in t:
                    a = np.zeros(f[0].shape)
                    a.fill(x)
                    f.append(a)
            else:
                f = np.meshgrid(*f)
            resp = np.zeros(np.shape(f[0]) + (len(self.F) + 1,))
            indices = util.state_number_enumerate(np.shape(f[0]))
            for inds in indices:
                resp[inds] = self._produce([F[inds] for F in f])
        except:
            # If only a single value is given, the except block will be
            # executed, solving the differential equation for the given
            # frequency.
            resp = self._produce(list(f))
        return resp
