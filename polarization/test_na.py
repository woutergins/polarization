import matplotlib.pyplot as plt
import numpy as np
import satlas as sat
from satlasaddon import RateModelPolar, RateModelDecay
import scipy.constants as csts

sat.set(['standard'])#, 'online'])
C = csts.physical_constants['speed of light in vacuum'][0]
EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6
AMU_TO_KG = csts.physical_constants['atomic mass unit-kilogram relationship'][0]
EV_TO_J = csts.physical_constants['electron volt-joule relationship'][0]

efficiency = 1.0 / 50000
area = 0.8 #cm^2
laser_power_pump = 10 #mW
laser_power_optical = 100 #mW
laser_intensity_pump = laser_power_pump / area * 10  # W/m^2
laser_intensity_optical = laser_power_optical / area * 10  # W/m^2
laser_mode = 1

field = 6 * 10 ** (-4)  # T

Na23_I = 1.5
Na23_mu = 2.2176556
Na23_Q = 1.1045
Na26_I = 3.0
Na26_mu = 2.851
Na26_Q = -0.0053
B_factor = Na26_Q / Na23_Q
A_factor = (Na26_mu / Na26_I) / (Na23_mu / Na23_I)

I = 3.0

L_D2 = [2.0, 2.0]

J_D2 = [1.5, 0.5]

ABC_D2 = [[18.534 * A_factor, 2.724 * B_factor, 0], [885.8130644 * A_factor, 0, 0]]

level_energies_D2 = [2.102297159, 0]

centroids_D2 = [0]

A_array_D2 = np.array([[0, 6.1542e6],
                       [0, 0]])

J = J_D2
L = L_D2
ABC = ABC_D2
level_energies = level_energies_D2
centroids = centroids_D2
A_array = A_array_D2

f_st = np.abs(np.diff(level_energies_D2)) * EV_TO_MHZ

mass = 25.992633
mass = mass * AMU_TO_KG

e = 50 * 10**3
l_pumping = 1.9
l_optical_detection = 0.2
energy = e * EV_TO_J
velocity = np.sqrt(2 * energy / mass)
tof_pump = l_pumping / velocity
tof_optical = l_optical_detection / velocity

args = (I, J, L, ABC, centroids, level_energies, A_array)
kwargs = {'laser_intensity': [laser_intensity_pump],
          'scale': 100,
          'laser_mode': [laser_mode],
          'shape': 'lorentzian',
          'fwhmL': None,
          'interaction_time': tof_pump,
          'field': field}

# Next, calculate with just 1 laser
model_pump = RateModelPolar(*args, **kwargs)
kwargs['interaction_time'] = tof_optical
kwargs['laser_intensity'] = [laser_intensity_optical]
kwargs['scale'] = 1 * efficiency
model_optical = RateModelDecay(*args, **kwargs)

step = 0.5
extra = 200
freqs = np.arange(model_pump.locations.min() - extra, model_pump.locations.max() + extra + step, step)
x_plot = freqs - f_st

resp_pump = model_pump(freqs)
resp_optical = model_optical(freqs)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(x_plot, resp_pump)
ax[1].plot(x_plot, resp_optical)

ax[0].set_ylabel('Polarisation [%]')
ax[1].set_ylabel('Decay rate/atom [Hz]')
f_st = f_st[0] * 1e6
ax[1].set_xlabel('Frequency offset from {:.5f} nm [MHz]'.format(C/f_st * 1e9))

# fig.savefig('Na_simulation.pdf', bbox_inches='tight')

plt.show()
