from matplotlib import cm
import numpy as np
from polarization.polar import Polar
from polarization.utilities import Level, Energy
import pandas as pd
from satlasaddon import RateModelPolar, RateModelDecay
EV_TO_MHZ = 2.417989348 * 10 ** 8

lev1 = Level(13.09487199, (125.7, 81.6), 2.5, 0.5, 2.0)
lev2 = Level(13.07571514, (125.7, 81.6), 2.5, 0.5, 3.0)
lev3 = Level(12.90701474, (125.7, 81.6), 0.5, 0.5, 1.0)
lev4 = Level(11.82807064, (266.0, 84.8), 0.5, 0.5, 1.0)
lev5 = Level(11.72315988, (266.0, 0.0),  0.5, 0.5, 0.0)
lev6 = Level(11.62359221, (266.0, 84.8), 1.5, 0.5, 1.0)
lev7 = Level(11.54835392, (266.0, 84.8), 1.5, 0.5, 2.0)
lev8 = Level(0.0, (0.0, 0.0), 0.5, 0.5, 0.0)

levsFull = [lev1, lev2, lev3, lev4, lev5, lev6, lev7, lev8]
lifetimesFull = np.array([[np.inf, np.inf, np.inf, 6.80272108844e-07, np.inf,            4.6511627907e-08,  1.0775862069e-07,  np.inf],
                          [np.inf, np.inf, np.inf, np.inf,            np.inf,            np.inf,            3.02114803625e-08, np.inf],
                          [np.inf, np.inf, np.inf, 5.26315789474e-06, 1.02040816327e-06, 1.84162062615e-07, 5.29100529101e-08, np.inf],
                          [np.inf, np.inf, np.inf, np.inf,            np.inf,            np.inf,            np.inf,            1.96078431373e-09],
                          [np.inf, np.inf, np.inf, np.inf,            np.inf,            np.inf,            np.inf,            np.inf],
                          [np.inf, np.inf, np.inf, np.inf,            np.inf,            np.inf,            np.inf,            8.40336134454e-09],
                          [np.inf, np.inf, np.inf, np.inf,            np.inf,            np.inf,            np.inf,            np.inf]])
populationFull = [0.0, 0.0, 0.0, 0.25, 0.0833333333333, 0.25, 0.416666666667, 0.0]

levsSimple = [lev2, lev7]
lifetimesSimple = np.array([[np.inf, 3.02114803625e-08],
                            [np.inf, np.inf]])
populationSimple = [0.0, 1.0]


levs = levsSimple
lifetimesSelected = lifetimesSimple
populationSelected = populationSimple

efficiency = 1.0 / 50000
laser_intensity = 125*2  # W/m^2
field = 10 * 10 ** (-4)  # T
laser_mode = 1
I = 3.0 / 2
L_simple = [2.5, 1.5]
L_full = [2.5, 2.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5]
J_simple = [3.0, 2.0]
J_full = [2.0, 3.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0]
ABC_simple = [[125.7, 81.6, 0], [266.0, 84.8, 0]]
ABC_full = [[125.7, 81.6, 0.0], [125.7, 81.6, 0.0], [125.7, 81.6, 0.0], [266.0, 84.8, 0.0], [266.0, 0.0, 0.0], [266.0, 84.8, 0.0], [266.0, 84.8, 0.0], [0.0, 0.0, 0.0]]

level_energies_simple = [13.07571514, 11.54835392]
level_energies_full = [13.09487199, 13.07571514, 12.90701474, 11.82807064, 11.72315988, 11.62359221, 11.54835392, 0]

centroids_simple = [0]
centroids_full = [0]*(len(level_energies_full)-1)
A_array_simple = np.array([[0, 3.3e7], [0, 0]])
A_array_full = lifetimesFull**-1

J = J_simple
L = L_simple
ABC = ABC_simple
level_energies = level_energies_simple
centroids = centroids_simple
A_array = A_array_simple

f_st = np.abs(np.diff(level_energies_simple)) * EV_TO_MHZ

f1 = -246.285305858
f2 = 132.06469357
f3 = 79.6246939898

df1 = f2 - f1
df2 = f3 - f1

mass = 34.9752576
AMU_TO_KG = 1.66053904 * 10**(-27)
mass = mass * AMU_TO_KG
EV_TO_J = 1.60217662 * 10**(-19)

e = 30 * 10**3
l_pumping = 1.5
l_optical_detection = 0.2
energy = e * EV_TO_J
velocity = np.sqrt(2 * energy / mass)
tof_pump = l_pumping / velocity
tof_optical = l_optical_detection / velocity

step = 1
freqs = np.arange(-1400, 500 + step, step) + f_st

args = (I, J, L, ABC, centroids, level_energies, A_array)
kwargs3 = {'laser_intensity': [laser_intensity]*3,
'scale': 100,
'laser_mode': [laser_mode]*3,
'shape': 'voigt',
'fwhmG': 50,
'interaction_time': tof_pump,
'fixed_frequencies': [df1, df2],
'frequency_mode': 'offset',
'field': field}
kwargs = {'laser_intensity': [laser_intensity],
'scale': 100,
'laser_mode': [laser_mode],
'shape': 'voigt',
'fwhmG': 50,
'interaction_time': tof_pump,
'frequency_mode': 'offset',
'field': field}
model_pump = RateModelPolar(*args, **kwargs3)
try:
    model_pump._set_population(level=[-5, -4, -3, -2])
except:
    pass
kwargs3['interaction_time'] = tof_optical
kwargs3['laser_intensity'] = [5]*3
kwargs3['scale'] = 1 * efficiency
model_optical = RateModelDecay(*args, **kwargs3)
try:
    model_optical._set_population(level=[-5, -4, -3, -2])
except:
    pass

resp_pump3 = model_pump(freqs)
resp_optical3 = model_optical(freqs)

model_pump = RateModelPolar(*args, **kwargs)
try:
    model_pump._set_population(level=[-5, -4, -3, -2])
except:
    pass
kwargs['interaction_time'] = tof_optical
kwargs['laser_intensity'] = [5]
kwargs['scale'] = 1 * efficiency
model_optical = RateModelDecay(*args, **kwargs)
try:
    model_optical._set_population(level=[-5, -4, -3, -2])
except:
    pass

resp_pump = model_pump(freqs)
resp_optical = model_optical(freqs)

import matplotlib.pyplot as plt
import satlas as sat
sat.set(['standard', 'online'])
fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].plot(freqs - f_st, resp_pump)
ax[1, 0].plot(freqs - f_st, resp_optical)
ax[0, 1].plot(freqs - f_st, resp_pump3)
ax[1, 1].plot(freqs - f_st, resp_optical3)

ax[0, 0].set_ylabel('Polarisation [%]')
ax[1, 0].set_ylabel('Decay rate/atom [Hz]')
import scipy.constants as csts
C = csts.physical_constants['speed of light in vacuum'][0]
f_st = f_st[0] * 1e6
ax[1, 0].set_xlabel('Frequency offset from {:.5f} nm [MHz]'.format(C/f_st * 1e9))
ax[0, 0].set_title('Single laser')
ax[0, 1].set_title('Single laser + 2 AOMs at {:.2f} and {:.2f} MHz'.format(df1, df2))
plt.show()
