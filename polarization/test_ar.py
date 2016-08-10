from matplotlib import cm
import numpy as np
from polarization.polar import Polar
from polarization.utilities import Level, Energy
import pandas as pd
from satlasaddon import RateModelPolar
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

laser_intensity = 125  # W/m^2
field = 0 * 10 ** (-3)  # T
laser_mode = 1
I = 3.0 / 2
L_simple = [2.5, 1.5, 0.5]
L_full = [2.5, 2.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5]
J_simple = [3.0, 2.0, 0.0]
J_full = [2.0, 3.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0]
ABC_simple = [[125.7, 81.6, 0], [266.0, 84.8, 0], [100, 0, 0]]
ABC_full = [[125.7, 81.6, 0.0], [125.7, 81.6, 0.0], [125.7, 81.6, 0.0], [266.0, 84.8, 0.0], [266.0, 0.0, 0.0], [266.0, 84.8, 0.0], [266.0, 84.8, 0.0], [0.0, 0.0, 0.0]]

level_energies_simple = [13.07571514, 11.54835392, 0]
level_energies_full = [13.09487199, 13.07571514, 12.90701474, 11.82807064, 11.72315988, 11.62359221, 11.54835392, 0]

centroids_simple = [0, 0, 0, 0]
centroids_full = [0]*(len(level_energies_full)-1)
A_array_simple = np.array([[0, 3.3e7, 0], [0, 0, 0], [0, 0, 0]])
A_array_full = lifetimesFull**-1

J = J_simple
L = L_simple
ABC = ABC_simple
level_energies = level_energies_simple
centroids = centroids_simple
A_array = A_array_simple

diff = Energy(1.52736122, unit='eV')
f_st = diff('MHz')

f1 = -247.682945669
f2 = 130.667053759
f3 = 78.2270541787
f1 = 130.667053759
f2 = 331.277053893

mass = 34.9752576
AMU_TO_KG = 1.66053904 * 10**(-27)
mass = mass * AMU_TO_KG
EV_TO_J = 1.60217662 * 10**(-19)
energies = [30*(10**3), 40*(10**3), 50*(10**3)]
energies = [50*(10**3)]
lengths = [1.5]#, 1.6, 1.7, 1.8, 1.9]
lasers = ['1 laser', '1 laser + 1 AOM', '1 laser + 2 AOMS']
# intensities = np.arange(10, 110, 10)
intensities = [125]
freqLen = 1901
f = np.arange(-1400, 501, 1)
columns = pd.MultiIndex.from_product([lasers, intensities, energies, lengths], names=['Lasers', 'Laser intensity', 'Energy', 'Interaction length'])
data = pd.DataFrame(index=f, columns=columns)

e = energies[0]
l = lengths[0]
energy = e * EV_TO_J
velocity = np.sqrt(2 * energy / mass)
tof = l / velocity
model = RateModelPolar(I, J, L, ABC, centroids, level_energies, A_array, laser_intensity=[laser_intensity], scale=100, laser_mode=[laser_mode], shape='lorentzian', interaction_time=tof)#, fixed_frequencies=[f1 + f_st, f2+f_st])
model._set_population(level=[-3, -2])
locs = model.locations - f_st
# freqs = np.arange(locs.min() - 100, locs.max() + 100, 5)
freqs = np.arange(-1400, 505, 5) + f_st
print(model.P)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
resp = model(freqs)
ax.plot(freqs - f_st, resp)
# print(resp)
plt.show()

# for e in energies:
#     for l in lengths:
#         for intensity in intensities:
#             print(e, l, intensity)

#             testN = Polar(levs, intensity,
#                           laser_mode, spin, field,
#                           lifetimesSelected, tof)
#             testN.changeInitialPopulation(populationSelected)
#             las = f + f_st
#             # for fg, fe, pos in testN.pos:
#             #     print(fg, fe, pos-f_st)
#             # raise ValueError
#             y = testN(las)
#             data[lasers[0], intensity, e, l] = y[:, 0]

#             testN = Polar(levs, [intensity, intensity],
#                           [laser_mode, laser_mode], spin, field,
#                           lifetimesSelected, tof)
#             testN.changeInitialPopulation(populationSelected)
#             las = (f + f_st, f1 + f_st)
#             y = testN(*las)
#             data[lasers[1], intensity, e, l] = y[:, 0]

#             testN = Polar(levs, [intensity, intensity, intensity],
#                           [laser_mode, laser_mode, laser_mode], spin, field,
#                           lifetimesSelected, tof)
#             testN.changeInitialPopulation(populationSelected)
#             las = (f + f_st, f1 + f_st, f2 + f_st)
#             y = testN(*las)
#             data[lasers[2], intensity, e, l] = y[:, 0]

# data = data.T
data.to_pickle('Simul_multi_positive_10mW.data')
