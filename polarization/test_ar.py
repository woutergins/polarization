import matplotlib.pyplot as plt
import numpy as np
import satlas as sat
from satlasaddon import RateModelPolar, RateModelDecay
import scipy.constants as csts

sat.set(['standard'])
C = csts.physical_constants['speed of light in vacuum'][0]
EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6
AMU_TO_KG = csts.physical_constants['atomic mass unit-kilogram relationship'][0]
EV_TO_J = csts.physical_constants['electron volt-joule relationship'][0]

beam_energy = 30e3
beam_energy *= EV_TO_J

efficiency = 1.0 / 50000000
area = 0.8 #cm^2
laser_power_pump = 100 #mW
laser_power_pump_AOM_1 = 50 #mW
laser_power_pump_AOM_2 =  0*10 #mW
laser_power_optical = 1 #mW
laser_intensity_pump = laser_power_pump / area * 10  # W/m^2
laser_intensity_pump_AOM_1 = laser_power_pump_AOM_1 / area * 10  # W/m^2
laser_intensity_pump_AOM_2 = laser_power_pump_AOM_2 / area * 10  # W/m^2
laser_intensity_optical = laser_power_optical / area * 10  # W/m^2
laser_mode = 1

field = 0 * 10 ** (-4)  # T

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

A_array_simple = np.array([[0, 3.31e7],
                           [0, 0]])
A_array_full = np.array([[0, 0, 0, 1.47e+06,   0,         2.15e+07,  9.28e+06,  0],
                         [0, 0, 0, 0,          0,         0,         3.31e+07,  0],
                         [0, 0, 0, 1.90e+05,   9.80e+05,  5.43e+06,  1.89e+07,  0],
                         [0, 0, 0, 0,          0,         0,         0,         5.10e+08],
                         [0, 0, 0, 0,          0,         0,         0,         0],
                         [0, 0, 0, 0,          0,         0,         0,         1.19e+08],
                         [0, 0, 0, 0,          0,         0,         0,         0]])

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

df1 = -(f2 - f1)
df2 = f3 - f1

mass = 34.9752576
mass *= AMU_TO_KG
velocity = np.sqrt(2 * beam_energy / mass)

detection_length = 100e-3
detection_tof = detection_length / velocity

interaction_length_optical = 20e-2
interaction_tof_optical = interaction_length_optical / velocity

interaction_length_polar = 1.5
interaction_tof_polar = interaction_length_polar / velocity

geometrical_efficiency = 14.941971804041613 / 100
quantum_efficiency = 0.05
scale = geometrical_efficiency * quantum_efficiency

step = 1
freqs = np.arange(-1400, 500 + step, step) + f_st
x_plot = freqs - f_st

args = (I, J, L, ABC, centroids, level_energies, A_array)
kwargs = {'laser_intensity': [laser_intensity_pump],
          'scale': 100,
          'laser_mode': [laser_mode],
          'shape': 'lorentzian',
          'fwhmG': 0,
          'interaction_time': interaction_tof_polar,
          'frequency_mode': 'offset',
          'field': field}
kwargs3 = {'laser_intensity': [laser_intensity_pump, laser_intensity_pump_AOM_1, laser_intensity_pump_AOM_2],
           'scale': 100,
           'laser_mode': [laser_mode]*3,
           'shape': 'lorentzian',
           'fwhmG': 0,
           'interaction_time': interaction_tof_polar,
           'fixed_frequencies': [df1, df2],
           'frequency_mode': 'offset',
           'field': field}

# First, calculate the polarisation and decay with 1 laser + 2 AOMs
model_pump = RateModelPolar(*args, **kwargs3)
try:
    model_pump._set_population(level=[-5, -4, -3, -2])
except:
    pass
kwargs3['interaction_time'] = interaction_tof_optical
kwargs3['laser_intensity'] = [laser_intensity_optical]*3
kwargs3['scale'] = 1 * scale
model_optical = RateModelDecay(*args, **kwargs3)
try:
    model_optical._set_population(level=[-5, -4, -3, -2])
except:
    pass

resp_pump3 = model_pump(freqs)
resp_optical3 = model_optical.integrate_with_time(freqs, interaction_tof_optical, detection_tof)

# Next, calculate with just 1 laser
model_pump = RateModelPolar(*args, **kwargs)
try:
    model_pump._set_population(level=[-5, -4, -3, -2])
except:
    pass
kwargs['interaction_time'] = interaction_tof_optical
kwargs['laser_intensity'] = [laser_intensity_optical]
kwargs['scale'] = 1 * scale
model_optical = RateModelDecay(*args, **kwargs)
try:
    model_optical._set_population(level=[-5, -4, -3, -2])
except:
    pass

resp_pump = model_pump(freqs)
resp_optical = model_optical.integrate_with_time(freqs, interaction_tof_optical, detection_tof)

fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].plot(x_plot, resp_pump)
ax[1, 0].plot(x_plot, resp_optical)
ax[0, 1].plot(x_plot, resp_pump3)
ax[1, 1].plot(x_plot, resp_optical3)

ax[0, 0].set_ylabel('Polarisation [%]')
ax[1, 0].set_ylabel('Photons/atom [#]')
f_st = f_st[0] * 1e6
ax[1, 0].set_xlabel('Frequency offset from {:.5f} nm [MHz]'.format(C/f_st * 1e9))
ax[0, 0].set_title('Single laser')
ax[0, 1].set_title('Single laser + 2 AOMs at {:.2f} and {:.2f} MHz'.format(df1, df2))

# fig.savefig('Ar_simulation_full_{}.pdf'.format(kwargs['fwhmL']), bbox_inches='tight')

plt.show()
