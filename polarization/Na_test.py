from satlasaddon import RateModel, RateModelPolar
import satlas as sat
sat.set(['standard', 'online'])
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as csts

EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6
A_array = np.array([[0, (16.249e-9)**-1], [0, 0]])
energies = [2.104428981, 0]
J = [1.5, 0.5]
L = [1, 0]

# 28Na
ABC = [[250*0.0211, 0.7, 0], [250, 0, 0]]
I = 1
laser_intensity = 50
scale = 1/50000
asymmetry = -0.75
laser_mode = 1
centroids = [5852]
field = 10e-4
background = 0

model_decay = RateModel(I, J, L, ABC, centroids, energies, A_array, laser_intensity=laser_intensity, scale=scale, laser_mode=laser_mode, interaction_time=1e-6, background=background, shape='voigt', field=field)
model_polar = RateModelPolar(I, J, L, ABC, centroids, energies, A_array, laser_intensity=laser_intensity, scale=asymmetry, laser_mode=laser_mode, interaction_time=10e-6, background=background, shape='voigt', field=field)

frequencies = np.arange(model_decay.locations.min()-500, model_decay.locations.max()+500, 1)
f_trans = energies[0] * EV_TO_MHZ
fig, ax = plt.subplots(2, 1, sharex=True)
ax[1].set_xlabel('Relative frequency [MHz]')
ax[0].set_ylabel('Decay rate [s$^{-1}$]')
ax[1].set_ylabel('Asymmetry')
ax[0].plot(frequencies - f_trans, model_decay(frequencies))
ax[1].plot(frequencies - f_trans, model_polar(frequencies))
plt.show()

from contextlib import
