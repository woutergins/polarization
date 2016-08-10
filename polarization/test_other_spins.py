from satlasaddon import RateModel, RateModelPolar
import satlas as sat
sat.set(['standard'])
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as csts

EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6

ABC = [[-520, 0, 0], [-3070, 0, 0]]
A_array = np.array([[0, (3.854e-9)**-1], [0, 0]])
energies = [4.422440762145948, 0]
I = 0.5
J = [0.5, 0.5]
L = [0, 1]
laser_intensity = 50
scale = -1

laser_mode = 1
centroids = [5852]
background = 0

model1 = RateModelPolar(I, J, L, ABC, centroids, energies, A_array, laser_intensity=laser_intensity, scale=scale, laser_mode=laser_mode, interaction_time=1.0363e-5, fwhmL=1, background=background, shape='voigt', field=.001)
variation = {'Laser_intensity': False, 'Interaction_time': False, 'Transition_strength_0_to_1': False, 'FWHMG': False, 'Asymmetry_parameter': True}
model1.set_variation(variation)
frequencies = np.arange(model1.locations.min()-500, model1.locations.max()+500, 5)
f_trans = np.mean(frequencies)
print(model1.locations-f_trans)
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('Relative frequency [MHz]')
ax.set_ylabel('Asymmetry')
ax.set_title(r'$^{31}$Mg')
ax.plot(frequencies, model1(frequencies))
frequencies = model1.locations
# ax.plot(frequencies, model1(frequencies))
# ax.plot(model1.locations, np.zeros(3), 'o')
plt.show()
