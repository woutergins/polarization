from satlasaddon import RateModelDecay, RateModelPolar
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
laser_intensity = [50]
scale = -1
import time
laser_mode = [-1]
centroids = [5852]

background = 0

model = RateModelPolar(I, J, L, ABC, centroids, energies, A_array, laser_intensity=laser_intensity, scale=scale, laser_mode=laser_mode, interaction_time=5e-6, background=background, shape='voigt', field=6*10**-4)

frequencies = np.arange(model.locations.min()-500, model.locations.max()+500, 5)
f_trans = energies[0] * EV_TO_MHZ
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('Relative frequency [MHz]')
ax.set_ylabel('Decay rate')
ax.set_title(r'$^{31}$Mg')
start = time.time()
response = model(frequencies)
print((time.time()-start)/len(frequencies))
ax.plot(frequencies - f_trans, model(frequencies))
plt.show()
