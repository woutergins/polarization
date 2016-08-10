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
scale = -0.15

laser_mode = -1
centroids = [5852]
# centroids = [0]

f_laser = 1071455771.3
f_trans = 1069341427.47
iscool = 50019.34633
A=0.075760925
B=0.99935566
C=-1.7894284e-009
iscool = A+B*iscool+C*iscool**2

fluke = -6.086265000000001
A=999.97869
B=0.0031695015
if fluke>=0:
    fluke = A*fluke+B*fluke*fluke
else:
    fluke = A*fluke-B*fluke*fluke

data = np.loadtxt('minus.dat')
line_voltage = data[:, 0]
total_voltage = iscool - fluke - 50.428133 * line_voltage
mass = 30.996546

import collaps
doppler = collaps.dopplerfactor(mass, total_voltage)
frequencies = f_laser * doppler
asymmetry = data[:, 1]

background = -.02

model1 = RateModelPolar(I, J, L, ABC, centroids, energies, A_array, laser_intensity=laser_intensity, scale=scale, laser_mode=laser_mode, interaction_time=1e-5, background=background, shape='lorentzian', field=0)
# model2 = RateModelPolar(I, J, L, ABC, centroids, energies, A_array, laser_intensity=laser_intensity, scale=scale, laser_mode=laser_mode, interaction_time=1e-5, background=background, shape='voigt', field=0)
variation = {'Laser_intensity': False, 'Interaction_time': False, 'Transition_strength_0_to_1': False, 'FWHMG': True, 'Asymmetry_parameter': True, 'Field': False}
model1.set_variation(variation)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel('Relative frequency [MHz]')
ax.set_ylabel('Asymmetry')
ax.set_title(r'$^{31}$Mg')
ax.errorbar(frequencies - f_trans, asymmetry, yerr=data[:, 2], fmt='o')
sat.chisquare_fit(model1, frequencies, asymmetry, yerr=data[:, 2])
# sat.chisquare_fit(model2, frequencies, asymmetry, yerr=data[:, 2])
from contextlib import redirect_stdout
with open('results_for_agi.txt', 'w') as f:
    with redirect_stdout(f):
        model1.display_chisquare_fit()
model1.save('test_model.spectrum')
# model2.display_chisquare_fit()
# frequencies = np.array([0])
ax.plot(frequencies - f_trans, model1(frequencies))
# fig.savefig('Test_31Mg_voigt.pdf', bbox_inches='tight')

# figc, axc, cbar = sat.generate_correlation_map(model1, frequencies, asymmetry, method='chisquare', fit_kws={'yerr': data[:, 2]}, filter=['A_level', 'Centroid', 'Scale'])
# figc.savefig('Test_31Mg_voigt_correlation.pdf', bbox_inches='tight')
plt.show()
