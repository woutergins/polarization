from satlasaddon import RateModelDecay, RateModelPolar, convolve_with_gaussian
import satlas as sat
sat.set(['standard'])
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as csts
import scipy.integrate as integrate

AMU_TO_KG = 1.66053904 * 10**(-27)
EV_TO_J = 1.60217662 * 10**(-19)

beam_energy = 30e3
beam_energy *= EV_TO_J

EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6
A_array = np.array([[0, (16.299e-9)**-1], [0, 0]])
energies = [2.102293941, 0]
J = [0.5, 0.5]
L = [1, 0]

geometrical_efficiency = 14.941971804041613 / 100
quantum_efficiency = 0.175

# 23Na
mu_23 = 2.2176556
q_23 = 0.1045
I_23 = 3/2
ABC = [[94.44, 0, 0], [885.813, 0, 0]]
mass_23 = 22.9897692809

# 26Na
mu_26 = 2.851
I_26 = 3
A_factor = I_23 / mu_23 * mu_26 / I_26
ABC_26 = [[94.44 * A_factor, 0, 0], [885.813* A_factor, 0, 0]]
mass_26 = 25.992633
asymmetry_26 = -0.93
title_26 = '26Na'

# 21Na
mu_21 = 2.38630
I_21 = 3/2
A_factor = I_23 / mu_23 * mu_21 / I_21
ABC_21 = [[94.44 * A_factor, 0, 0], [885.813* A_factor, 0, 0]]
mass_21 = 20.9976552
asymmetry_21 = -0.75
title_21 = '21Na'

# 28Na
mu_28 = 2.426
I_28 = 1
A_factor = I_23 / mu_23 * mu_28 / I_28
ABC_28 = [[94.44 * A_factor, 0, 0], [885.813* A_factor, 0, 0]]
mass_28 = 27.998938
asymmetry_28 = -0.75
title_28 = '28Na'

i = [I_21, I_26, I_28]
abc = [ABC_21, ABC_26, ABC_28]
asym = [asymmetry_21, asymmetry_26, asymmetry_28]
m = [mass_21, mass_26, mass_28]
tit = [title_21, title_26, title_28]

area = 0.8 #cm^2
laser_power = 15 #mW
laser_intensity = laser_power / area * 10  # W/m^2
laser_intensity = [laser_intensity]

for I, ABC, mass, asymmetry, title in zip(i, abc, m, asym, tit):
    mass *= AMU_TO_KG
    velocity = np.sqrt(2 * beam_energy / mass)

    detection_length = 100e-3
    detection_tof = detection_length / velocity

    interaction_length_optical = 20e-2
    interaction_tof_optical = interaction_length_optical / velocity

    interaction_length_polar = 1.5
    interaction_tof_polar = interaction_length_polar / velocity

    scale = geometrical_efficiency * quantum_efficiency
    # laser_mode = [1]
    centroids = [0]
    field = 0 * 10e-4
    background = 0
    args = (I, J, L, ABC, centroids, energies, A_array)
    for laser_mode, title_add in zip([[1], [0], [-1]], ['$\sigma^+$', '$\pi$', '$\sigma^-$']):
        kwargs = {
            'laser_intensity': laser_intensity,
            'scale': scale,
            'laser_mode': laser_mode,
            'interaction_time': interaction_tof_optical,
            'background': background,
            'shape': 'lorentzian',
            'field': field,
            'fwhmG': 50}
        model_decay = RateModelDecay(*args, **kwargs)
        kwargs['interaction_time'] = interaction_tof_polar
        kwargs['scale'] = asymmetry
        model_polar = RateModelPolar(*args, **kwargs)

        frequencies = np.arange(model_decay.locations.min()-500, model_decay.locations.max()+500, 1)
        f_trans = energies[0] * EV_TO_MHZ
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[1].set_xlabel('Relative frequency [MHz]')
        ax[0].set_ylabel('Photons per nucleus [#]')
        ax[1].set_ylabel('Asymmetry')
        sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
        ax[0].plot(frequencies - f_trans, model_decay.integrate_with_time(frequencies, interaction_tof_optical, detection_tof))
        ax[1].plot(frequencies - f_trans, model_polar(frequencies))
        ax[0].set_title(title + ', ' + title_add)
        fig.savefig(title + '_' + str(laser_mode[0]), bbox_inches='tight')
# plt.show()
