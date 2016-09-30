from satlasaddon import RateModelDecay, RateModelPolar, convolve_with_gaussian
import satlas as sat
sat.set(['standard'])
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as csts
import copy

KEPCO = 100
AMU_TO_KG = 1.66053904 * 10**(-27)
EV_TO_J = 1.60217662 * 10**(-19)
EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6

beam_energy = 45e3
beam_energy *= EV_TO_J

A_array_D1 = np.array([[0, 61.353e6], [0, 0]])
A_array_D2 = np.array([[0, 61.542e6], [0, 0]])

energies_D1 = [2.102297159, 0]
energies_D2 = [2.104429184, 0]

J_D1 = [0.5, 0.5]
J_D2 = [1.5, 0.5]
L = [1, 0]

geometrical_efficiency = 14.941971804041613 / 100
quantum_efficiency = 0.175

# 23Na
mu_23 = 2.2176556
q_23 = 0.1045
I_23 = 3/2

ABC_23_D1 = np.array([[94.44, 0, 0], [885.813, 0, 0]])
ABC_23_D2 = np.array([[18.534, 2.724, 0], [885.813, 0, 0]])

mass_23 = 22.9897692809
asymmetry_23 = 0
title_23 = '23Na'

# 26Na
mu_26 = 2.851
I_26 = 3
q_26 = -0.0053

A_factor = I_23 / mu_23 * mu_26 / I_26
B_factor = q_26 / q_23

ABC_26_D1 = copy.deepcopy(ABC_23_D1)
ABC_26_D1[0, 0] *= A_factor
ABC_26_D1[1, 0] *= A_factor

ABC_26_D2 = copy.deepcopy(ABC_23_D2)
ABC_26_D2[0, 0] *= A_factor
ABC_26_D2[1, 0] *= A_factor
ABC_26_D2[0, 1] *= B_factor
ABC_26_D2[1, 1] *= B_factor

IS_26_D1 = 1397.5
IS_26_D2 = 1397.5

mass_26 = 25.992633
asymmetry_26 = -0.93
title_26 = '26Na'

# 28Na
mu_28 = 2.426
I_28 = 1
q_28 = q_26 * -7.7

B_factor = q_28 / q_23
A_factor = I_23 / mu_23 * mu_28 / I_28

ABC_28_D1 = copy.deepcopy(ABC_23_D1)
ABC_28_D1[0, 0] *= A_factor
ABC_28_D1[1, 0] *= A_factor

ABC_28_D2 = copy.deepcopy(ABC_23_D2)
ABC_28_D2[0, 0] *= A_factor
ABC_28_D2[1, 0] *= A_factor
ABC_28_D2[0, 1] *= B_factor
ABC_28_D2[1, 1] *= B_factor

IS_28_D1 = 2985.8
IS_28_D2 = 2985.8

mass_28 = 27.998938
asymmetry_28 = -0.75
title_28 = '28Na'

IS_23 = 0
energies = energies_D2
ABC_23 = ABC_23_D2
ABC_26 = ABC_26_D2
IS_26 = IS_26_D2
IS_28 = IS_28_D2
ABC_28 = ABC_28_D2
J = J_D2
A_array = A_array_D2

i = [I_23, I_26, I_28]
abc = [ABC_23, ABC_26, ABC_28]
asym = [asymmetry_23, asymmetry_26, asymmetry_28]
m = [mass_23, mass_26, mass_28]
tit = [title_23, title_26, title_28]
cen = [IS_23, IS_26, IS_28]

area = 0.8 #cm^2
laser_power_optical = 10 #mW
laser_intensity_optical = laser_power_optical / area * 10  # W/m^2
laser_power_asymmetry = 40 #mW
laser_intensity_asymmetry = laser_power_asymmetry / area * 10  # W/m^2
laser_intensity_optical = [laser_intensity_optical]
laser_intensity_asymmetry = [laser_intensity_asymmetry]

for I, ABC, centroid, mass, asymmetry, title in zip(i, abc, cen, m, asym, tit):
    mass *= AMU_TO_KG
    velocity = np.sqrt(2 * beam_energy / mass)

    detection_length = 100e-3
    detection_tof = detection_length / velocity

    interaction_length_optical = 20e-2
    interaction_tof_optical = interaction_length_optical / velocity

    interaction_length_polar = 2.2
    interaction_tof_polar = interaction_length_polar / velocity

    scale = geometrical_efficiency * quantum_efficiency
    centroids = [centroid]
    field = 2 * 10e-4
    background = [0]
    args = (I, J, L, ABC, centroids, energies, A_array)
    fig, ax = plt.subplots(2, 1, sharex=True)
    for laser_mode, title_add in zip([[-1]], ['$\sigma^-$']):
        kwargs = {
            'laser_intensity': laser_intensity_optical,
            'scale': scale,
            'laser_mode': laser_mode,
            'interaction_time': interaction_tof_optical,
            'background_params': background,
            'shape': 'voigt',
            'field': field,
            'fwhmG': 0}
        model_decay = RateModelDecay(*args, **kwargs)
        kwargs['interaction_time'] = interaction_tof_polar
        kwargs['scale'] = asymmetry
        kwargs['laser_intensity'] = laser_intensity_asymmetry
        model_polar = RateModelPolar(*args, **kwargs)

        f_trans = energies[0] * EV_TO_MHZ + centroids[0]
        laser_wavenumber = (f_trans *1e6) * csts.physical_constants['hertz-inverse meter relationship'][0] / 100
        step = 1
        voltage = np.arange(-200, 200 + step, step)
        doppler_factor = sat.utilities.dopplerfactor(mass / AMU_TO_KG, beam_energy / EV_TO_J - voltage)
        f_trans = f_trans / sat.utilities.dopplerfactor(mass / AMU_TO_KG, beam_energy / EV_TO_J)
        deriv = f_trans
        # f_trans = 17000 / (1e6 * csts.physical_constants['hertz-inverse meter relationship'][0] / 100)
        laser_wavenumber = (f_trans *1e6) * csts.physical_constants['hertz-inverse meter relationship'][0] / 100

        frequencies = f_trans * doppler_factor
        print((frequencies[-1] - frequencies[0])/(voltage[-1]-voltage[0]))
        import lmfit as lm
        print(laser_intensity_asymmetry)
        lm.report_fit(model_polar.params)
        ax[1].set_xlabel('Line voltage [V]')
        ax[0].set_ylabel('Photons per nucleus [#]')
        ax[1].set_ylabel('Asymmetry')
        sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
        # ax[0].plot(voltage / KEPCO, model_decay.integrate_with_time(frequencies, interaction_tof_optical, detection_tof), label=title_add)
        ax[1].plot(voltage / KEPCO, model_polar(frequencies))
        ax[0].set_title(title + ', {:.4f} cm$^{{-1}}$'.format(laser_wavenumber))
        # fig.savefig(title + '_' + str(laser_mode[0]), bbox_inches='tight')
        ax[0].legend(loc=0)
plt.show()
