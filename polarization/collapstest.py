from satlasaddon import RateModelDecay, RateModelPolar, convolve_with_gaussian
import satlas as sat
sat.set(['standard'])
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as csts
import copy
EV_TO_MHZ = csts.physical_constants['electron volt-hertz relationship'][0] * 1e-6


model_decay = RateModelDecay(I=0,J=[0.5,0.5],L=[1,0],ABC=[[0,0,0],[0,0,0]],centroids = [0],energies = [2.9380,0],A_array=np.array([[0,10*10**6],[0,0]]),scale=1,shape='Lorentzian'\
							,laser_intensity = 10,laser_mode=[0],interaction_time=100*10**(-6),fwhmL=None,background=0,\
							field=0,fixed_frequencies=None, frequency_mode='fixed')

fig, ax = plt.subplots(2, 1, sharex=True)

f_trans = 2.938 * EV_TO_MHZ
f = f_trans + np.linspace(-10, 10, 1000)
ax[0].plot(f-f_trans, model_decay.integrate_with_time(f, 20*10**(-6), 1*10**(-6), steps=1001), label='20')
ax[0].plot(f-f_trans, model_decay.integrate_with_time(f, 50*10**(-6), 1*10**(-6), steps=1001), label='50')
ax[1].plot(f-f_trans, model_decay(f))
ax[0].legend(loc=0)
plt.show()
