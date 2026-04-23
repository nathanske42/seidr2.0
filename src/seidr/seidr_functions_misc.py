#%%
"""
Defines Functions for Use in End-to-End Seidr Simulations
"""

#%% Import Libraries and Modules

import h5py
import jax
import jax.numpy as np
from jax import random
import h5py

import dLux.utils as dlu
import matplotlib.pyplot as plt

#%% Define Functions

##############################################################################
def correlated_noise(correlation_time, rms_amplitudes, sample_times, 
                     key=random.PRNGKey(0)):
    """
        Used to generate correlated aberrations
    """
    ## Sample the correlated aberrations
    
    del_ts = np.diff(sample_times)

    ## Draw the initial Gaussian sample
    key, subkey = random.split(key)
    noise = random.normal(subkey, (1, len(rms_amplitudes))) \
        * rms_amplitudes

    rs = np.exp(-del_ts / correlation_time)
    tilde_sigs = np.sqrt(1 - rs**2)[:, None] * rms_amplitudes[None, :]

    for i in range(1, len(sample_times)):

        key, subkey = random.split(key)
        noise = np.vstack(
            [
                noise,
                rs[i - 1] * noise[-1]
                + tilde_sigs[i - 1]
                * random.normal(subkey, (1, len(rms_amplitudes))),
            ]
        )

    return noise

##############################################################################
def load_lb_transfer_matrix(f_path, f_pl_name, 
                            wl, r_ms, ds, 
                            dz, rv, xywidth, 
                            z_len, tr):
    """
        Used to load complex transfer matrices for the lantern fiber,
        as calculated using lightbea.
    """

    f_name = f_path + f_pl_name + '_C_lm_array__wl=' + str(wl) \
        + '_rms=' + str(r_ms) + '_ds=' + str(ds) + '_dz=' + str(dz) \
        + '_rv=' + str(rv) + '_xyw=' + str(xywidth) + '_zlen=' + str(z_len) \
        + '_tr=' + str(tr) + '.h5'

    C_lm_data = h5py.File(f_name, 'r')
    C_lm_array = C_lm_data['C_lm'][:]
    C_lm_data.close()

    return C_lm_array