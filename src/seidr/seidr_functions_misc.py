#%%
"""
Defines Functions for Use in End-to-End Seidr Simulations
"""

#%% Import Libraries and Modules

import jax
import jax.numpy as np
from jax import random

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