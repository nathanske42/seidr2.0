#%%###########################################################################
"""
Overall plan:
- program inputs: companion properties
- fixed constants: detector, VLTI properties
- outputs: null depth distribution and hence SNR as ratio of companion light 
  to starlight
- define inputs from baldr as a distribution of zernikies
- apply arbitrary correction from PL loop
- assume first n LP modes are injected (for n=1,3)
- Correction due to kernel nuller chip
- look at overall null depth
"""

#%%###########################################################################
### Import Libraries and Modules

import jax
import jax.numpy as np
from jax import random

import dLux.utils as dlu
import matplotlib.pyplot as plt


#%%###########################################################################
### Define Functions

##############################################################################
def nuller_given_position(Tscope_positions, companion_positions, null_matrix,
                          wavelength):
    """Calculate the nuller response for a given companion position."""

    phases = (
        (Tscope_positions - Tscope_positions[2]) \
            * dlu.arcsec2rad(companion_positions) / wavelength
    ).sum(axis=1)

    fields = np.exp(1j * phases)

    # what does the detector see
    detector_outputs = np.abs(null_matrix @ fields) ** 2

    return detector_outputs


##############################################################################
def nuller_response_matrix(m_type='changaipe'):
    """Return the nuller response matrix for the specified type."""

    if m_type == 'chingaipe':
        # Chingaipe version

        M_matrix = 0.25 * np.array(
            [
                [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j],
                [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
                [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j],
                [1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j],
                [1 + 1j, -1 - 1j, -1 + 1j, 1 - 1j],
            ],
            dtype=np.complex64,
        )

    elif m_type == 'martinache':
        # Martinache version

        theta = np.pi / 2
        M_matrix = 0.25 * np.array(
            [
                [
                    1 + np.exp(1j * theta),
                    1 - np.exp(1j * theta),
                    -1 + np.exp(1j * theta),
                    -1 - np.exp(1j * theta),
                ],
                [
                    1 - np.exp(-1j * theta),
                    -1 - np.exp(-1j * theta),
                    1 + np.exp(-1j * theta),
                    -1 + np.exp(-1j * theta),
                ],
                [
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                ],
                [
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                ],
                [
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                ],
                [
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                    1 + np.exp(1j * theta),
                ],
            ],
            dtype=np.complex64,
        )
    else:
        raise ValueError("Invalid matrix type")

    return M_matrix


#%%###########################################################################
### Select Telescope Parameters

# tscope_type = "UT"  # UT or AT

# if tscope_type == "UT":
#     diameter = 8.2
# elif tscope_type == "AT":
#     diameter = 1.8
# else:
#     raise ValueError("Invalid scope type")

## Telescope positions in meters (x, y)
#  "martinache" example or "vlti"
telescope = "vlti"

if telescope == "martinache":
    Tscope_positions = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [40.0, 0.0],
            [60.0, 0.0],
        ]
    )
elif telescope == "vlti":
    Tscope_positions = np.array(
        [
            [-9.925, -20.335],
            [14.887, 30.502],
            [44.915, 66.183],
            [103.306, 44.999],
        ]
    )

#%%###########################################################################
### Setup Matrices

## Nuller response matrix
M_matrix = nuller_response_matrix(m_type='chingaipe')

# raise NotImplementedError

## Nulling matrix
N_matrix = 0.5 * np.array(
    [
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)

## Kernel operator matrix (used to erase second order phase errors)
K_matrix = np.array(
    [
        [1, -1, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 1, -1],
    ],
    dtype=np.float32,
)


#%%###########################################################################
### Define Companion Properties

# contrast = 1e-2 # -> not used
wavelength = 1.6e-6  # meters

n_samp = 100 # number of samples
max_sep = 42 # maximum separation *in arcsecs (lambda/D for UT at 1.6um is ~40mas)* ?


companion_positions = np.vstack(
    [np.linspace(-max_sep, max_sep, n_samp) * 1e-3, np.zeros(n_samp)]
).T  # mas
print(companion_positions)


#%%###########################################################################
### Nuller Troughput as a Function of Companion Position

oned_throughputs = jax.vmap(lambda c: nuller_given_position(Tscope_positions, 
                                                            c, N_matrix, 
                                                            wavelength))(
    companion_positions,
)
print(oned_throughputs.shape)

plt.figure()
plt.plot(companion_positions[:, 0] * 1e3, oned_throughputs)
plt.xlabel("R.A. Off-Axis Position [mas]")
plt.ylabel("Nuller Output Transmission")
plt.title("Classical Nuller Design")
plt.grid(linestyle=':', linewidth=0.5)
plt.legend(['output #1', 'output #2', 'output #3'])
plt.show()

#%%###########################################################################
### Modified Nuller Design

oned_throughputs = jax.vmap(lambda c: nuller_given_position(Tscope_positions, 
                                                            c, M_matrix, 
                                                            wavelength))(
    companion_positions,
)
print(oned_throughputs.shape)

plt.figure()
plt.plot(companion_positions[:, 0] * 1e3, oned_throughputs)
plt.xlabel("Companion Position [mas]")
plt.ylabel("Nuller Throughput")
plt.xlabel("R.A. Off-Axis Position [mas]")
plt.ylabel("Nuller Output Transmission")
plt.title("Modified Nuller Design")
plt.grid(linestyle=':', linewidth=0.5)
plt.legend(['output #1', 'output #2', 'output #3', 
            'output #4', 'output #5', 'output #6'])
plt.show()


#%%###########################################################################
### Kernel Nuller Outputs
kernel_outputs = K_matrix @ oned_throughputs.T

plt.figure()
plt.plot(companion_positions[:, 0] * 1e3, kernel_outputs.T)
plt.xlabel("R.A. Off-Axis Position [mas]")
plt.ylabel("Kernel Output")
plt.title("Kernel Outputs")
plt.grid(linestyle=':', linewidth=0.5)
plt.legend(['output #1', 'output #2', 'output #3'])
plt.show()


#%%###########################################################################
### Kernel Nulling Example Based on Input Noise (rather than actual geometry)

## Simulation parameters
n_beams = 4  # number of beams
n_runs = 1000

## Noise properties
sigma_I = 0.001  # rms intensity error
sigma_phi = 10e-9  # rms phase error

## Generate Random Noisy Input Fields
input_beam_amplitude = random.normal(random.PRNGKey(0), 
                                     (n_beams, n_runs)) * sigma_I + 1

input_beam_phase = random.normal(random.PRNGKey(1), 
                                 (n_beams, n_runs)) * sigma_phi

input_beam_field = input_beam_amplitude * np.exp(1j * input_beam_phase)

## Calculate Nuller Outputs
detector_outputs = np.abs(M_matrix @ input_beam_field) ** 2

## Calculate Kernel Nuller Outputs
kernel_outputs = K_matrix @ detector_outputs

print(np.std(kernel_outputs, axis=1))

## Plot Results
plt.figure()
plt.hist(kernel_outputs[0], 
         bins=50, alpha=0.5, 
         label="kernel 1")
plt.hist(kernel_outputs[1], 
         bins=50, alpha=0.5, 
         label="kernel 2")
plt.hist(kernel_outputs[2], 
         bins=50, alpha=0.5, 
         label="kernel 3")

plt.legend()
plt.grid(linestyle=':', linewidth=0.5)
plt.xlabel("Kernel Output")
plt.ylabel("Count")
plt.title("Kernel Output Distribution with Noise")
plt.show()

# %%
