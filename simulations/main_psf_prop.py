"""
Load a pre-computed transfer matrix and use it to predict photonic lantern outputs.

The transfer matrix maps complex LP mode amplitudes at the MM input to complex
amplitudes at each SM output waveguide. In practice only the output intensities
(|amplitude|^2) are measurable.
"""

import matplotlib.pyplot as plt
import numpy as np
from seidr.lanternfiber import lanternfiber
plt.ion()


datadir = './'
mm2sm_filename = 'extractedvals_probeset_19LP__Good202107.npz'

f = lanternfiber(datadir=datadir, nmodes=6, nwgs=7)
f.load_savedvalues(mm2sm_filename)
f.make_transfer_matrix_mm2sm(show_plots=True)


transfer_matrix = f.Cmat  # complex transfer matrix, shape (nwgs, nmodes)

# Input: complex amplitudes of each LP mode at the MM fiber entrance
input_modecoeffs = (np.random.uniform(-1, 1, f.nmodes) +
                    1j * np.random.uniform(-1, 1, f.nmodes))

# Predicted complex output amplitudes at each SM waveguide
pl_outputs = transfer_matrix @ input_modecoeffs

# Measurable output: intensity at each SM waveguide
pl_output_fluxes = np.abs(pl_outputs) ** 2

print("Output fluxes:", pl_output_fluxes)
