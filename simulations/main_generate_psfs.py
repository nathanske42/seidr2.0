#%%###########################################################################
"""
Generate PSFs for a lantern fiber, using the lanternfiber class. 
"""

#%%###########################################################################
import numpy as np
import matplotlib.pyplot as plt

from seidr.lanternfiber import lanternfiber

# %autoreload


#%%###########################################################################
### Setup Parameters ###

wavelength = 1.5 # microns

### Define HMSPL MMF Parameters
n_core = 1.44 # Refractive index of the cladding (input core to PL)
n_cladding = 1.4345

core_radius = 32.8/2 # microns

### Scale parameters
max_r = 2 # Maximum radius to calculate mode field, where r=1 is the core diameter
npix = 200 # Half-width of mode field calculation in pixels

show_plots = True

#%%###########################################################################
### Make the fiber and modes
f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
f.find_fiber_modes()
f.make_fiber_modes(npix=npix, show_plots=False, max_r=max_r)


#%%###########################################################################
### Make an arbitrary input field - here two spots, in antiphase
power = 0.5
posn_microns = [0, 0]
sigma = 10

phase = 0
f.make_arb_input_field('gaussian', 
                       power=power, 
                       location=posn_microns, 
                       sigma=sigma, 
                       add_to_existing=False, 
                       show_plots=show_plots, 
                       phase=phase)


#%%
posn_microns = [13, 0]
phase = np.pi
f.make_arb_input_field('gaussian', 
                       power=power, 
                       location=posn_microns, 
                       sigma=sigma,
                       add_to_existing=True, 
                       show_plots=show_plots, 
                       phase=phase)


#%%
### Calculate coupling for one mode, and make some plots
mode_to_measure = 1
f.plot_injection_field(f.input_field)
coupling, coupling_complex = f.calc_injection(mode_field_number=mode_to_measure, verbose=True)


### Calculate coupling for all fiber modes
modes_to_measure = np.arange(f.nmodes)
coupling, mode_coupling, mode_coupling_complex = f.calc_injection_multi(mode_field_numbers=modes_to_measure,
                                                 verbose=True, show_plots=True, fignum=2, complex=True)




