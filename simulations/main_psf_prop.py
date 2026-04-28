#%%###########################################################################
"""
Generate PSFs for a lantern fiber, using the lanternfiber and SeidrSim classes 
"""

#%%########################################################################
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from seidr.zernike_to_lp import generate_temporal_lp_from_zernikes
from seidr.seidr_functions_misc import make_psf_lp_video


#%%###########################################################################
### Define Mode Orders ###

# LP modes
lp_mode_labels = {
    0: "LP01",
    1: "LP02",
    2: "LP03",
    3: "LP11a",
    4: "LP11b",
    5: "LP12a",
    6: "LP12b",
    7: "LP13a",
    8: "LP13b",
    9: "LP21a",
    10: "LP21b",
    11: "LP22a",
    12: "LP22b",
    13: "LP31a",
    14: "LP31b",
    15: "LP32a",
    16: "LP32b",
    17: "LP41a",
    18: "LP41b",
}

# Zernike modes
zernike_mode_labels = {
    # 0th Radial
    1: "Piston",
    # 1st Radial
    2: "Tilt X",
    3: "Tilt Y",
    # Second Radial
    4: "Defocus",
    5: "Astig X",
    6: "Astig Y",
    # Third Radial
    7: "Coma X",
    8: "Coma Y",
    9: "Trefoil X",
    10: "Trefoil Y",
    # Fourth Radial
    11: "Spherical",
    12: "2nd Astig X",
    13: "2nd Astig Y",
    14: "Quadrafoil X",
    15: "Quadrafoil Y",
    # Fifth Radial
    16: "2nd Coma X",
    17: "2nd Coma Y",
    18: "2nd Trefoil X",
    19: "2nd Trefoil Y",
    20: "Pentafoil X",
    21: "Pentafoil Y",
    # Sixth Radial
    22: "2nd Spherical",
    23: "3rd Coma X",
    24: "3rd Coma Y",
    25: "3rd Astig X",
    26: "3rd Astig Y",
    27: "Hexafoil X",
    28: "Hexafoil Y",
    # Seventh Radial
    29: "4th Coma X",
    30: "4th Coma Y",
    31: "4th Astig X",
    32: "4th Astig Y",
    33: "3rd Trefoil X",
    34: "3rd Trefoil Y",
    35: "Heptafoil X",
    36: "Heptafoil Y",
}

# hmspl7c_modes = [0, 1, 3, 4, 9, 10]

#%%########################################################################
### Set HMSPL and Simulation Parameters ##

## Simulation parameters
n_sims = 100 # number of simulations to run

wavel = 1.55 # wavelenth [um]

wf_npixels = 32  # number of pixels across the pupil plane for SeidrSim optics
psf_npixels = 96  # number of pixels across the PSF plane for SeidrSim optics

n_zernikes = 30  # number of Zernike modes to include in the random aberrations

focal_length = 20000 # focal length of the optics in um
pupil_diameter = 4500 #256*17  # diameter of the pupil in um
f_number = focal_length / pupil_diameter  # f-number of the optics

## HMSPL parameters ##

## Set the final cross-sectional scale
taper_ratio = 6.25

r_clad_out = 62.5 # [um]

## Input Radii ##
r_core_mm = r_clad_out/taper_ratio # cladding radius [um]
d_core_mm = 2 * r_core_mm  # MMF end core diameter [um]

# numerical window / padding scaling factor
max_r = 3 # [um]

## Refractive Indices ##
n_core_mm = 1.44
n_clad_mm = 1.4345

## Define Zernike coefficient wavefront error RMS values using Gaussian kernel smoothing
smooth_amt = 7 # Gaussian kernel samples / time steps

# Vary RMS per zernike mode, with a linear drop-off from start to end mode
max_rms_per_mode = 7e-8 #0.4 
min_rms_per_mode = 1e-8 #0.05

# ## Wavefront error RMS
# tiptilt_rms = 1e-7 # m
# ho_rms = 5e-8 # m

save_data = False # whether to save the generated dataset to disk

plot_video = True # whether to make a video of the PSF and LP power evolution
save_video = True # whether to save the video to disk

#%%
if __name__ == "__main__":
    
    sim, results = generate_temporal_lp_from_zernikes(
        n_sims=n_sims,
        # seed=1,
        wavel=wavel,
        f_number=f_number,
        pupil_diameter=pupil_diameter,
        n_core=n_core_mm,
        n_cladding=n_clad_mm,
        core_diameter=d_core_mm,
        max_r=max_r,
        wf_npixels=wf_npixels,
        psf_npixels=psf_npixels,
        n_zernikes=n_zernikes,
        # tiptilt_rms =tiptilt_rms,
        # ho_rms = ho_rms,
        max_rms_perterm=max_rms_per_mode,
        min_rms_perterm=min_rms_per_mode,
        smooth_amt=smooth_amt,
        return_fields=True
    )

    print("powers shape :", results["lp_powers"].shape)
    print("coeffs shape :", results["lp_coeffs"].shape)
    print("fields shape :", results["fields"].shape)
    print("number of LP modes :", results["nmodes"])

    ##########################################################################
    ### Plotting ###

    ## Plot one example
    idx = 0
    field = results["fields"][idx]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(field)**2)
    plt.title("PSF intensity at HMSPL input")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("PSF phase")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.bar(np.arange(results["nmodes"]), results["lp_powers"][idx])
    plt.xticks(np.arange(results["nmodes"]), results["modelabels"], rotation=90)
    plt.title("LP modal powers")
    plt.tight_layout()
    plt.show()

    ## Plot Zernike coefficients
    plt.figure(1)
    plt.plot(results["zernikes"][:100,:], '-o',markersize=2)
    plt.xlabel('Simulation Step')
    plt.ylabel('Zernike Coefficient Value')
    # plt.ylim([-1, 1])
    plt.grid(':', linewidth=0.5, alpha=0.5)
    plt.legend(['%s' % (zernike_mode_labels[k]) for k in range(1, n_zernikes+1)],
            loc='best', 
            fontsize=8,
            ncol=4)
    plt.show()

    if plot_video:
        make_psf_lp_video(results, outname="psf_lp_evolution.gif", 
                          save_video=save_video, fps=30, dpi=150)


    ########################################################################
    ### Save Dataset ###
    if save_data:
        np.savez(
            "seidr_option1_lp_dataset.npz",
            total_coupling=results["total_coupling"],
            powers=results["powers"],
            coeffs=results["coeffs"],
            zernikes=results["zernikes"],
            modelabels=results["modelabels"],
            fields=results["fields"],
        )
# %%
