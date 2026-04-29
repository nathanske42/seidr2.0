#%%###########################################################################
"""
Generate PSFs for a lantern fiber, using the lanternfiber and SeidrSim classes 
"""

#%%########################################################################
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import datetime

from seidr.zernike_to_lp import generate_temporal_lp_from_zernikes
from seidr.seidr_functions_misc import plot_wf_psf_zernike_lp, \
    make_wf_psf_video

#%%########################################################################
### Filenames ###

outname_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")

dir = "/import/roci1/nlon0790/Results/psf_prop/"
dir_plot = "/suphys/nlon0790/Documents/python_code/seidr2.0/figures/"

f_data = dir + "seidr_wf_psf_lp_dataset_test.npz" # + outname_datetime + ".npz"

f_plot = dir_plot + "seidr_wf_psf_lp_example.pdf" #+ outname_datetime + ".pdf"
f_video = dir_plot + "seidr_wf_psf_lp_evolution.gif" #+ outname_datetime + ".gif"


#%%########################################################################
### Set HMSPL and Simulation Parameters ##

## Simulation parameters
n_sims = 1000 # number of simulations to run

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

save_data = True # whether to save the generated dataset to disk
plot_example = False # whether to plot one example of the generated PSF, wavefront, and LP powers
save_video = False # whether to save the video to disk

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
        return_fields=True,
        return_wfs=True,
    )

    print("powers shape :", results["lp_powers"].shape)
    print("coeffs shape :", results["lp_coeffs"].shape)
    print("fields shape :", results["psf_fields"].shape)
    print("pupil wfs shape :", results["pupil_wfs"].shape)
    print("number of LP modes :", results["nmodes"])
    print("zernike coeffs shape :", results["zernike_coeffs"].shape)

    ##########################################################################
    ### Plotting ###

    ## Plot one example
    idx_rand = np.random.randint(0, n_sims) # pick a random simulation to plot

    if plot_example:
        plot_wf_psf_zernike_lp(results, idx=idx_rand, save_plot=True,
                               fname_plot=f_plot)

    if save_video:
        make_wf_psf_video(results, outname=f_video,
                          save_video=save_video, fps=30, dpi=150)


    ########################################################################
    ### Save Dataset ###
    if save_data:
        np.savez(
            f_data,
            total_coupling=results["total_coupling"],
            lp_powers=results["lp_powers"],
            lp_coeffs=results["lp_coeffs"],
            zernike_coeffs=results["zernike_coeffs"],
            modelabels=results["modelabels"],
            psf_fields=results["psf_fields"],
            pupil_wfs=results["pupil_wfs"],
        )

        print(f"Saved dataset to {f_data}")
# %%

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# plt.imshow(np.abs(field)**2)
# plt.title("PSF intensity at HMSPL input")
# plt.colorbar()

# plt.subplot(1, 3, 2)
# plt.imshow(np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
# plt.title("PSF phase")
# plt.colorbar()

# plt.subplot(1, 3, 3)
# plt.bar(np.arange(results["nmodes"]), results["lp_powers"][idx])
# plt.xticks(np.arange(results["nmodes"]), results["modelabels"], rotation=90)
# plt.title("LP modal powers")
# plt.tight_layout()
# plt.show()

 # field = results["fields"][idx_rand]
    # wf = results["pupil_wfs"][idx_rand]
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(wf, cmap="twilight", 
    #            vmin=-np.pi, vmax=np.pi)
    # plt.title("Wavefront")
    # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.imshow(np.abs(field)**2)
    # plt.title("PSF intensity at HMSPL input")
    # plt.colorbar()

    # plt.subplot(1, 3, 3)
    # plt.bar(np.arange(results["nmodes"]), results["lp_powers"][idx])
    # plt.xticks(np.arange(results["nmodes"]), results["modelabels"], rotation=90)
    # plt.title("LP modal powers")
    # plt.tight_layout()
    # plt.show()



    # ## Plot Zernike coefficients
    # plt.figure(1)
    # plt.plot(results["zernikes"][:100,:], '-o',markersize=2)
    # plt.xlabel('Simulation Step')
    # plt.ylabel('Zernike Coefficient Value')
    # # plt.ylim([-1, 1])
    # plt.grid(':', linewidth=0.5, alpha=0.5)
    # plt.legend(['%s' % (zernike_mode_labels[k]) for k in range(1, n_zernikes+1)],
    #         loc='best', 
    #         fontsize=8,
    #         ncol=4)
    # plt.show()