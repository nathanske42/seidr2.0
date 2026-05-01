#%%###########################################################################
"""
Generate PSFs for a lantern fiber, using the 
lanternfiber and zernikePSF classes 
"""

#%%########################################################################
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import datetime

from seidr.source2pl import source2pl_temporal
from seidr.seidr_functions_misc import plot_wf_psf_zernike_lp, \
    make_wf_psf_video_square, make_wf_psf_lp_pl_video_row

#%%########################################################################
### Filenames ###

outname_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")

dir = "/import/roci1/nlon0790/Results/psf_prop/"
dir_plot = "/suphys/nlon0790/Documents/python_code/seidr2.0/figures/"

f_data = dir + "seidr_wf_psf_lp_dataset_test.npz" # + outname_datetime + ".npz"

f_plot = dir_plot + "seidr_wf_psf_lp_example.pdf" #+ outname_datetime + ".pdf"
f_video = dir_plot + "seidr_wf_psf_lp_evolution.gif" #+ outname_datetime + ".gif"

## transfer matrix 
f_pl_name = "hms-pl7c"
f_pl_path = "/import/roci1/nlon0790/Results/hms-pl7/cores/"

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
max_rms_per_mode = 7e-8 #0.4 [m]
min_rms_per_mode = 1e-8 #0.05 [m]

# ## Wavefront error RMS
# tiptilt_rms = 1e-7 # m
# ho_rms = 5e-8 # m

## Transfer Matrix Parameters
calc_pl_outputs = True # whether to calculate PL outputs using the transfer matrix
if calc_pl_outputs:
    n_cores = 7
    wavel = 1.55 # wavelenth [um]
    r_ms = 0.559 # radius of mode selective core [um]
    ds = 0.25 # x-y step size for LB sims [um]
    dz = 2 # z-step size for LB sims [um]
    rv = 1 # reference value for LB sims
    xywidth = 135.0 # width of the simulation window [um]
    z_len = 50000 # length of PL [um]
    tr = 6.25 # taper ratio
else:
    f_pl_path = None
    r_ms = None
    ds = None
    dz = None
    rv = None
    xywidth = None
    z_len = None
    tr = None

save_data = False # whether to save the generated dataset to disk
plot_example = False # whether to plot one example of the generated PSF, wavefront, and LP powers
save_video = True # whether to save the video to disk

#%%
if __name__ == "__main__":
    
    lf, results = source2pl_temporal(
        n_sims=n_sims,
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
        max_rms_perterm=max_rms_per_mode,
        min_rms_perterm=min_rms_per_mode,
        smooth_amt=smooth_amt,
        return_wfs=True,
        calc_pl_outputs=True,
        f_pl_path=f_pl_path,
        f_pl_name=f_pl_name,
        r_core=r_ms,
        ds=ds,
        dz=dz,
        rv=rv, 
        xywidth=xywidth,
        z_len=z_len, 
        tr=tr,
    )

    print("powers shape :", results["lp_powers"].shape)
    print("coeffs shape :", results["lp_coeffs"].shape)
    print("fields shape :", results["psf_fields"].shape)
    print("pupil wfs shape :", results["pupil_wfs"].shape)
    print("number of LP modes :", results["nmodes"])
    print("zernike coeffs shape :", results["zernike_coeffs"].shape)

    if calc_pl_outputs:
        print("PL outputs shape :", results["pl_outputs"].shape)

    ##########################################################################
    ### Plotting ###

    ## Plot one example
    idx_rand = np.random.randint(0, n_sims) # pick a random simulation to plot

    if plot_example:
        plot_wf_psf_zernike_lp(results, idx=idx_rand, save_plot=True,
                               fname_plot=f_plot)

    if save_video:
        make_wf_psf_video_square(results, outname=f_video,
                          save_video=save_video, fps=30, dpi=150)
        
        # make_wf_psf_lp_pl_video_row(results, 
        #                             outname=f_video.replace(".gif", "_row.gif"),
        #                             save_video=save_video, fps=30, dpi=150)


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
            pl_outputs=results["pl_outputs"] if calc_pl_outputs else None,
            pl_powers=results["pl_powers"] if calc_pl_outputs else None,
        )

        print(f"Saved dataset to {f_data}")
#%%

idx_example = np.random.randint(0, n_sims)
plt.figure(figsize=(8, 4))
plt.bar(range(n_cores), np.abs(results["pl_outputs"][idx_example, :])**2)
plt.xlabel('Core Index')
plt.ylabel('Output Intensity')
plt.title('Propagation of PSF through PL - Example Simulation')
plt.grid(':', linewidth=0.5, alpha=0.5)
plt.show()


#%%

idx_example = np.random.randint(0, n_sims)

print(np.sum(np.abs(results["psf_fields"][idx_example, :])**2))
print(np.sum(results["lp_powers"][idx_example, :]))
print(np.sum(np.abs(results["pl_outputs"][idx_example, :])**2))

#%%
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