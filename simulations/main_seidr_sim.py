#%%########################################################################
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from seidr.zernike_to_lp import generate_lp_from_zernikes


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

## Wavefront error RMS
tiptilt_rms = 1e-6 # m
ho_rms = 5e-7 # m

save_data = False # whether to save the generated dataset to disk


#%%
if __name__ == "__main__":
    
    sim, results = generate_lp_from_zernikes(
        n_sims=n_sims,
        seed=1,
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
        tiptilt_rms =tiptilt_rms,
        ho_rms = ho_rms,
        return_fields=True
    )

    print("powers shape :", results["lp_powers"].shape)
    print("coeffs shape :", results["lp_coeffs"].shape)
    print("fields shape :", results["fields"].shape)
    print("number of LP modes :", results["nmodes"])

    # Plot one example
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

    # Save dataset
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
