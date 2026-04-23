#%%########################################################################
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from seidr.zernike_to_lp import generate_lp_with_zernikes


#%%########################################################################
### Set HMSPL and Simulation Parameters ##

## Simulation parameters
n_sims = 100 # number of simulations to run

wavel = 1.55 # wavelenth [um]

wf_npixels = 32  # number of pixels across the pupil plane for SeidrSim optics
psf_npixels = 96  # number of pixels across the PSF plane for SeidrSim optics

n_zernikes = 30  # number of Zernike modes to include in the random aberrations

f_number = 5.5  # f-number of the optics


## HMSPL parameters ##

## Set the final cross-sectional scale
taper_ratio = 6.25

r_clad_out = 62.5 # [um]

## Input Radii ##
r_core_mm = r_clad_out/taper_ratio # cladding radius [um]
d_core_mm = 2 * r_core_mm  # MMF end core diameter [um]

max_r = r_core_mm + 5 # [um]

## Refractive Indices ##
n_core_mm = 1.44
n_clad_mm = 1.4345





#%%
if __name__ == "__main__":
    
    sim, results = generate_lp_with_zernikes(
        n_sims=n_sims,
        seed=1,
        wavel=wavel,
        n_core=n_core_mm,
        n_cladding=n_clad_mm,
        core_diameter=d_core_mm,
        max_r=max_r,
        wf_npixels=wf_npixels,
        psf_npixels=psf_npixels,
        n_zernikes=n_zernikes,
        f_number=f_number,
        tiptilt_rms=50e-9,
        ho_rms=20e-9,
        return_fields=True,
        save_data=False,
    )

    print("powers shape :", results["powers"].shape)
    print("coeffs shape :", results["coeffs"].shape)
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
    plt.bar(np.arange(results["nmodes"]), results["powers"][idx])
    plt.xticks(np.arange(results["nmodes"]), results["modelabels"], rotation=90)
    plt.title("LP modal powers")
    plt.tight_layout()
    plt.show()

    # Save dataset
    if save_data == True:
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
