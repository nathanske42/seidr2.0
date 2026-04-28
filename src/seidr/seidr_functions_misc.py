#%%
"""
Defines Functions for Use in End-to-End Seidr Simulations
"""

#%% Import Libraries and Modules

import h5py
import jax
import jax.numpy as jnp
from jax import random
import h5py
from scipy import ndimage
import numpy as np


import dLux.utils as dlu
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#%% Define Functions

##############################################################################
def correlated_noise(correlation_time, rms_amplitudes, sample_times, 
                     key=random.PRNGKey(0)):
    """
        Used to generate correlated aberrations
    """
    ## Sample the correlated aberrations
    
    del_ts = jnp.diff(sample_times)

    ## Draw the initial Gaussian sample
    key, subkey = random.split(key)
    noise = random.normal(subkey, (1, len(rms_amplitudes))) \
        * rms_amplitudes

    rs = jnp.exp(-del_ts / correlation_time)
    tilde_sigs = jnp.sqrt(1 - rs**2)[:, None] * rms_amplitudes[None, :]

    for i in range(1, len(sample_times)):

        key, subkey = random.split(key)
        noise = jnp.vstack(
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


##############################################################################
def make_smoothrand(nsteps, nvecs=1, smthamt=10., 
                    finalsd=1.):

    smthrand_all = np.zeros((nsteps, nvecs))

    for k in range(nvecs):
        noisevec = np.random.randn(nsteps)
        smthrand = ndimage.gaussian_filter1d(noisevec, smthamt)
        smthrand = smthrand / np.std(smthrand) * finalsd
        smthrand_all[:,k] = smthrand

    return smthrand_all


##############################################################################
def make_smoothrand_multi(nsteps, nvecs=1, smthamt=10., 
                          finalsds=1.):
    
    smthrand_all = np.zeros((nsteps, nvecs))

    for k in range(nvecs):
        noisevec = np.random.randn(nsteps)
        smthrand = ndimage.gaussian_filter1d(noisevec, smthamt)
        smthrand = smthrand / np.std(smthrand) * finalsds[k]
        smthrand_all[:,k] = smthrand
        
    return smthrand_all

##############################################################################
def zernike_rms_per_mode(max_rms_perterm, min_rms_perterm, n_zernikes):

   # Vary RMS per zernike mode, with a linear drop-off from start to end mode 
    rms_perterm_multi = np.linspace(max_rms_perterm, min_rms_perterm, 
                                    n_zernikes-1)
    
    # Add a zero for the piston mode
    rms_perterm_multi = np.concatenate(([0], rms_perterm_multi))

    # make tip / tilt the same
    rms_perterm_multi[1] = rms_perterm_multi[2]
        
    return rms_perterm_multi


##############################################################################
# def norm_coeffs(coeffs_in):
#     # Normalise cofficients so polynomials are [-1,1], like zernfun.m
#     coeffs_out = np.zeros_like(coeffs_in)
#     for k in range(coeffs_in.shape[1]):
#         n = cart.ntab[k]
#         m = cart.mtab[k]
#         if m == 0:
#             normfact = np.sqrt(n + 1)
#         else:
#             normfact = np.sqrt(2 * (n + 1))
#         coeffs_out[:, k] = coeffs_in[:, k] / normfact
#     return coeffs_out



##############################################################################
def make_psf_lp_video(results, outname="psf_lp_evolution.gif", 
                      save_video=False, fps=15, dpi=150):

    fields = np.asarray(results["fields"])
    lp_powers = np.asarray(results["lp_powers"])
    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]

    # Fixed limits so the movie does not flicker frame-to-frame
    intensity_all = np.abs(fields)**2
    intensity_vmax = np.max(intensity_all)
    lp_vmax = np.max(lp_powers)

    fig = plt.figure(figsize=(12, 4))

    # ------------------------------------------------------------------
    # Panel 1: PSF intensity
    # ------------------------------------------------------------------
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(
        intensity_all[0],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
    )
    ax1.set_title("PSF intensity at HMSPL input")
    cbar1 = plt.colorbar(im1, ax=ax1)

    # ------------------------------------------------------------------
    # Panel 2: PSF phase
    # ------------------------------------------------------------------
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(
        np.angle(fields[0]),
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax2.set_title("PSF phase")
    cbar2 = plt.colorbar(im2, ax=ax2)

    # ------------------------------------------------------------------
    # Panel 3: LP modal powers
    # ------------------------------------------------------------------
    ax3 = plt.subplot(1, 3, 3)
    x = np.arange(nmodes)
    bars = ax3.bar(x, lp_powers[0])
    ax3.set_xticks(x)
    ax3.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax3.set_ylim(0, 1.05 * lp_vmax)
    ax3.set_title("LP modal powers")

    frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

    plt.tight_layout()

    def update(idx):
        field = fields[idx]

        # Update PSF intensity
        im1.set_data(np.abs(field)**2)

        # Update PSF phase
        im2.set_data(np.angle(field))

        # Update LP bar heights
        for bar, height in zip(bars, lp_powers[idx]):
            bar.set_height(height)

        frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

        return [im1, im2, frame_title, *bars]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_sims,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        writer = PillowWriter(fps=fps)
        # else:
        #     writer = FFMpegWriter(fps=fps, bitrate=3000)

        anim.save(outname, writer=writer, dpi=dpi)
        plt.close(fig)

        print(f"Saved video to {outname}")