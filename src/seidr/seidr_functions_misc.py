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

    f_name = f_path + f_pl_name + '_C_lm_array_wl=' + str(wl) \
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
# def make_psf_lp_video(results, outname="psf_lp_evolution.gif", 
#                       save_video=False, fps=15, dpi=150):

#     fields = np.asarray(results["fields"])
#     lp_powers = np.asarray(results["lp_powers"])
#     modelabels = np.asarray(results["modelabels"])
#     nmodes = int(results["nmodes"])

#     n_sims = fields.shape[0]

#     # Fixed limits so the movie does not flicker frame-to-frame
#     intensity_all = np.abs(fields)**2
#     intensity_vmax = np.max(intensity_all)
#     lp_vmax = np.max(lp_powers)

#     fig = plt.figure(figsize=(12, 4))

#     # ------------------------------------------------------------------
#     # Panel 1: PSF intensity
#     # ------------------------------------------------------------------
#     ax1 = plt.subplot(1, 3, 1)
#     im1 = ax1.imshow(
#         intensity_all[0],
#         origin="lower",
#         vmin=0,
#         vmax=intensity_vmax,
#     )
#     ax1.set_title("PSF intensity at HMSPL input")
#     cbar1 = plt.colorbar(im1, ax=ax1)

#     # ------------------------------------------------------------------
#     # Panel 2: PSF phase
#     # ------------------------------------------------------------------
#     ax2 = plt.subplot(1, 3, 2)
#     im2 = ax2.imshow(
#         np.angle(fields[0]),
#         origin="lower",
#         cmap="twilight",
#         vmin=-np.pi,
#         vmax=np.pi,
#     )
#     ax2.set_title("PSF phase")
#     cbar2 = plt.colorbar(im2, ax=ax2)

#     # ------------------------------------------------------------------
#     # Panel 3: LP modal powers
#     # ------------------------------------------------------------------
#     ax3 = plt.subplot(1, 3, 3)
#     x = np.arange(nmodes)
#     bars = ax3.bar(x, lp_powers[0])
#     ax3.set_xticks(x)
#     ax3.set_xticklabels(modelabels[:nmodes], rotation=90)
#     ax3.set_ylim(0, 1.05 * lp_vmax)
#     ax3.set_title("LP modal powers")

#     frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

#     plt.tight_layout()

#     def update(idx):
#         field = fields[idx]

#         # Update PSF intensity
#         im1.set_data(np.abs(field)**2)

#         # Update PSF phase
#         im2.set_data(np.angle(field))

#         # Update LP bar heights
#         for bar, height in zip(bars, lp_powers[idx]):
#             bar.set_height(height)

#         frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

#         return [im1, im2, frame_title, *bars]

#     anim = FuncAnimation(
#         fig,
#         update,
#         frames=n_sims,
#         interval=1000 / fps,
#         blit=False,
#     )

#     if save_video:
#         writer = PillowWriter(fps=fps)
#         # else:
#         #     writer = FFMpegWriter(fps=fps, bitrate=3000)

#         anim.save(outname, writer=writer, dpi=dpi)
#         plt.close(fig)

#         print(f"Saved video to {outname}")



##############################################################################
def make_n_distinct_colors(n, cmap="turbo"):
    """
    Return n visually distinct RGBA colours from a given colormap.

    Parameters
    ----------
    n : int
        Number of colours.
    cmap : str or Colormap
        Matplotlib colormap name or object (e.g. "viridis", "plasma", plt.cm.tab10).

    Returns
    -------
    colors : (n, 4) ndarray
        RGBA colours.
    """
    cmap_obj = plt.get_cmap(cmap)
    return cmap_obj(np.linspace(0, 1, n, endpoint=False))


##############################################################################
def plot_wf_psf_zernike_lp(
    results,
    idx=0,
    figsize=(12, 10),
    wf_key="pupil_wfs",
    zernike_key="zernike_coeffs",
    power_key="lp_powers",
    save_plot=False,
    fname_plot='wf_psf_zernike_lp_example.png'
):
    """
    2x2 layout with aligned columns and square panels.

        Top-left     : Pupil wavefront
        Top-right    : PSF intensity
        Bottom-left  : Zernike coefficient bar chart
        Bottom-right : LP modal power bar chart
    """

    field = np.asarray(results["fields"][idx])
    wf = np.asarray(results[wf_key][idx])
    z = np.asarray(results[zernike_key][idx])
    lp = np.asarray(results[power_key][idx])

    nmodes = int(results["nmodes"])
    modelabels = np.asarray(results["modelabels"])

    # Colour sets for bars
    z_colors = make_n_distinct_colors(len(z), cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 1],
    )

    # ==========================================================
    # Top-left : Pupil wavefront
    # ==========================================================
    ax00 = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])

    im1 = ax00.imshow(
        wf,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        aspect="equal",
    )
    ax00.set_title("Wavefront")
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.set_box_aspect(1)

    cb1 = fig.colorbar(im1, cax=cax00)
    cb1.ax.set_title("phase [rad]", fontsize=10, pad=8)

    # ==========================================================
    # Top-right : PSF intensity
    # ==========================================================
    ax01 = fig.add_subplot(gs[0, 2])
    cax01 = fig.add_subplot(gs[0, 3])

    im2 = ax01.imshow(
        np.abs(field)**2,
        origin="lower",
        aspect="equal",
    )
    ax01.set_title("PSF at HMSPL Input")
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.set_box_aspect(1)

    cb2 = fig.colorbar(im2, cax=cax01)
    cb2.ax.set_title("intensity", fontsize=10, pad=8)

    # ==========================================================
    # Bottom-left : Zernike coefficients
    # ==========================================================
    ax10 = fig.add_subplot(gs[1, 0])
    zx = np.arange(len(z))

    ax10.bar(zx, z, width=0.8, color=z_colors)
    ax10.set_title("Zernike Coefficients")
    ax10.set_xlabel("Mode Index")
    ax10.set_ylabel("Coefficient")
    ax10.set_xticks(zx)
    ax10.set_xticklabels(zx + 1, rotation=90)
    ax10.grid(":", linewidth=0.5, alpha=0.4)
    ax10.set_box_aspect(1)

    ax_blank1 = fig.add_subplot(gs[1, 1])
    ax_blank1.axis("off")

    # ==========================================================
    # Bottom-right : LP modal powers
    # ==========================================================
    ax11 = fig.add_subplot(gs[1, 2])
    x = np.arange(nmodes)

    ax11.bar(x, lp, width=0.8, color=lp_colors)
    ax11.set_title("LP Modal Powers")
    ax11.set_xlabel("LP Mode")
    ax11.set_ylabel("Coupled Power")
    ax11.set_xticks(x)
    ax11.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax11.grid(":", linewidth=0.5, alpha=0.4)
    ax11.set_box_aspect(1)

    ax_blank2 = fig.add_subplot(gs[1, 3])
    ax_blank2.axis("off")

    if save_plot:
        plt.savefig(fname_plot, dpi=150)

    plt.show()


##############################################################################
def make_wf_psf_video_square(
    results,
    outname="wf_psf_evolution.gif",
    save_video=False,
    fps=30,
    dpi=150,
    figsize=(12, 10),
    psf_key="psf_fields",
    wf_key="pupil_wfs",
    zernike_key="zernike_coeffs",
    power_key="lp_powers",
):
    """
    Create a video of the 2x2 evolution plot:

        Top-left     : Pupil wavefront
        Top-right    : PSF intensity
        Bottom-left  : Zernike coefficient bar chart
        Bottom-right : LP modal power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wfs = np.asarray(results[wf_key])
    zernikes = np.asarray(results[zernike_key])
    lp_powers = np.asarray(results[power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]

    intensity_all = np.abs(fields)**2

    # Fixed colour/axis limits so the movie does not flicker
    intensity_vmax = np.nanmax(intensity_all)
    wf_vmin = -np.pi
    wf_vmax = np.pi

    z_absmax = np.nanmax(np.abs(zernikes))
    lp_vmax = np.nanmax(lp_powers)

    z_colors = make_n_distinct_colors(zernikes.shape[1], cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 1],
    )

    # ==========================================================
    # Top-left : Pupil wavefront
    # ==========================================================
    ax00 = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])

    im_wf = ax00.imshow(
        pupil_wfs[0],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )
    ax00.set_title("Wavefront")
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.set_box_aspect(1)

    cb1 = fig.colorbar(im_wf, cax=cax00)
    cb1.ax.set_title("phase [rad]", fontsize=10, pad=8)

    # ==========================================================
    # Top-right : PSF intensity
    # ==========================================================
    ax01 = fig.add_subplot(gs[0, 2])
    cax01 = fig.add_subplot(gs[0, 3])

    im_psf = ax01.imshow(
        intensity_all[0],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )
    ax01.set_title("PSF at HMSPL Input")
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.set_box_aspect(1)

    cb2 = fig.colorbar(im_psf, cax=cax01)
    cb2.ax.set_title("intensity", fontsize=10, pad=8)

    # ==========================================================
    # Bottom-left : Zernike coefficients
    # ==========================================================
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.axhline(y=0, xmin=0, xmax=len(zernikes)+1,
                 color="k", 
                 linestyle="--", 
                 linewidth=0.5, 
                 alpha=0.7)
    zx = np.arange(zernikes.shape[1])

    z_bars = ax10.bar(
        zx,
        zernikes[0],
        width=0.8,
        color=z_colors,
    )
    ax10.set_title("Zernike Coefficients")
    ax10.set_xlabel("Mode Index")
    ax10.set_ylabel("Coefficient")
    ax10.set_xticks(zx)
    ax10.set_xticklabels(zx + 1, rotation=90)
    ax10.set_ylim(-1.05 * z_absmax, 1.05 * z_absmax)
    # ax10.grid(":", linewidth=0.5, alpha=0.4)
    ax10.set_box_aspect(1)

    ax_blank1 = fig.add_subplot(gs[1, 1])
    ax_blank1.axis("off")

    # ==========================================================
    # Bottom-right : LP modal powers
    # ==========================================================
    ax11 = fig.add_subplot(gs[1, 2])
    x = np.arange(nmodes)

    lp_bars = ax11.bar(
        x,
        lp_powers[0],
        width=0.8,
        color=lp_colors,
    )
    ax11.set_title("LP Modal Powers")
    ax11.set_xlabel("LP Mode")
    ax11.set_ylabel("Coupled Power")
    ax11.set_xticks(x)
    ax11.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax11.set_ylim(0, 1.05 * lp_vmax)
    # ax11.grid(":", linewidth=0.5, alpha=0.4)
    ax11.set_box_aspect(1)

    ax_blank2 = fig.add_subplot(gs[1, 3])
    ax_blank2.axis("off")

    frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

    def update(idx):
        # Update wavefront
        im_wf.set_data(pupil_wfs[idx])

        # Update PSF intensity
        im_psf.set_data(intensity_all[idx])

        # Update Zernike bar heights
        for bar, height in zip(z_bars, zernikes[idx]):
            bar.set_height(height)

        # Update LP bar heights
        for bar, height in zip(lp_bars, lp_powers[idx]):
            bar.set_height(height)

        frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

        return [im_wf, im_psf, frame_title, *z_bars, *lp_bars]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_sims,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        if outname.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
        # else:
        #     writer = FFMpegWriter(fps=fps, bitrate=3000)

        anim.save(outname, writer=writer, dpi=dpi)
        plt.close(fig)

        print(f"Saved video to {outname}")
    else:
        plt.show()

    return anim


##############################################################################
def make_wf_psf_lp_pl_video_row(
    results,
    outname="zernike_wf_psf_lp_pl_evolution.gif",
    save_video=True,
    fps=30,
    dpi=150,
    figsize=(22, 4.5),
    psf_key="psf_fields",
    wf_key="pupil_wfs",
    zernike_key="zernike_coeffs",
    lp_power_key="lp_powers",
    pl_power_key="pl_powers",
):
    """
    Create a 1x5 video:

        1. Zernike coefficient bar chart
        2. Pupil wavefront
        3. PSF intensity at HMSPL input
        4. LP modal power bar chart
        5. PL output core power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wfs = np.asarray(results[wf_key])
    zernikes = np.asarray(results[zernike_key])
    lp_powers = np.asarray(results[lp_power_key])
    pl_powers = np.asarray(results[pl_power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]
    n_zernikes = zernikes.shape[1]
    n_pl_cores = pl_powers.shape[1]

    intensity_all = np.abs(fields) ** 2

    # Fixed limits to avoid flickering
    z_absmax = np.nanmax(np.abs(zernikes))
    wf_vmin, wf_vmax = -np.pi, np.pi
    intensity_vmax = np.nanmax(intensity_all)
    lp_vmax = np.nanmax(lp_powers)
    pl_vmax = np.nanmax(pl_powers)

    z_colors = make_n_distinct_colors(n_zernikes, cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")
    pl_colors = make_n_distinct_colors(n_pl_cores, cmap="cividis")

    fig, axes = plt.subplots(
        1, 5,
        figsize=figsize,
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # 1. Zernike coefficients
    # ----------------------------------------------------------
    ax_z = axes[0]
    zx = np.arange(n_zernikes)

    z_bars = ax_z.bar(
        zx,
        zernikes[0],
        width=0.8,
        color=z_colors,
    )
    ax_z.set_title("Zernike Coefficients")
    ax_z.set_xlabel("Mode Index")
    ax_z.set_ylabel("Coefficient")
    ax_z.set_xticks(zx)
    ax_z.set_xticklabels(zx + 1, rotation=90)
    ax_z.set_ylim(-1.05 * z_absmax, 1.05 * z_absmax)
    # ax_z.grid(":", linewidth=0.5, alpha=0.4)
    ax_z.set_box_aspect(1)

    # ----------------------------------------------------------
    # 2. Pupil wavefront
    # ----------------------------------------------------------
    ax_wf = axes[1]

    im_wf = ax_wf.imshow(
        pupil_wfs[0],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )
    ax_wf.set_title("Wavefront")
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_wf.set_box_aspect(1)

    cb_wf = fig.colorbar(im_wf, ax=ax_wf, fraction=0.046, pad=0.04)
    cb_wf.ax.set_title("phase [rad]", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 3. PSF intensity
    # ----------------------------------------------------------
    ax_psf = axes[2]

    im_psf = ax_psf.imshow(
        intensity_all[0],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )
    ax_psf.set_title("PSF Intensity")
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    ax_psf.set_box_aspect(1)

    cb_psf = fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
    cb_psf.ax.set_title("intensity", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 4. LP modal powers
    # ----------------------------------------------------------
    ax_lp = axes[3]
    lx = np.arange(nmodes)

    lp_bars = ax_lp.bar(
        lx,
        lp_powers[0],
        width=0.8,
        color=lp_colors,
    )
    ax_lp.set_title("LP Modal Powers")
    ax_lp.set_xlabel("LP Mode")
    ax_lp.set_ylabel("Coupled Power")
    ax_lp.set_xticks(lx)
    ax_lp.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax_lp.set_ylim(0, 1.05 * lp_vmax)
    # ax_lp.grid(":", linewidth=0.5, alpha=0.4)
    ax_lp.set_box_aspect(1)

    # ----------------------------------------------------------
    # 5. PL output core powers
    # ----------------------------------------------------------
    ax_pl = axes[4]
    px = np.arange(n_pl_cores)

    pl_bars = ax_pl.bar(
        px,
        pl_powers[0],
        width=0.8,
        color=pl_colors,
    )
    ax_pl.set_title("PL Output Core Powers")
    ax_pl.set_xlabel("Core Index")
    ax_pl.set_ylabel("Power")
    ax_pl.set_xticks(px)
    ax_pl.set_xticklabels(px + 1)
    ax_pl.set_ylim(0, 1.05 * pl_vmax)
    # ax_pl.grid(":", linewidth=0.5, alpha=0.4)
    ax_pl.set_box_aspect(1)

    frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

    def update(idx):
        im_wf.set_data(pupil_wfs[idx])
        im_psf.set_data(intensity_all[idx])

        for bar, height in zip(z_bars, zernikes[idx]):
            bar.set_height(height)

        for bar, height in zip(lp_bars, lp_powers[idx]):
            bar.set_height(height)

        for bar, height in zip(pl_bars, pl_powers[idx]):
            bar.set_height(height)

        frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

        return [
            im_wf,
            im_psf,
            frame_title,
            *z_bars,
            *lp_bars,
            *pl_bars,
        ]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_sims,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        if outname.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)

        anim.save(outname, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Saved video to {outname}")
    else:
        plt.show()

    return anim