import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from seidr.SeidrSim import SeidrSim


def generate_lp_from_zernikes(
    n_sims=100,
    seed=0,
    wavel=1.55,
    f_number=4.5,
    pupil_diameter=1.8,
    n_core=1.4440,
    n_cladding=1.4400,
    core_diameter=18.0,
    max_r=3.0,
    wf_npixels=512,
    psf_npixels=256,
    n_zernikes=30,
    tiptilt_rms=50e-9,
    ho_rms=20e-9,
    return_fields=True,
):
    """
    Generate point-source PSFs using SeidrSim, distorted by random Zernike
    wavefront aberrations, then decompose the focal-plane field into LP modes.

    Returns
    -------
    sim : SeidrSim
        Configured simulation object.
    results : dict
        Dataset containing PSFs, Zernikes, LP powers, and LP complex coefficients.
    """

    ##########################################################################
    ### Build SeidrSim object
    sim = SeidrSim(
        wavel=wavel,
        n_core=n_core,
        n_cladding=n_cladding,
        core_diameter=core_diameter,
        max_r=max_r,
        wf_npixels=wf_npixels,
        psf_npixels=psf_npixels,
        n_zernikes=n_zernikes,
        f_number=f_number,
        pupil_diameter=pupil_diameter,
    )

    ##########################################################################
    ### Setup

    # LP modes created internally by sim.lf
    mode_numbers = list(range(len(sim.lf.allmodefields_rsoftorder)))

    ## Set up random-number generator
    key = jr.PRNGKey(seed)

    ## Initialize lists to store results
    all_zernikes = []
    all_total_coupling = []
    all_lp_powers = []
    all_lp_coeffs = []
    all_fields = [] if return_fields else None


    ##########################################################################
    ### Main simulation loop
    for i in range(n_sims):

        # split into three random numbers
        key, key_tiptilt, key_ho = jr.split(key, 3)

        ### Zernike coefficient vector -> ADD RANDOM WALK FROM BN
        # z[0]      = piston, set to zero
        # z[1:3]    = tip/tilt
        # z[3:]     = higher-order aberrations
        zernike_coeffs = jnp.concatenate([
            jnp.zeros((1,)),
            tiptilt_rms * jr.normal(key_tiptilt, (2,)),
            ho_rms * jr.normal(key_ho, (n_zernikes - 3,)),
        ])

        ### Apply Zernike aberration to pupil-plane wavefront
        sim.optics = sim.optics.set(
            "aperture.coefficients",
            zernike_coeffs,
        )

        ### Propagate point source through optics to focal plane
        # Returns the complex focal-plane field:
        # E(x, y) = A(x, y) exp(i phi(x, y))
        field = sim.propagate_wf()

        ### Decompose focal-plane field into LP modes
        total_coupling, lp_powers, lp_coeffs = sim.lf.calc_injection_multi(
            input_field=field,
            mode_field_numbers=mode_numbers,
            show_plots=False,
            return_abspower=True,
            complex=True,
        )

        ## Convert to NumPy arrays and store result
        all_zernikes.append(np.array(zernike_coeffs))
        all_total_coupling.append(np.array(total_coupling))
        all_lp_powers.append(np.array(lp_powers))
        all_lp_coeffs.append(np.array(lp_coeffs))

        if return_fields:
            all_fields.append(np.array(field))

    ## Reset to base, unaberrated state
    sim.remove_aberrations()

    ## Combine outputs ** ADD wavefronts! **
    results = {
        "zernikes": np.array(all_zernikes),
        "total_coupling": np.array(all_total_coupling),
        "lp_powers": np.array(all_lp_powers),
        "lp_coeffs": np.array(all_lp_coeffs),
        "modelabels": np.array(sim.lf.modelabels, dtype=object),
        "nmodes": sim.lf.nmodes,
        "microns_per_pixel": sim.lf.microns_per_pixel,
    }

    if return_fields:
        results["fields"] = np.array(all_fields)

    return sim, results