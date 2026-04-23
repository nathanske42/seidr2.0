#%%
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from seidr.SeidrSim import SeidrSim


#%%
def generate_lp_with_zernikes(
    n_sims=100,
    seed=0,
    wavel=1.55,              # microns
    n_core=1.4440,
    n_cladding=1.4400,
    core_diameter=18.0,      # microns, MM entrance diameter
    max_r=3.0,
    wf_npixels=512,
    psf_npixels=256,
    n_zernikes=30,
    f_number=4.5,
    tiptilt_rms=50e-9,
    ho_rms=20e-9,
    return_fields=True,
    save_data=False,
):
    """
    Generate n_sims simulated PSFs at the HMSPL input using SeidrSim with
    random Zernike aberrations, then decompose each one into LP mode coefficients.

    Parameters
    ----------
    n_sims : int
        Number of simulations.
    seed : int
        Random seed.
    wavel : float
        Wavelength in microns.
    n_core, n_cladding : float
        Fiber refractive indices.
    core_diameter : float
        Diameter of the multimode entrance core in microns.
    max_r : float
        Half-width of simulation grid in units of core radius.
    wf_npixels : int
        Number of pupil-plane pixels for dLux optics.
    psf_npixels : int
        Number of pixels across the PSF grid.
    n_zernikes : int
        Number of Zernike modes.
    f_number : float
        Optical f-number.
    tiptilt_rms : float
        RMS coefficient scale for tip/tilt Zernikes.
    ho_rms : float
        RMS coefficient scale for higher-order Zernikes.
    return_fields : bool
        If True, also store the complex PSF fields.

    Returns
    -------
    results : dict
        Dictionary containing modal powers, coefficients, and metadata.
    """

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
    )

    key = jr.PRNGKey(seed)

    all_coeffs = []
    all_powers = []
    all_total_coupling = []
    all_zernikes = []
    all_fields = [] if return_fields else None

    mode_numbers = list(range(len(sim.lf.allmodefields_rsoftorder)))

    for i in range(n_sims):
        key, k1, k2 = jr.split(key, 3)

        # Zernike vector:
        # index 0 = piston (kept at zero here)
        # indices 1,2 = tip/tilt
        # remaining = higher-order aberrations
        z = jnp.concatenate([
            jnp.zeros((1,)),
            tiptilt_rms * jr.normal(k1, (2,)),
            ho_rms * jr.normal(k2, (n_zernikes - 3,))
        ])

        # Apply aberration to the optical system
        sim.optics = sim.optics.set("aperture.coefficients", z)

        # Complex PSF / input field at the HMSPL entrance
        field = sim.propagate_wf()

        # Decompose into LP modes
        total_coupling, modal_powers, modal_coeffs = sim.lf.calc_injection_multi(
            input_field=field,
            mode_field_numbers=mode_numbers,
            show_plots=False,
            return_abspower=True,
            complex=True,
        )

        all_zernikes.append(np.array(z))
        all_total_coupling.append(np.array(total_coupling))
        all_powers.append(np.array(modal_powers))
        all_coeffs.append(np.array(modal_coeffs))

        if return_fields:
            all_fields.append(np.array(field))

    # Clean up by removing aberrations
    sim.remove_aberrations()

    results = {
        "total_coupling": np.array(all_total_coupling),   # shape (n_sims,)
        "powers": np.array(all_powers),                   # shape (n_sims, nmodes)
        "coeffs": np.array(all_coeffs),                   # shape (n_sims, nmodes), complex
        "zernikes": np.array(all_zernikes),               # shape (n_sims, n_zernikes)
        "modelabels": np.array(sim.lf.modelabels, dtype=object),
        "nmodes": sim.lf.nmodes,
    }

    if return_fields:
        results["fields"] = np.array(all_fields)          # shape (n_sims, Ny, Nx), complex

    return sim, results

