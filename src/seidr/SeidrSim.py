#%%

import jax
import jax.numpy as np
import jax.random as jr

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import dLux as dl
import dLux.utils as dlu

import copy

from seidr.lanternfiber import lanternfiber

#%%
class SeidrSim:
    def __init__(
        self,
        wavel,
        n_core,
        n_cladding,
        core_diameter,
        max_r=3,
        wf_npixels=512,
        psf_npixels=512,
        n_zernikes=100,
        f_number=4.5,
        pupil_diameter=1.8,
    ) -> None:
        """
        A simulation of the SEIDR instrument

        Parameters
        ----------
        wavel : float
            The wavelength of the light in microns
        n_core : float
            The refractive index of the core
        n_cladding : float
            The refractive index of the cladding
        core_diameter : float
            The diameter of the core in microns
        max_r : float, optional
            The maximum radius of the fiber, by default 6
        wf_npixels : int, optional
            The number of pixels in the wavefront, by default 512
        psf_npixels : int, optional
            The number of pixels in the PSF, by default 512
        n_zernikes : int, optional
            The number of Zernike modes to use, by default 5.
            1 is piston, 2 is tip/tilt, 3 is defocus, etc.
        f_number : float, optional
            The f number of the system, by default 4.5
        """

        self.wavel = wavel
        self.n_core = n_core
        self.n_cladding = n_cladding
        self.core_diameter = core_diameter
        self.max_r = max_r
        self.pupil_diameter = pupil_diameter

        self.n_zernikes = n_zernikes

        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.f_number = f_number

        self.lf = lanternfiber(
            n_core=n_core,
            n_cladding=n_cladding,
            core_radius=core_diameter / 2,
            wavelength=wavel,
        )
        self.lf.find_fiber_modes()
        self.lf.make_fiber_modes(npix=self.psf_npixels // 2, 
                                 show_plots=False, max_r=max_r)

        self._optics = self._make_optics()


    ##########################################################################
    def _make_optics(self):
        # # Wavefront properties
        # diameter = self.pupil_diameter  # [m]
        # wf_npixels = 512

        # psf params
        focal_length = self.f_number * self.pupil_diameter
        psf_pixel_scale = self.max_r * self.core_diameter / self.psf_npixels

        coords = dlu.pixel_coords(self.wf_npixels, self.pupil_diameter)
        circle = dlu.circle(coords, self.pupil_diameter / 2)

        # Zernike aberrations
        zernike_indexes = np.arange(1, self.n_zernikes + 1)
        coeffs = np.zeros(zernike_indexes.shape)
        coords = dlu.pixel_coords(self.wf_npixels, self.pupil_diameter)
        basis = dlu.zernike_basis(zernike_indexes, coords, self.pupil_diameter)

        ## Store Zernike seeing information -> added NKL
        self.pupil_coords = coords
        self.pupil_mask = circle
        self.zernike_indexes = zernike_indexes
        self.zernike_basis = basis

        layers = [
            ("aperture", dl.layers.BasisOptic(basis, circle, coeffs, 
                                              normalise=True))
        ]

        ## Construct Optics
        self.optics = dl.CartesianOpticalSystem(
            self.wf_npixels, self.pupil_diameter, layers, 
            focal_length, self.psf_npixels, psf_pixel_scale
        )

        self.source = dl.PointSource(flux=1.0, 
                                     wavelengths=[self.wavel * 1e-6])


    ##########################################################################
    def propagate_wf(self):

        output = self.source.model(self.optics, return_wf=True)
        ouput_wf_complex = (
            (output.amplitude * np.exp(1j * output.phase))
            * self.source.spectrum.weights[:, None, None]
        ).sum(axis=0)

        return ouput_wf_complex


    ##########################################################################
    def remove_aberrations(self):
        self.optics = self.optics.set(
            "aperture.coefficients", np.zeros(self.n_zernikes)
        )


    ##########################################################################
    def propagate_injections(self, is_complex=False):
        """
        Given the current state of the system, propagate the wavefront and 
        calculate the injection efficiency
        """
        wf = self.propagate_wf()

        return self.lf.calc_injection_multi(
            input_field=wf,
            mode_field_numbers=list(range(
                len(self.lf.allmodefields_rsoftorder))),
            show_plots=False,
            return_abspower=True,
            complex=is_complex,
        )[0:2]


    ##########################################################################
    def make_pupil_wavefront(self, zernike_coeffs, return_phase=True):
        """
        Reconstruct the pupil-plane wavefront from Zernike coefficients.

        Parameters
        ----------
        zernike_coeffs : array
            Zernike coefficients, same ordering used by the aperture BasisOptic.

        return_phase : bool
            If True, return phase in radians.
            If False, return OPD / wavefront error in metres.

        Returns
        -------
        pupil_wf : array
            Pupil-plane wavefront map.
        """

        opd = np.sum(
            self.zernike_basis * zernike_coeffs[:, None, None],
            axis=0,
        )

        opd = np.where(self.pupil_mask, opd, np.nan)

        if return_phase:
            wavelength_m = self.wavel * 1e-6
            phase = 2 * np.pi * opd / wavelength_m
            return phase

        return opd


    ##########################################################################
    def make_aberrations_gif(self, zernike_coeffs, fname):
        
        n_frames = zernike_coeffs.shape[0]

        Figure = plt.figure(figsize=(8, 4))

        def set_zern_and_prop_wf(z_coeffs):
            self.optics = self.optics.set("aperture.coefficients", z_coeffs)
            return self.propagate_wf()

        non_aberrated = set_zern_and_prop_wf(
            np.zeros(zernike_coeffs.shape[1]))

        wavefronts = jax.vmap(set_zern_and_prop_wf)(zernike_coeffs)
        self.remove_aberrations()

        # add circle to show the PL input
        circle = plt.Circle(
            (non_aberrated.shape[0] // 2, non_aberrated.shape[1] // 2),
            self.optics.psf_npixels // (self.max_r * 2),
            fill=False,
            linestyle="--",
            color="w",
        )

        plt.subplot(1, 2, 1)
        amp_img = plt.imshow(np.abs(non_aberrated), cmap="inferno")
        # mark centre with a little cross
        plt.plot(
            non_aberrated.shape[0] // 2, non_aberrated.shape[1] // 2, 
            "+", color="r"
        )
        plt.colorbar()
        plt.title("Amplitude")
        plt.gca().add_artist(copy.copy(circle))

        plt.subplot(1, 2, 2)
        phase_img = plt.imshow(np.angle(non_aberrated), cmap="twilight")
        plt.plot(
            non_aberrated.shape[0] // 2, non_aberrated.shape[1] // 2, 
            "+", color="r"
        )
        plt.colorbar()
        plt.title("Phase")
        plt.gca().add_artist(copy.copy(circle))

        plt.savefig(fname + ".png")

        def animate(
            frame_idx,
        ):
            amp_img.set_data(np.abs(wavefronts[frame_idx]))
            phase_img.set_data(np.angle(wavefronts[frame_idx]))

            return amp_img, phase_img

        anim_created = FuncAnimation(Figure, animate, frames=n_frames)

        anim_created.save(fname + ".gif", fps=15)


    ##########################################################################
    @staticmethod
    def make_default(type="smf", **kwargs):
        n_core = 1.44
        n_cladding = 1.4345

        if type == "smf":
            core_diameter = 8.2
        elif type == "mmf5":
            core_diameter = 15.9
        else:
            raise NotImplementedError()

        return SeidrSim(
            wavel=1.63,
            n_core=n_core,
            n_cladding=n_cladding,
            core_diameter=core_diameter,
            **kwargs
        )


#%%###########################################################################
if __name__ == "__main__":
    sim = SeidrSim.make_default()
    n_zernikes = sim.n_zernikes

    sim.remove_aberrations()

    print(sim.propagate_injections())

    # n_runs = 100

    # tip_tilt_rms = 200e-9 / 4 / np.sqrt(2)
    # rest_rms = 20e-9 / 4

    # zernike_coeffs = np.concatenate(
    #     [
    #         np.zeros((n_runs, 1)),
    #         tip_tilt_rms * jr.normal(jr.PRNGKey(1), (n_runs, 2)),
    #         rest_rms * jr.normal(jr.PRNGKey(1), (n_runs, n_zernikes - 3)),
    #     ],
    #     axis=1,
    # )

    # sim.make_aberrations_gif(zernike_coeffs, "test")

# %%
