#%%

import astropy.units as u
import astropy.constants as c
import numpy as np


from seidr.star_planet import Star, Planet
# from


#%%###########################################################################
### Main

if __name__ == "__main__":
    # p = Planet(
    #     "GJ 86 b",
    #     M_sin_i=4.27 * u.M_jup,
    #     semi_major_axis=0.1177 * u.au,
    #     density=1.64 * u.g / u.cm**3,
    # )

    # s = Star(
    #     "GJ 86",
    #     distance=10.9 * u.pc,
    #     effective_temp=5_350 * u.K,
    #     radius=0.855 * u.R_sun,
    #     mass=0.8 * u.M_sun,
    #     planets=[p],
    # )
    p = Planet(
        "HD 77946 b",
        M_sin_i=0.02637 * u.M_jup,
        semi_major_axis=0.072 * u.au,
        density=1.64 * u.g / u.cm**3,
    )

    s = Star(
        "HD 77946",
        distance=99 * u.pc,
        effective_temp=6_046 * u.K,
        radius=1.31 * u.R_sun,
        mass=1.17 * u.M_sun,
        planets=[p],
    )

    earth = Planet(
        "Earth",
        M_sin_i=1 * u.M_earth,
        semi_major_axis=1 * u.au,
        density=5.51 * u.g / u.cm**3,
    )

    sun = Star(
        "Sun",
        distance=0 * u.pc,
        effective_temp=5_780 * u.K,
        radius=1.0 * u.R_sun,
        mass=1.0 * u.M_sun,
        planets=[earth],
    )

    print([x.to(u.mas) for x in s.planet_angular_separation])

    print(s.planet_contrast_thermal(1.630 * u.micron))
    print(s.planet_contrast_thermal(3.450 * u.micron))
    print(s.planet_contrast_reflected(1.630 * u.micron, 0.1, 0.5))
    print(s.planet_contrast_reflected(3.450 * u.micron, 0.1, 0.5))

    print(s.planet_total_contrast(1.630 * u.micron, 0.1, 0.5))
    print(s.planet_total_contrast(3.450 * u.micron, 0.1, 0.5))
# %%
