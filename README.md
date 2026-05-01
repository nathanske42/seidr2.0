## SEIDR
----------------------

A revamped repo for Seidr - an instrument to perform photonic lantern fed kernel nulling at the VLTI.

Original repository by Adam Taras can be found at ```bash https://github.com/ataras2/Seidr ```

**very early developmental stage


### Development installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/SAIL-Labs/seidr2.0.git
cd seidr2.0
pip install -e .
```

seidr2.0 uses the fiber-imaging, which should also be installed:

```bash
git clone https://github.com/SAIL-Labs/fiber-imaging.git
cd fiber-imaging
pip install .
```

### Structure

```
seidr2.0
│   README.md
│
└───src/seidr/
│       __init__.py
│       source2pl.py              # end-to-end: wavefront → PSF → LP modes → PL outputs
│       SeidrSim.py               # dLux-based optical propagation (legacy)
│       seidr_functions_misc.py   # helper functions (smoothed random WFE, plotting, video)
│       star_planet.py            # star/planet modelling (draft)
│
└───simulations/
│       main_wf_psf_pl_prop.py    # temporal WF → PSF → LP → PL dataset generation
│
└───examples/
│       example_psf_pl_prop.py
│       example_kernel_nulling.py
│       example_star_planet.py
│       example_make_cont_coeffs.py
│
```


### TO DO

* Continue sifting through Seidr1.0 to extract useful code
* Map extrasolar planet (++) to kernel nulling chip performance


### Citation

If you found this code useful, please cite the following paper:

```bibtex
@inproceedings{taras2024kernel,
  title={Kernel nulling at VLTI with photonic lanterns for optimal fibre injection},
  author={Taras, Adam K and Norris, Barnaby and Chhabra, Sorabh and Cvetojevic, Nick and Foriel, Vincent and Ireland, Michael and Kraus, Stefan and Leon-Saval, Sergio and Martinache, Frantz and Paul, Jyotirmay and Spaldin, Eckhart and Sweeney, David and Tuthill, Peter},
  booktitle={Optical and Infrared Interferometry and Imaging IX},
  volume={13095},
  pages={242--250},
  year={2024},
  organization={SPIE}
}
```

