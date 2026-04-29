#%%###########################################################################
"""
Propagate PSFs through photonic lantern using pre-saved transfer matrices
"""

#%%########################################################################
### Import Libraries and Modules ###
import numpy as np
import matplotlib.pyplot as plt
import datetime

from seidr.seidr_functions_misc import load_lb_transfer_matrix


#%%########################################################################
### Filenames ###

outname_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")


#%%########################################################################
### Load Transfer Dataset ###

# total_coupling=results["total_coupling"],
# powers=results["lp_powers"],
# coeffs=results["lp_coeffs"],
# zernike_coeffs=results["zernike_coeffs"],
# modelabels=results["modelabels"],
# psf_fields=results["psf_fields"],
# pupil_wfs=results["pupil_wfs"],

## Filename of the dataset to load
dir_data = "/import/roci1/nlon0790/Results/psf_prop/"
f_data = dir_data + "seidr_wf_psf_lp_dataset_test.npz" # + outname_datetime + ".npz"

## Dataset
results = np.load(f_data)
print("Loaded dataset from ", f_data)
print("Dataset keys: ", results.keys())
# print("Dataset shape: ", results[list(results.keys())[0]].shape)

# powers=results["lp_powers"]
lp_coeffs = results["lp_coeffs"]
print("lp_coeffs shape: ", lp_coeffs.shape)

n_sims = lp_coeffs.shape[0]
print("Number of simulations in dataset: ", n_sims)

#%%########################################################################
### Load Transfer Matrix ###

## Transfer Matrix Parameters
# wl=1.51_rms=0.5589999999999999_ds=0.25_dz=2_rv=1_xyw=135.0_zlen=50000_tr=6.25.h5

# dir_tf = "/import/roci1/nlon0790/Results/"
f_pl_name = "hms-pl7c"
# f_pl_path = dir_tf + f_pl_name + "/cores/"

f_pl_path_temp = "/suphys/nlon0790/Documents/python_code/seidr2.0/data/transfer_matrices/"

wavel = 1.55 # wavelenth [um]
r_ms = 0.559 # radius of mode selective core [um]
ds = 0.25 # x-y step size for LB sims [um]
dz = 2 # z-step size for LB sims [um]
rv = 1 # reference value for LB sims
xywidth = 135.0 # width of the simulation window [um]
z_len = 50000 # length of PL [um]
tr = 6.25 # taper ratio

# fname_tf = f_pl_path_temp + f"wl={wavel}_rms={r_ms}_ds={ds}_dz={dz}_rv={rv}_xyw={xywidth}_zlen={z_len}_tr={tr}.h5"

transfer_matrix = load_lb_transfer_matrix(f_pl_path_temp, f_pl_name, 
                                     wavel, r_ms, ds, 
                                     dz, rv, xywidth, 
                                     z_len, tr)

transfer_matrix = transfer_matrix.T
print("Transfer matrix shape: ", transfer_matrix.shape) # (n_cores, n_modes)

n_cores, n_modes = transfer_matrix.shape
print(f"Number of cores: {n_cores}, Number of modes: {n_modes}")

#%%########################################################################
### Propagate PSFs through PL - example ###)

pl_outputs_array = np.zeros((n_sims, n_cores), dtype=complex)

for i in range(n_sims):
    lp_coeffs_i = lp_coeffs[i, :]
    pl_outputs_array[i, :] = transfer_matrix @ lp_coeffs_i

# # Predicted complex output amplitudes at each SM waveguide
# pl_outputs_array = transfer_matrix @ lp_coeffs

# Measurable output: intensity at each SM waveguide
pl_output_intensity_array = np.abs(pl_outputs_array) ** 2

print("Output intensity shape:", pl_output_intensity_array.shape)


#%%########################################################################
### Test: Plot output intensities for one time step ###

idx_example = np.random.randint(0, n_sims)
print(f"Plotting output intensities for simulation index {idx_example}")

plt.figure(figsize=(8, 4))
plt.bar(range(n_cores), pl_output_intensity_array[idx_example, :])
plt.xlabel('Core Index')
plt.ylabel('Output Intensity')
plt.title('Propagation of PSF through PL - Example Simulation')
plt.grid(':', linewidth=0.5, alpha=0.5)
plt.show()

#%%########################################################################
### Test: Plot output intensities for a few example simulations ###

n_examples = 20
plt.figure(figsize=(12, 6))
for i in range(n_examples):
    plt.plot(pl_output_intensity_array[i, :], '-o', label=f't={i+1}')
plt.xlabel('Core Index')
plt.ylabel('Output Intensity')
plt.title('Propagation of PSFs through PL')
plt.legend(ncol=n_examples//2, fontsize=8)
plt.grid(':', linewidth=0.5, alpha=0.5)
plt.show()
# %%
