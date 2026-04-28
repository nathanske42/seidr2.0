#%%########################################################################
### Import Libraries and Modules ###
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.io import savemat
plt.ion()

from seidr.seidr_functions_misc import make_smoothrand_multi,\
      zernike_rms_per_mode #make_smoothrand, 

#%%###########################################################################
### Define Mode Orders ###

# Zernike modes
zernike_mode_labels = {
    # 0th Radial
    1: "Piston",
    # 1st Radial
    2: "Tilt X",
    3: "Tilt Y",
    # Second Radial
    4: "Defocus",
    5: "Astig X",
    6: "Astig Y",
    # Third Radial
    7: "Coma X",
    8: "Coma Y",
    9: "Trefoil X",
    10: "Trefoil Y",
    # Fourth Radial
    11: "Spherical",
    12: "2nd Astig X",
    13: "2nd Astig Y",
    14: "Quadrafoil X",
    15: "Quadrafoil Y",
    # Fifth Radial
    16: "2nd Coma X",
    17: "2nd Coma Y",
    18: "2nd Trefoil X",
    19: "2nd Trefoil Y",
    20: "Pentafoil X",
    21: "Pentafoil Y",
    # Sixth Radial
    22: "2nd Spherical",
    23: "3rd Coma X",
    24: "3rd Coma Y",
    25: "3rd Astig X",
    26: "3rd Astig Y",
    27: "Hexafoil X",
    28: "Hexafoil Y",
    # Seventh Radial
    29: "4th Coma X",
    30: "4th Coma Y",
    31: "4th Astig X",
    32: "4th Astig Y",
    33: "3rd Trefoil X",
    34: "3rd Trefoil Y",
    35: "Heptafoil X",
    36: "Heptafoil Y",
}

#%%########################################################################
### Set Filenames ###

### Directory and Save Parameters 
datadir = './data/'
savefilename = None
# savefilename = 'contcoeffs_20230217-001_20modes_rms0.4-0.05_smth7'


#%%########################################################################
### Set Parameters ###

n_steps = 10000
n_zernikes = 20 #10

# define the Gaussian kernel samples / time steps
smooth_amt = 7

# Vary RMS per zernike mode, with a linear drop-off from start to end mode
start_rms_perterm = 0.4 #0.3
end_rms_perterm = 0.05

rms_perterm_multi = zernike_rms_per_mode(start_rms_perterm, end_rms_perterm, 
                                         n_zernikes)

all_coeffs = make_smoothrand_multi(n_steps, n_zernikes, 
                                   finalsds=rms_perterm_multi, 
                                   smthamt=smooth_amt)

#%%########################################################################
### Plots ###
plt.figure(2)
plt.clf()
plt.plot(rms_perterm_multi, '-x')
plt.ylim([0, start_rms_perterm])
print(rms_perterm_multi[9]/start_rms_perterm)
print(rms_perterm_multi[19]/start_rms_perterm)

#%%
plt.figure(1)
plt.clf()
plt.plot(all_coeffs[:100,:], '-o',markersize=2)

plt.xlabel('Simulation Step')
plt.ylabel('Coefficient Value')

plt.ylim([-1, 1])

plt.grid(':', linewidth=0.5, alpha=0.5)

plt.legend(['%s' % (zernike_mode_labels[k]) for k in range(1, n_zernikes+1)],
           loc='best', 
           fontsize=8,
           ncol=4)

plt.show()


#%%########################################################################
# rms_perterm = 0.555 # 0.555 = 0.88 rad av. RMS WF for 19 terms
# # rms_perterm = 0.95 # 0.95 = 1.5 rad av. RMS WF for 19 terms
# rms_perterm = 0.735 # 0.735 = 0.88 rad av. RMS WF for 9 terms
# rms_perterm = 1.25 # 0.735 = 1.5 rad av. RMS WF for 9 terms

# rms_perterm = 0.15
# rms_perterm = 0.3


## RMS per term with power law drop-off
# exp = -0.5
# x=np.arange(1,ncoeffs+1).astype('float32')
# scl = x**exp
# rms_perterm_multi = start_rms_perterm * scl


# all_coeffs = make_smoothrand(nsteps, ncoeffs, finalsd=rms_perterm, smthamt=smthamt)

# rms_perterm_multi = np.linspace(start_rms_perterm, end_rms_perterm, 
#                                 n_zernikes-1)
# rms_perterm_multi = np.concatenate(([0], rms_perterm_multi))

# rms_perterm_multi[1] = rms_perterm_multi[2] # make tip / tilt the same

# tt = (rms_perterm_multi[0] + rms_perterm_multi[1])/2
# rms_perterm_multi[0] = tt
# rms_perterm_multi[1] = tt



#%%########################################################################

if savefilename is not None:
    print('Saving to ' + savefilename)
    np.savez(datadir + savefilename + '.npz', 
             all_coeffs=all_coeffs, 
             start_rms_perterm=start_rms_perterm,
             end_rms_perterm=end_rms_perterm, 
             smthamt=smthamt)
        # np.savez(datadir+savefilename+'.npz', 
    #          all_coeffs=all_coeffs, 
    #          rms_perterm=rms_perterm, 
    #          smthamt=smthamt)


#%%########################################################################
### Define Functions ###

def make_smoothrand(nsteps, nvecs=1, smthamt=10., 
                    finalsd=1.):

    smthrand_all = np.zeros((nsteps, nvecs))

    for k in range(nvecs):
        noisevec = np.random.randn(nsteps)
        smthrand = ndimage.gaussian_filter1d(noisevec, smthamt)
        smthrand = smthrand / np.std(smthrand) * finalsd
        smthrand_all[:,k] = smthrand

    return smthrand_all

def make_smoothrand_multi(nsteps, nvecs=1, smthamt=10., 
                          finalsds=1.):
    
    smthrand_all = np.zeros((nsteps, nvecs))

    for k in range(nvecs):
        noisevec = np.random.randn(nsteps)
        smthrand = ndimage.gaussian_filter1d(noisevec, smthamt)
        smthrand = smthrand / np.std(smthrand) * finalsds[k]
        smthrand_all[:,k] = smthrand
        
    return smthrand_all

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

# # Check normalisation
# ncoeffs=19
# for k in range(ncoeffs):
#     coeff_vec = np.zeros(ncoeffs)
#     coeff_vec[k] = 1
#
#     coeff_vec_col = coeff_vec.reshape(1,-1)
#     coeff_vec_col = norm_coeffs(coeff_vec_col)
#     coeff_vec = coeff_vec_col.reshape(-1)
#
#     im = make_zernim(coeff_vec)
#     print('Min: %.3f, Max: %.3f' % (np.nanmin(im), np.nanmax(im)))
#     plt.pause(0.2)



#%%########################################################################








# all_phasemaps = np.zeros((nsteps, sz, sz))
# all_wfrms = np.zeros(nsteps)
#
# print('Using [-1,1] normalisation for phasemap generation')
# all_coeffs_normd = norm_coeffs(all_coeffs)
# for k in range(nsteps):
#     # coeffs = all_coeffs[k, :]
#     coeffs = all_coeffs_normd[k, :]
#     coeff_vec = np.zeros(cart.nk)
#     coeff_vec[:ncoeffs] = coeffs
#     phasemap = cart.eval_grid(coeff_vec, matrix=True)
#     all_phasemaps[k,:,:] = phasemap
#     all_wfrms[k] = np.nanstd(phasemap)
#     # if k % 100 == 0:
#     #     print(k)
#     # plt.clf()
#     # plt.imshow(all_phasemaps[k,:,:], clim=[-clim,clim])
#     # plt.colorbar()
#     # plt.pause(0.01)
#



