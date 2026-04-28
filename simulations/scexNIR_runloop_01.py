#%%########################################################################
### Import Libraries and Modules ###
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
plt.ion()
import time
import os
from pyMilk.interfacing.isio_shmlib import SHM as shm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi


#%%########################################################################
### Set Parameters ###

datapath = './data/'
# model_filename = 'modelsave_20220217-001_10modes_5XradScale0.7_n100000-001'
model_filename = 'modelsave_20220217-001_10modes_5Xrad_n100000-001'

nmodes = 9
nwgs = 19

dm_modes_filename = 'DMzernikecube_46px_n1000.npy'
dm_modes_trimpist = True # Remove 0th mode from dm_modes cube

wl = 1.6 # microns

testmode = False
testmode_plflux_filename = 'PLrespdata_20220202-013_10modes-withpist_5Xrad_n100000'

play_wfe = True
wfecube_scale = 0.7 #1.0
trim_zerowfe = 100
# wfecube_filename = 'wfcube_20220203-001_contcoeffs_10modes-0.15rad_n100000'
wfecube_filename = 'wfcube_20220203-001_contcoeffs_10modes-0.3rad_n100000'
wfecube_filename = 'wfcube_contcoeffs_20230217-001_20modes_rms0.4-0.05_smth7_n10000'
wfe_coeffs_trimpist = True

n_its = 10000 #1000
start_closeloop = 100 #100
openloop_inds = None
openloop_inds = np.concatenate([np.arange(0,200), np.arange(2000,3000), np.arange(6000,7000)])

save_filename = None
# save_filename = 'closeloopdata_testsave'
# save_filename = 'closeloopdata_20220217-001_contcoeffs_10modes-0.3rad-010'
save_filename = 'closeloopdata_20220217-001_contcoeffs_10modes_in20modes_rms0.4-0.05_scl0.7-002'
save_ircam = True
save_dmmaps = True

gain = 0.6 #0.5
leak = 0.02 #0.02

#%%########################################################################
### Instantiate SHM objects
dm_wfe = shm('dm00disp04')
dm_corr = shm('dm00disp05')
plfluxes = shm('aol6_imWFS0')


#%%########################################################################
### Load Model and Data ###

# Load model
model = keras.models.load_model(datapath+model_filename+'.h5')
model_md_file = np.load(datapath+model_filename+'.npz')

normfacts = model_md_file['normfacts']
dm_modes = np.load(datapath + dm_modes_filename)

if dm_modes_trimpist:
    dm_modes = dm_modes[:, :, 1:]

dm_modes = dm_modes[:, :, :nmodes]
size_dm = dm_modes.shape[0]
dm_modes_vec = dm_modes.reshape(size_dm**2, -1)
dm_modes_vec = dm_modes_vec / np.pi * wl / 2 / 2  # /2 since DM in reflection

all_rmse = np.zeros(n_its)
all_predcoeffs = np.zeros((nmodes, n_its))
all_measPLfluxes = np.zeros((nwgs, n_its))

if save_ircam:
    ircam = shm('ircam0')
    ircam_im = ircam.get_data()
    all_ircam = np.zeros((ircam_im.shape[0], ircam_im.shape[1], n_its), dtype='float32')
else:
    all_ircam = None

if save_dmmaps:
    all_dm_wfe = np.zeros((size_dm, size_dm, n_its), dtype='float32')
    all_dm_corr = np.zeros((size_dm, size_dm, n_its), dtype='float32')
else:
    all_dm_wfe = None
    all_dm_corr = None

cur_corrmap = np.zeros((size_dm,size_dm)).astype('float32')
dm_corr.set_data(cur_corrmap)


#%%########################################################################
### Play Back WFE ###

# Load cube of WFE to play back
if play_wfe:
    print('Loading WFE cube file ' + wfecube_filename)
    wfefile = np.load(datapath+wfecube_filename+'.npz')
    wfe_cube = wfefile['all_dmmaps']
    wfe_coeffs = wfefile['all_coeffs_rad']
    if trim_zerowfe is not None:
        wfe_cube = wfe_cube[trim_zerowfe:, :, :]
        wfe_coeffs = wfe_coeffs[trim_zerowfe:, :]
    if wfe_coeffs_trimpist:
        wfe_coeffs = wfe_coeffs[:,1:]
    wfe_cube *= wfecube_scale

if testmode:
    print('Test mode - loading PL fluxes from ' + testmode_plflux_filename)
    plflux_file = np.load(datapath + testmode_plflux_filename + '.npz', allow_pickle=True)
    all_plfluxes_test = plflux_file['all_plfluxes'].T
    if trim_zerowfe is not None:
        all_plfluxes_test = all_plfluxes_test[trim_zerowfe:, :]

def plot_wf(true_wf, pred_wf):
    # plt.figure(1, figsize=(10,3))
    plt.figure(1, figsize=(4, 9))
    plt.clf()
    vmax = np.max(true_wf)
    vmin = np.min(true_wf)
    plt.subplot(311)
    plt.imshow(true_wf, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.title('True WF')
    plt.subplot(312)
    plt.imshow(pred_wf, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.title('Pred WF')
    plt.subplot(313)
    plt.imshow(true_wf-pred_wf, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.title('Residual WF')
    plt.tight_layout()
    plt.pause(0.001)


#%%########################################################################
###
for kk in range(n_its):
    if kk % 100 == 0:
        print('Iteration %d' % kk)

    # Apply WFE to DM
    if play_wfe & (not testmode):
        dm_wfe.set_data(wfe_cube[kk, :, :])
    cur_wfe_coeffs = wfe_coeffs[kk,:]

    # Get current WFS data
    if not testmode:
        for l in range(4):
            pl_flux = plfluxes.get_data(check=True).reshape(1,-1)
    else:
        pl_flux = all_plfluxes_test[kk, :].reshape(1,-1)
    pl_flux -= normfacts[0]
    pl_flux /= normfacts[1]
    pl_flux -= normfacts[2]
    all_measPLfluxes[:,kk] = pl_flux

    # Predict WF
    pred_coeffs = model.predict(pl_flux, verbose=0).reshape(-1,1)
    pred_wf = (dm_modes_vec @ pred_coeffs).reshape(size_dm, size_dm)
    all_predcoeffs[:,kk] = np.squeeze(pred_coeffs)

    if save_ircam:
        all_ircam[:,:,kk] = ircam.get_data()

    if testmode:
        all_rmse[kk] = np.std(pred_wf-wfe_cube[kk, :, :])

    # plot_wf(wfe_cube[kk, :, :], pred_wf)
    # plt.pause(0.2)

    # Do correction
    if openloop_inds is not None:
        if kk not in openloop_inds:
            cur_corrmap = ((1-leak)*cur_corrmap - gain*pred_wf).astype('float32')
            dm_corr.set_data(cur_corrmap)
        else:
            dm_corr.set_data(np.zeros((size_dm, size_dm), dtype='float32'))
    else:
        if kk >= start_closeloop:
            cur_corrmap = ((1-leak)*cur_corrmap - gain*pred_wf).astype('float32')
            dm_corr.set_data(cur_corrmap)

    if save_dmmaps:
        all_dm_wfe[:,:,kk] = wfe_cube[kk, :, :]
        all_dm_corr[:,:,kk] = cur_corrmap


#%%########################################################################

if testmode:
    print('Average RMS reconstruction error (DM): %f' % np.mean(all_rmse))

if save_filename:
    print('Saving data to ' + save_filename)
    np.savez(datapath+save_filename+'.npz', all_predcoeffs=all_predcoeffs, all_ircam=all_ircam,
             all_measPLfluxes=all_measPLfluxes, model_filename=model_filename,
             wfecube_filename=wfecube_filename, gain=gain, leak=leak, all_dm_wfe=all_dm_wfe,
             all_dm_corr=all_dm_corr)

print('Done.')