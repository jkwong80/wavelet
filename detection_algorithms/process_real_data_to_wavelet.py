"""


"""
import os, sys, glob, time
import h5py, cPickle
import copy
import numpy as np

# assumes that you are at the base of the repo
sys.path.append('common')

def ProcessRebinReadingsFile(input_fullfilename, feature_selection_fullfilename, mask_filtered_features_name, snr_fullfilename, detector_name, output_fullfilename, run_background_first = False, background_window = None):
    # arguments
    # number_bins
    # filename
    #
    # not arguments
    # feature_selection_file - construct from kS, kB, gap??

    print('Loadings snr: {}'.format(snr_fullfilename))
    with open(snr_fullfilename, 'rb') as fid:
        f_snr = cPickle.load(fid)

    with h5py.File(input_fullfilename, 'r') as dat:
        try:
            if 'rebinned_spectra' in dat[detector_name]:
                spectra = dat[detector_name]['rebinned_spectra'].value
                t = dat[detector_name]['timestamp_us'].value
            else:
                spectra = dat[detector_name]['spectra'].value
                t = dat[detector_name]['time'].value
        except:
            print
            spectra = dat[detector_name]['adc_channel_counts'].value

    with h5py.File(feature_selection_fullfilename, 'r') as f:
        mask_filtered_features = f[mask_filtered_features_name].value
    print('Keeping {} features'.format(sum(mask_filtered_features)))
    number_samples_save = np.sum(mask_filtered_features)

    if run_background_first:
        for background_sample_index in xrange(background_window[0], background_window[1]):
            f_snr.ingest(spectra[background_sample_index, :].astype(float))

    with h5py.File(output_fullfilename, 'w') as f:
        # no compression as this slows things down quite a bit

        f.create_dataset('t', data = t)
        SNR_filtered_matrix = f.create_dataset('SNR_filtered', shape = (spectra.shape[0], number_samples_save))

        # get the signal and background portions

        for sample_index in xrange(SNR_filtered_matrix.shape[0]):

            if sample_index % 100 == 0:
                print('sample index {}/{}'.format(sample_index, SNR_filtered_matrix.shape[0]))
            SNR_filtered_matrix[sample_index, :] = f_snr.ingest(spectra[sample_index,:].astype(float))[mask_filtered_features]

if __name__ == '__main__':
    ProcessRebinReadingsFile(sys.argv[0], sys.argv[1], sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5], sys.argv[6])


