"""
Tests out the SNR wavelet features.
Creates snr objects of various kS, kB, gap values and saves them to file (for use later).


"""
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import glob
import numpy as np

import cPickle

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print('Current path: {}'.format(file_directory))
sys.path.append(os.path.join(file_directory, ".."))


import wavelet_core.isoPerceptron

plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']


# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'

if 'INJECTION_RESOURCES' in os.environ:
    base_dir = os.environ['INJECTION_RESOURCES']
else:
    base_dir = os.path.join(os.environ['HOME'], 'injection_resources')
plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')
snr_root_path = os.path.join(base_dir, 'snr_functions')

if not os.path.exists(snr_root_path):
    os.mkdir(snr_root_path)

snr_path = os.path.join(snr_root_path, '20170824')
if not os.path.exists(snr_path):
    os.mkdir(snr_path)


if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# LOAD DATA FROM SIMULATION FILE
training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'

# get the sim file names
training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)

training_dataset_fullfilename_list = glob.glob(os.path.join(training_dataset_path, '*.h5'))
training_dataset_filename_list = [os.path.split(f)[-1] for f in training_dataset_fullfilename_list]

training_dataset_index_list = [int(f.split('__')[1]) for f in training_dataset_filename_list]

file_list_index = 1

# sim_dataset_filename = 'InjectionDataset__JobIndex_000__20170719_101101.h5'
training_dataset_filename = training_dataset_filename_list[file_list_index]
training_dataset_fullfilename = training_dataset_fullfilename_list[file_list_index]
training_dataset_index = training_dataset_index_list[file_list_index]

training_dataset_filename_prefix ='__'.join( training_dataset_filename.split('__')[0:2] )

training_dataset = h5py.File(training_dataset_fullfilename, 'r')

# Assemble the spectrum by adding the injection and background
# DIMENSIONS: [# instances] x [# detectors] x [# samples in passby] x [# bins in spectrum]
signal = training_dataset['injection_spectra'].value
background = training_dataset['background_matrix'].value
spectrum = signal+background

# number of samples (acquisitions) in the pass-by
number_samples = spectrum.shape[2]

# let's focus on a single detector
detector_index = 0

# create instance of Dan's wavelet snr generator

number_bins = 1024
number_bins = 512

# this defines the grid
kS_list = [1, 2, 4, 8]
# gap_list = [2, 4, 8]
gap_list = [16,]
kB_list = [16, 32]

if number_bins == 512:
    number_wavelet_bins = 4107
elif number_bins == 1024:
    number_wavelet_bins = 9228

number_samples_skip = 20
number_samples_save = 50


SNR_matrix = np.zeros((number_wavelet_bins,number_samples_save))
SNR_background_matrix = np.zeros((number_wavelet_bins,number_samples))


# parameter_string = 'kB: %d, gap: %d, kS: %d' % (kB, gap, kS)
# plot_name_suffix = 'kB_%d__gap_%d__kS_%d' % (kB, gap, kS)
#
# # plot of total counts vs sample
# plt.figure()
# plt.plot(signal[training_index, detector_index, :, :].sum(1))
# plt.plot(background[training_index, detector_index, :, :].sum(1))
# plt.plot(spectrum[training_index, detector_index, :, :].sum(1))
#
# plt.title('Counts Signal')
# plt.xlabel('Energy Bin', fontsize=16)
# plt.ylabel('Count', fontsize=16)
#
# plt.savefig(os.path.join(plot_dir, '%s__total_counts__%s.png' % (training_dataset_filename_prefix, plot_name_suffix)))
# plt.close()
#
# # plot of spectrum at peak
# plt.figure()
# index_max_counts = np.argmax(spectrum[training_index, detector_index, :, :].sum(1))
#
# plt.plot(spectrum[training_index, detector_index, index_max_counts, :])
# plt.plot(spectrum[training_index, detector_index, 0, :])
#
# plt.savefig(os.path.join(plot_dir, '%s__spectra__%s.png' % (training_dataset_filename_prefix, plot_name_suffix)))
# plt.close()
#
# # plot of 2d spetra Signal
# plt.figure()
# plt.pcolor(spectrum[training_index, detector_index, :, 1:].T)
#
# plt.title('Counts Signal, %s' % parameter_string)
# plt.xlabel('Acquisition', fontsize=16)
# plt.ylabel('Energy Bin', fontsize=16)
# plt.colorbar()
#
# plt.savefig(
#     os.path.join(plot_dir, '%s__Signal_Background__%s.png' % (training_dataset_filename_prefix, plot_name_suffix)))
# plt.close()

create_plots = 1

training_index = 0

wavelet_index = 530

for kS in kS_list:

    for kB in kB_list:

        for gap in gap_list:

            parameter_string = 'kB: %d, gap: %d, kS: %d' %(kB, gap, kS)

            plot_name_suffix = 'kS_%02d__kB_%02d__gap_%02d' %(kS, kB, gap)

            print(plot_name_suffix)

            f_snr = wavelet_core.isoPerceptron.isoSNRFeature(number_bins, kB, gap, kS)


            # loop over all acquisitions of the pass-by
            t0 = time.time()

            for sample_index in xrange(number_samples):
                if sample_index % 20 == 0:
                    print('sample index {}/{}'.format(sample_index, number_samples))
                SNR_background_matrix[:,sample_index] = f_snr.ingest(background[training_index,detector_index, sample_index,:].astype(float))

            # save the snr function
            with open(os.path.join(snr_path, 'f_snr__%s.pkl' %plot_name_suffix), 'wb') as fid:
                cPickle.dump(f_snr, fid, 2)

            print('Saved:{}'.format(os.path.join(snr_path, 'f_snr__%s.pkl' %plot_name_suffix)))

#             for sample_index in xrange(number_samples_save):
#
#                 acquisition_index = number_samples_skip + sample_index
#                 if sample_index%10 == 0:
#                     print('sample index {}/{}'.format(sample_index, number_samples))
#                 SNR_matrix[:,sample_index] = f_snr.ingest(spectrum[training_index,detector_index, acquisition_index,:].astype(float))
#             print('time elapsed: %3.3f' %(time.time() - t0))
#
#             if create_plots:
#
#                 # plot of 2d spetra background
#                 plt.figure()
#                 plt.pcolor(background[training_index,detector_index, :,1:].T)
#
#                 plt.title('Counts Background, %s' %parameter_string)
#                 plt.xlabel('Acquisition', fontsize  = 16)
#                 plt.ylabel('Energy Bin', fontsize  = 16)
#                 plt.colorbar()
#
#                 plt.savefig(os.path.join(plot_dir, '%s__Background__%s.png' %(training_dataset_filename_prefix, plot_name_suffix)))
#                 plt.close()
#
#                 plt.figure()
#                 plt.pcolor(SNR_matrix[:,:])
#
#                 plt.title('SNR Signal + Background, %s' %parameter_string)
#                 plt.xlabel('Acquisition', fontsize  = 16)
#                 plt.ylabel('SNR Bin', fontsize  = 16)
#                 plt.colorbar()
#
#                 plt.savefig(os.path.join(plot_dir, '%s__SNR_Signal_Background__%s.png' %(training_dataset_filename_prefix, plot_name_suffix)))
#                 plt.close()
#
#
#                 # SNR background
#
#                 plt.figure()
#                 plt.pcolor(SNR_background_matrix[:,:])
#
#                 plt.title('SNR Background, %s' %parameter_string)
#                 plt.xlabel('Acquisition', fontsize  = 16)
#                 plt.ylabel('SNR Bin', fontsize  = 16)
#                 plt.colorbar()
#
#                 plt.savefig(
#                     os.path.join(plot_dir, '%s__SNR_Background__%s.png' % (training_dataset_filename_prefix, plot_name_suffix)))
#                 plt.close()
#
#                 plt.figure()
#                 acquisition_indices = np.arange(SNR_background_matrix.shape[1])
#                 plt.plot(acquisition_indices, SNR_background_matrix[wavelet_index,:])
#
#                 acquisition_indices = np.arange(number_samples_skip, number_samples_skip+number_samples_save)
#                 plt.plot(acquisition_indices, SNR_matrix[wavelet_index,:])
#                 plt.title('SNR, %s' %parameter_string)
#                 plt.xlabel('Acquisition', fontsize  = 16)
#
#                 plt.savefig(
#                     os.path.join(plot_dir, '%s__SNR__waveletindex_%04d__%s.png' % (training_dataset_filename_prefix, wavelet_index, plot_name_suffix)))
#                 plt.close()


# training_index_list = [0, 3, 5, 7, 9]
#
#
# for kS_index, kS in enumerate(kS_list):
#
#     for kB_index, kB in enumerate(kB_list):
#
#         for gap_index, gap in enumerate(gap_list):
#
#             # calculate the SNR for this pass-by instance
#             for training_index_index, training_index in enumerate(training_index_list):
#
#                 parameter_string = 'kB: %d, gap: %d, kS: %d' %(kB, gap, kS)
#
#                 snr_name_suffix = 'kS_%02d__kB_%02d__gap_%02d' %(kS, kB, gap)
#
#                 # f_snr = wavelet_core.isoPerceptron.isoSNRFeature(number_bins, kB, gap, kS)
#
#                 # save the snr
#                 with open(os.path.join(snr_path, 'f_snr__%s.pkl' %plot_name_suffix), 'rb') as fid:
#                     f_snr = cPickle.load(fid)
#
#                 # loop over all acquisitions of the pass-by
#                 t0 = time.time()
#
#                 # for sample_index in xrange(number_samples):
#                 #     if sample_index % 10 == 0:
#                 #         print('sample index {}/{}'.format(sample_index, number_samples))
#                 #     SNR_background_matrix[:,sample_index] = f_snr.ingest(background[training_index,detector_index, sample_index,:].astype(float))
#
#                 # for sample_index in xrange(number_samples):
#                 #     if sample_index%10 == 0:
#                 #         print('sample index {}/{}'.format(sample_index, number_samples))
#                 #     SNR_matrix[:,sample_index] = f_snr.ingest(spectrum[training_index,detector_index, sample_index,:].astype(float))
#
#                 for sample_index in xrange(number_samples_save):
#
#                     acquisition_index = number_samples_skip + sample_index
#                     if sample_index % 10 == 0:
#                         print('sample index {}/{}'.format(sample_index, number_samples))
#                     SNR_matrix[:, sample_index] = f_snr.ingest(
#                         spectrum[training_index, detector_index, acquisition_index, :].astype(float))
#                 print('time elapsed: %3.3f' % (time.time() - t0))
#
#                 plot_name_suffix = 'instance_%05d__kS_%02d__kB_%02d__gap_%02d' %(kS, kB, gap, training_index)
#
#                 # plot of 2d spectra background
#
#                 if (kS_index == 0) and (kB_index == 0) and (gap_index == 0):
#                     plt.figure()
#                     plt.pcolor(background[training_index,detector_index, :,1:].T)
#
#                     plt.title('Counts Background, %s' %parameter_string)
#                     plt.xlabel('Acquisition', fontsize  = 16)
#                     plt.ylabel('Energy Bin', fontsize  = 16)
#                     plt.colorbar()
#
#                     plt.savefig(os.path.join(plot_dir, '%s__Background__%s.png' %(training_dataset_filename_prefix, plot_name_suffix)))
#                     plt.close()
#
#
#                 plt.figure()
#                 plt.pcolor(SNR_matrix[:,:])
#
#                 plt.title('SNR Signal + Background, %s' %parameter_string)
#                 plt.xlabel('Acquisition', fontsize  = 16)
#                 plt.ylabel('SNR Bin', fontsize  = 16)
#                 plt.colorbar()
#
#                 plt.savefig(os.path.join(plot_dir, '%s__SNR_Signal_Background__%s.png' %(training_dataset_filename_prefix, plot_name_suffix)))
#                 plt.close()
#
#
#                 # # SNR background
#                 #
#                 # plt.figure()
#                 # plt.pcolor(SNR_background_matrix[:,:])
#                 #
#                 # plt.title('SNR Background, %s' %parameter_string)
#                 # plt.xlabel('Acquisition', fontsize  = 16)
#                 # plt.ylabel('SNR Bin', fontsize  = 16)
#                 # plt.colorbar()
#                 #
#                 # plt.savefig(
#                 #     os.path.join(plot_dir, '%s__SNR_Background__%s.png' % (training_dataset_filename_prefix, plot_name_suffix)))
#                 # plt.close()
#
#
#                 plt.figure()
#                 acquisition_indices = np.arange(SNR_background_matrix.shape[1])
#                 plt.plot(acquisition_indices, SNR_background_matrix[wavelet_index,:])
#
#                 acquisition_indices = np.arange(number_samples_skip, number_samples_skip+number_samples_save)
#                 plt.plot(acquisition_indices, SNR_matrix[wavelet_index,:])
#                 plt.title('SNR, %s' %parameter_string)
#                 plt.xlabel('Acquisition', fontsize  = 16)
#
#                 plt.savefig(
#                     os.path.join(plot_dir, '%s__SNR__waveletindex_%04d__%s.png' % (training_dataset_filename_prefix, wavelet_index, plot_name_suffix)))
#                 plt.close()
