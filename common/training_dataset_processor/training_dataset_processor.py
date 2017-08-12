
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import glob
import numpy as np

import wavelet_core.isoPerceptron


file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, ".."))

plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']


# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'
base_dir = os.path.join(os.environ['HOME'], 'injection_resources')
plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# LOAD DATA FROM SIMULATION FILE
training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'


# open the file


def ProcessTrainingDataset(param):

    training_dataset_fullfilename = param['input_filename']

    training_dataset_filename = os.path.split(training_dataset_fullfilename)[-1]
    output_dir = param['output_dir']

    gap = param['gap']
    kS_list = param['kS_list']
    kB = param['kB']

    training_dataset_filename = os.path.split(training_dataset_fullfilename)[-1]

    training_dataset_filename_prefix = '__'.join(training_dataset_filename.split('__')[0:2])

    training_dataset_index = int(training_dataset_filename.split('__')[1])

    training_dataset = h5py.File(training_dataset_fullfilename, 'r')


    # Assemble the spectrum by adding the injection and background
    # DIMENSIONS: [# instances] x [# detectors] x [# samples in passby] x [# bins in spectrum]

    dimensions = training_dataset['injection_spectra'].shape

    # # number of samples (acquisitions) in the pass-by
    # number_instances = spectrum.shape[0]
    # # number of detectors
    # number_detectors = spectrum.shape[1]
    # # number of samples (acquisitions) in the pass-by
    # number_acquisitions = spectrum.shape[2]
    # number_spectral_bins = spectrum.shape[3]


    # number of samples (acquisitions) in the pass-by
    number_instances = dimensions[0]
    # number of detectors
    number_detectors = dimensions[1]
    # number of samples (acquisitions) in the pass-by
    number_acquisitions = dimensions[2]
    number_spectral_bins = dimensions[3]



    t_start_before_loop = time.time()

    for kS_index, kS in enumerate(kS_list):

        SNR_matrix = np.zeros((number_instances, number_detectors, number_acquisitions, 9228))
        SNR_background_matrix = np.zeros((number_instances, number_detectors, number_acquisitions, 9228))

        for instance_index in xrange(number_instances):

            f_snr = wavelet_core.isoPerceptron.isoSNRFeature(number_spectral_bins, kB, gap, kS)

            for detector_index in xrange(number_detectors):

                t_detector_loop = time.time()

                signal = training_dataset['injection_spectra'][instance_index, detector_index, :,:]
                background = training_dataset['background_matrix'][instance_index, detector_index, :,:]

                # versions where the whole array was already read into memory
                for acquisition_index in xrange(number_acquisitions):
                    SNR_matrix[instance_index, detector_index, acquisition_index,:] = f_snr.ingest(
                        (signal[acquisition_index,:]+background[acquisition_index,:]).astype(float))
                    # break

                for acquisition_index in xrange(number_acquisitions):
                    SNR_background_matrix[instance_index, detector_index, acquisition_index,:] = f_snr.ingest(
                        background[acquisition_index,:].astype(float))
                    # break


                # # versions where the whole array was already read into memory
                # for acquisition_index in xrange(number_acquisitions):
                #     SNR_matrix[instance_index, detector_index, acquisition_index,:] = f_snr.ingest(
                #         spectrum[instance_index, detector_index, acquisition_index,:].astype(float))
                #
                # for acquisition_index in xrange(number_acquisitions):
                #     SNR_background_matrix[instance_index, detector_index, acquisition_index,:] = f_snr.ingest(
                #         background[instance_index, detector_index, acquisition_index,:].astype(float))

                t_elapsed = time.time() - t_detector_loop

                if 'worker_no' in param:
                    print('worker_no: {}, instance_index: {}/{}, sample detector_index: {}/{}, t_elapsed: {} sec'.format(param['worker_no'], instance_index, number_instances, detector_index, number_detectors, t_elapsed))
                else:
                    print('instance_index: {}/{}, sample detector_index: {}/{}, t_elapsed: {} sec'.format(instance_index, number_instances, detector_index, number_detectors, t_elapsed))


        filename = training_dataset_filename.replace('TrainingDataset', 'kS_%02d__ProcessedDataset' %kS)
        output_fullfilename = os.path.join(output_dir, filename)

        with h5py.File(output_fullfilename, "w") as f:

            f.create_dataset('SNR_matrix', data=SNR_matrix, compression='gzip')
            f.create_dataset('SNR_background_matrix', data=SNR_background_matrix, compression='gzip')
            f.create_dataset('gap', data=gap)
            f.create_dataset('kS_list', data=kS_list)
            f.create_dataset('kB', data=kB)

            print("Saved: %s" % (output_fullfilename))
