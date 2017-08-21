
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import glob
import numpy as np

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, ".."))

import wavelet_core.isoPerceptron


plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']


# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'
# base_dir = os.path.join(os.environ['HOME'], 'injection_resources')
# plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
# training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
# processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')
#
# if not os.path.exists(plot_dir):
#     os.mkdir(plot_dir)
#
# # LOAD DATA FROM SIMULATION FILE
# training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'


# open the file


def ProcessTrainingDataset(param):

    training_dataset_fullfilename = param['input_filename']

    training_dataset_filename = os.path.split(training_dataset_fullfilename)[-1]
    output_dir = param['output_dir']

    gap = param['gap']
    kS = param['kS']
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

    filename = training_dataset_filename.replace('TrainingDataset', 'kS_%02d__ProcessedDataset' % kS)
    output_fullfilename = os.path.join(output_dir, filename)

    with h5py.File(output_fullfilename, "w") as f:

        # SNR_matrix = f.create_dataset('SNR_matrix', shape=(number_instances, number_detectors, number_acquisitions, 9228), compression='gzip')
        # SNR_background_matrix = f.create_dataset('SNR_background_matrix', shape=(number_instances, number_detectors, number_acquisitions, 9228), compression='gzip')
        SNR_matrix = f.create_dataset('SNR_matrix', shape=(number_instances, number_detectors, number_acquisitions, 9228))
        SNR_background_matrix = f.create_dataset('SNR_background_matrix', shape=(number_instances, number_detectors, number_acquisitions, 9228))

        f.create_dataset('gap', data=gap)
        f.create_dataset('kS', data= kS)
        f.create_dataset('kB', data= kB)

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

    print("Saved: %s" % (output_fullfilename))





def CalculateTargetValues(filename_input, filename_output):
    """Calculate target values"""

    training_dataset = h5py.File(filename_input, 'r')


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

    number_sources = len(set(training_dataset['source_index']))

    t_start_before_loop = time.time()


    with h5py.File(filename_output, "w") as f:

        signal = training_dataset['injection_spectra'].value
        background = training_dataset['background_matrix'].value

        # signal counts

        # collapse in energy bins
        # dimensions: number_instances x number_detectors x number_acquisitions
        f.create_dataset('signal_total_counts', data = signal.sum(3))
        f.create_dataset('signal_total_counts_no_first_bin', data = signal[:,:,:,1:].sum(3))

        # collapse in energy bins and detector index
        # dimensions: number_instances x number_acquisitions
        f.create_dataset('signal_total_counts_all_detectors', data = signal.sum(3).sum(1))
        f.create_dataset('signal_total_counts_no_first_bin_all_detectors', data = signal[:,:,:,1:].sum(3).sum(1))

        # background

        # collapse in energy bins
        f.create_dataset('background_total_counts', data = signal.sum(3))
        f.create_dataset('background_total_counts_no_first_bin', data = background[:,:,:,1:].sum(3))

        # collapse in energy bins and detector index
        f.create_dataset('background_total_counts_all_detectors', data = background.sum(3).sum(1))
        f.create_dataset('background_total_counts_no_first_bin_all_detectors', data = background[:,:,:,1:].sum(3).sum(1))


        f.create_dataset('background_mean_counts', data = background.sum(3).mean(2))
        f.create_dataset('background_mean_counts_no_first_bin', data = background[:,:,:,1:].sum(3).mean(2))

        f.create_dataset('background_std_total_counts', data = background.sum(3).std(2))
        f.create_dataset('background_std_counts_no_first_bin', data = background[:,:,:,1:].sum(3).std(2))


        # signal to noise ratio
        # dimensions: number_instances x number_detectors x number_acquisitions
        f.create_dataset('signal_noise', data = signal.sum(3) / np.transpose(np.tile(background.sum(3).std(2), (background.shape[2], 1, 1) ), [1, 2, 0] ))
        # f.create_dataset('signal_noise_no_first_bin', data = signal[:,:,:,1:].sum(3) / np.transpose(np.tile(background[:,:,:,1:].sum(3).std(2), (background.shape[2], 1, 1, 1) ), [1, 2, 3, 0] ))
        f.create_dataset('signal_noise_no_first_bin', data = signal[:,:,:,1:].sum(3) / np.transpose(np.tile(background[:,:,:,1:].sum(3).std(2), (background.shape[2], 1, 1) ), [1, 2, 0] ))

        f.create_dataset('source_index', data = training_dataset['source_index'])
        f.create_dataset('source_name_list', data = training_dataset['source_name_list'])

        # Calcuate the main target array - total counts for each isotope
        # dimensions: number_instances x number_sources x number_acquisitions
        #
        source_index = training_dataset['source_index'].value
        # signal_total_counts = signal.sum(3)
        signal_total_counts_all_detectors = signal.sum(3).sum(1)
        temp = np.zeros((number_instances, number_sources, number_acquisitions))

        for index in xrange(len(source_index)):


            temp[index, int(source_index[index]), :] = signal_total_counts_all_detectors[index,:]

        f.create_dataset('source_signal_total_counts_all_detectors_matrix', data = temp)

        # pass on the geometry values
        source_detector_distance = f.create_dataset('source_detector_distance', shape = (number_instances, number_acquisitions))
        source_detector_angle = f.create_dataset('source_detector_angle', shape = (number_instances, number_acquisitions))
        detector_x = f.create_dataset('detector_x', shape = (number_instances, number_acquisitions))

        # load the arrays so that it doesn't have to load at each iteration
        speed_index = training_dataset['speed_index'].value
        print(speed_index)
        speed_index = speed_index.astype(int)

        distance_array = training_dataset['distance_array'].value
        angle_array = training_dataset['angle_array'].value
        x_array = training_dataset['x_array'].value

        for index in xrange(number_instances):
            sub_index = speed_index[index]
            source_detector_distance[index,:] = distance_array[sub_index,:]
            source_detector_angle[index,:] = angle_array[sub_index,:]
            detector_x[index,:] = x_array[sub_index,:]

    print("Saved: %s" % (filename_output))
