
from __future__ import print_function
import os, sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import numpy as np
import cPickle

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

    number_samples_save = param['number_samples_save']
    number_samples_skip = param['number_samples_skip']

    param['snr_fullfilename']

    number_bins = param['number_bins']


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

    filename = training_dataset_filename.replace('TrainingDataset', 'kS_%02d__kB_%02d__gap_%02d__ProcessedDataset' %(kS, kB, gap))
    output_fullfilename = os.path.join(output_dir, filename)

    with h5py.File(output_fullfilename, "w") as f:

        # SNR_matrix = f.create_dataset('SNR_matrix', shape=(number_instances, number_detectors, number_acquisitions, 9228), compression='gzip')
        # SNR_background_matrix = f.create_dataset('SNR_background_matrix', shape=(number_instances, number_detectors, number_acquisitions, 9228), compression='gzip')

        if number_bins == 512:
            number_wavelet_bins = 4107
        elif number_bins == 1024:
            number_wavelet_bins = 9228

        # no compression as this slows things down quite a bit
        SNR_matrix = f.create_dataset('SNR_matrix', shape=(number_instances, number_detectors, number_samples_save, number_wavelet_bins))

        # can skip depending on what the user wants
        if param['run_snr_background']:
            SNR_background_matrix = f.create_dataset('SNR_background_matrix', shape=(number_instances, number_detectors, number_acquisitions, number_wavelet_bins))

        f.create_dataset('gap', data = gap)
        f.create_dataset('kS', data = kS)
        f.create_dataset('kB', data = kB)
        f.create_dataset('snr_fullfilename', data = param['snr_fullfilename'])
        f.create_dataset('number_samples_save', data = param['number_samples_save'])
        f.create_dataset('number_samples_skip', data = param['number_samples_skip'])
        f.create_dataset('number_bins', data = param['number_bins'])

        for instance_index in xrange(number_instances):

            if 'snr_fullfilename' in param:
                print('Loadings snr: {}'.format(param['snr_fullfilename']))
                with open(param['snr_fullfilename'], 'rb') as fid:
                    f_snr = cPickle.load(fid)
            else:
                f_snr = wavelet_core.isoPerceptron.isoSNRFeature(number_spectral_bins, kB, gap, kS)

            for detector_index in xrange(number_detectors):

                t_detector_loop = time.time()

                signal = training_dataset['injection_spectra'][instance_index, detector_index, :,:]
                background = training_dataset['background_matrix'][instance_index, detector_index, :,:]

                if param['run_snr_background']:
                    for acquisition_index in xrange(number_acquisitions):
                        SNR_background_matrix[instance_index, detector_index, acquisition_index,:] = f_snr.ingest(
                            background[acquisition_index,:].astype(float))

                # # versions where the whole array was already read into memory
                # for acquisition_index in xrange(number_acquisitions):
                #     SNR_matrix[instance_index, detector_index, acquisition_index,:] = f_snr.ingest(
                #         (signal[acquisition_index,:]+background[acquisition_index,:]).astype(float))

                for sample_index in xrange(number_samples_save):
                    # Get the index in the spectra array
                    acquisition_index = number_samples_skip + sample_index
                    if sample_index % 20 == 0:
                        print('sample index {}/{}'.format(sample_index, number_samples_save))

                    SNR_matrix[instance_index, detector_index, sample_index,:] = f_snr.ingest(
                        (signal[acquisition_index,:]+background[acquisition_index,:]).astype(float))

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
        f.create_dataset('background_total_counts', data = background.sum(3))
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

        source_name_list = training_dataset['source_name_list']


        isotope_string_list = []

        for source_name in source_name_list:
            index_underscore = source_name.find('_')
            isotope_string_list.append(source_name[:index_underscore])

        isotope_string_list = np.array(isotope_string_list)
        isotope_string_set = np.array(list(set(isotope_string_list)))
        isotope_string_set.sort()

        print('Number of isotopes: {}'.format(len(isotope_string_set)))

        # map from isotope string to y value
        isotope_mapping = {}
        for isotope_string_index, isotope_string in enumerate(isotope_string_list):
            isotope_mapping[isotope_string_index] = np.where(isotope_string_set == isotope_string)[0][0] + 1
        # this the
        y_new = np.zeros(y.shape[0])

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


# def CreateFilteredFeaturesFile(processed_dataset_fullfilename, target_values_fullfilename, filtered_features_dataset_fullfilename, best_features_fullfilename,\
#                                number_acquisitions_save, acquisitions_skip, detector_index):
def CreateFilteredFeaturesFile(param):

    """

    :param processed_dataset_fullfilename:
    :param target_values_fullfilename:
    :param filtered_features_dataset_fullfilename:
    :param best_features_fullfilename:
    :param number_acquisitions_save: number of acquisitions to save
    :param acquisitions_skip: number of acquisitions to stkip
    :param detector_index: detector index
    :return:
    """
    # open the top features file
    processed_dataset_fullfilename = param['processed_dataset_fullfilename']
    target_values_fullfilename = param['target_values_fullfilename']
    filtered_features_dataset_fullfilename = param['filtered_features_dataset_fullfilename']
    best_features_fullfilename = param['best_features_fullfilename']
    number_acquisitions_save = param['number_acquisitions_save']
    acquisitions_skip = param['acquisitions_skip']
    detector_index = param['detector_index']

    mask_filtered_features = param['mask_filtered_features']

    with h5py.File(best_features_fullfilename, 'r') as f:
        mask_filtered_features = f[mask_filtered_features].value
    print('Keeping {} features'.format(sum(mask_filtered_features)))

    # open the wavelet file
    processed_dataset = h5py.File(processed_dataset_fullfilename, 'r')

    # get the dimensions of the matrices
    dimensions = processed_dataset['SNR_matrix'].shape
    # number of samples (acquisitions) in the pass-by
    number_instances = dimensions[0]
    # number of detectors
    number_detectors = dimensions[1]
    # number of samples (acquisitions) in the pass-by
    number_acquisitions = dimensions[2]
    # number of wavelet bins
    number_wavelet_bins = dimensions[3]

    number_samples_save = processed_dataset['number_samples_save'].value
    number_samples_skip = processed_dataset['number_samples_skip'].value

    # focus on a single detector

    # Note that this has already been truncated - it will be truncated some more by
    # acquisitions_skip to (acquisitions_skip+number_acquisitions_save)
    SNR_matrix = processed_dataset['SNR_matrix'].value

    target_values = h5py.File(target_values_fullfilename, 'r')

    # this has not been truncate so need to truncated it here
    # after the truncation here (as it is loaded), it should be on equal footing as SNR_matrix
    source_signal_total_counts_all_detectors_matrix =\
        target_values['source_signal_total_counts_all_detectors_matrix'][:,:,number_samples_skip:(number_samples_skip+number_samples_save)]

    source_index = target_values['source_index']
    source_name_list = target_values['source_name_list']

    number_sources = len(source_name_list)

    # the number of training instances is the number of drive by instances (number_instances) times the number of acquisitions to save per drive-by
    source_signal_matrix_all = np.zeros((number_instances * number_acquisitions_save, source_signal_total_counts_all_detectors_matrix.shape[1]))

    # reshaping the matrix
    SNR_matrix_all = np.zeros((number_instances * number_acquisitions_save, number_wavelet_bins))

    for instance_index in xrange(number_instances):
        # if instance_index % 10 == 0:
        #     print('{}/{}'.format(instance_index, number_instances))
        # indices in the drive by space
        start0 = acquisitions_skip
        stop0 = acquisitions_skip + number_acquisitions_save

        # indices in the training instance space
        start = instance_index * number_acquisitions_save
        stop = (instance_index + 1) * number_acquisitions_save

        # remove the features later
        SNR_matrix_all[start:stop,:] = SNR_matrix[instance_index,detector_index,start0:stop0,:]

        source_signal_matrix_all[start:stop,:] = source_signal_total_counts_all_detectors_matrix[instance_index, :, start0:stop0].T

    X = SNR_matrix_all[:,mask_filtered_features]
    y = source_signal_matrix_all[:, :]


    # taken from train_classifiers.py
    # Calculate projection to isotope axes (y is to source axes)

    # create the mapping (dict)
    isotope_string_list = []
    for source_name in source_name_list:
        index_underscore = source_name.find('_')
        isotope_string_list.append(source_name[:index_underscore])

    isotope_string_list = np.array(isotope_string_list)
    isotope_string_set = np.array(list(set(isotope_string_list)))
    isotope_string_set.sort()

    print('Number of isotopes: {}'.format(len(isotope_string_set)))

    # map from isotope string to y value
    # in the train_classifiers.py script I have the isotpe index start at 1 because the 0 is background - I don't do that
    # here; first isotope is 0
    isotope_mapping = {}
    for isotope_string_index, isotope_string in enumerate(isotope_string_list):
        isotope_mapping[isotope_string_index] = np.where(isotope_string_set == isotope_string)[0][0]
    # this the isotope index
    y_isotope = np.zeros(y.shape[0]).astype(np.int16)
    # this a 2d matrix [instance, isotope id] of the isotope counts.  should only be one non-zero value per row (as
    # we are simulating single source values)
    y_isotope_count = np.zeros((y.shape[0], len(isotope_string_set) ))

    count_threshold = 50

    # count_threshold_fraction_max_counts = 0.10

    for instance_index in xrange(y.shape[0]):
        if y[instance_index, :].sum() > count_threshold:
            sub_index = isotope_mapping[np.argmax(y[instance_index, :])]
            y_isotope[instance_index] = sub_index
            y_isotope_count[instance_index, sub_index] = y[instance_index, :].sum()


    with h5py.File(filtered_features_dataset_fullfilename, 'w') as f:
        f.create_dataset('y', data=y, compression = 'gzip')
        f.create_dataset('X', data=X, compression = 'gzip')
        f.create_dataset('y_isotope', data=y_isotope, compression = 'gzip')
        f.create_dataset('y_isotope_count', data=y_isotope_count, compression = 'gzip')

        f.create_dataset('mask_filtered_features', data=mask_filtered_features, compression = 'gzip')

    print('Wrote: {}'.format(filtered_features_dataset_fullfilename))


def ConsolidateFilteredFeatturesFiles(fullfilename_list, output_filename):

    # ############## concatenate separate files  #######################

    number_files = len(fullfilename_list)
    print("number_files: {}".format(number_files))

    for dataset_index, fullfilename in enumerate(fullfilename_list):

        with h5py.File(fullfilename, 'r') as f:

            if dataset_index == 0:
                X_dimensions = f['X'].shape
                y_dimensions = f['y'].shape

                print(X_dimensions)

                X = np.zeros((X_dimensions[0] * number_files, X_dimensions[1]))

                y = np.zeros((y_dimensions[0] * number_files, y_dimensions[1]))
            start_index = dataset_index * X_dimensions[0]
            stop_index = (dataset_index + 1) * X_dimensions[0]

            X[start_index:stop_index, :] = f['X']
            y[start_index:stop_index, :] = f['y']

        print('read: {}'.format(fullfilename))

    # write to file

    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('y', data=y, compression='gzip')
        f.create_dataset('X', data=X, compression='gzip')

    print('Wrote: {}'.format(output_filename))