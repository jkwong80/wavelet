""" Examine the wavelet features and perform feature selection.
Also create rough
"""
#

import os, sys, glob, time
import h5py, ast
import cPickle

import numpy as np

from collections import Counter

import training_dataset_processor
from multiprocessing import Pool

# ######################

# create the condensed datasets

detector_index = 0


# these' have already been defined above
# number_acquisitions_save = 25
# acquisitions_skip = 10

dataset_index_start = 0
dataset_index_stop = 5

# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'mm
# base_dir = os.path.join(os.environ['HOME'], 'injection_resources')
if 'INJECTION_RESOURCES' in os.environ:
    base_dir = os.environ['INJECTION_RESOURCES']
else:
    base_dir = os.path.join(os.environ['HOME'], 'injection_resources')

plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

snr_root_path = os.path.join(base_dir, 'snr_functions')

if not os.path.exists(snr_root_path):
    os.mkdir(snr_root_path)

snr_path = os.path.join(snr_root_path, '20170824')

feature_selection_path = os.path.join(base_dir, 'feature_selection')

feature_selection_filename = '5b178c11-a4e4-4b19-a925-96f27c49491b__kS_02__kB_16__gap_04__top_features.h5'
feature_selection_fullfilename = os.path.join(feature_selection_path, feature_selection_filename)


filtered_features_dataset_root_path = os.path.join(base_dir, 'filtered_features_datasets')


# CreateFilteredFeaturesFile(processed_dataset_fullfilename, target_values_fullfilename,
#                            filtered_features_dataset_fullfilename, best_features_fullfilename, \
#                            number_acquisitions_save, acquisitions_skip, detector_index):

if __name__ == '__main__':

    # append a time string so that we know all the jobs are part of the same unit
    t_start = time.time()

    # LOAD DATA FROM SIMULATION FILE
    training_set_id = str(sys.argv[1])

    number_threads = int(sys.argv[2])

    pool = Pool(processes=number_threads)

    arguments_list = []

    job_number = 0

    # for file_list_index in xrange(4, len(training_dataset_fullfilename_list)):

    kS_list = ast.literal_eval(sys.argv[3])
    kB = ast.literal_eval(sys.argv[4])

    gap = ast.literal_eval(sys.argv[5])

    acquisitions_skip = ast.literal_eval(sys.argv[6])
    number_acquisitions_save = ast.literal_eval(sys.argv[7])

    file_index_start = int(sys.argv[8])
    file_index_stop = int(sys.argv[9])+1

    print('kS_list', kS_list)
    print('file_index_start: {}'.format(file_index_start))
    print('file_index_stop: {}'.format(file_index_stop))

    skip_list = []

    filtered_features_dataset_path = os.path.join(filtered_features_dataset_root_path, training_set_id)

    if not os.path.exists(filtered_features_dataset_path):
        os.mkdir(filtered_features_dataset_path)

    for file_list_index in xrange(file_index_start, file_index_stop):

        for kS in kS_list:

            # get the processed dataset files
            processed_dataset_path = os.path.join(processed_datasets_root_path, training_set_id)

            print('os.path.exists(processed_dataset_path): {}'.format(os.path.exists(processed_dataset_path)))
            # glob_filter = 'kS_%02d__kB_%02d__gap_%02d__ProcessedDataset' % (kS, kB, gap)
            # processed_dataset_fullfilename_list = glob.glob(
            #     os.path.join(processed_dataset_path, '*ProcessedDataset.h5'))
            # processed_dataset_filename_list = [os.path.split(f)[-1] for f in processed_dataset_fullfilename_list]
            # processed_dataset_narrow_fullfilename_list = [f for f in processed_dataset_fullfilename_list if
            #                                               glob_filter in f]
            # # This is the file index
            # processed_dataset_narrow_index_list = [int(f.split('__')[1]) for f in
            #                                        processed_dataset_narrow_fullfilename_list]


            filtered_features_dataset_fullfilename = os.path.join(filtered_features_dataset_path,
                                                                  '%s__%03d__kS_%02d__kB_%02d__gap_%02d__FilteredFeaturesDataset.h5' \
                                                                  % (training_set_id, file_list_index, kS, kB, gap))

            processed_dataset_filename = '%s__%03d__kS_%02d__kB_%02d__gap_%02d__ProcessedDataset.h5' %(training_set_id, file_list_index, kS, kB, gap)
            processed_dataset_fullfilename = os.path.join(processed_dataset_path, processed_dataset_filename)

            # build the target file name
            target_values_path = os.path.join(processed_datasets_root_path, training_set_id)
            target_values_fullfilename = os.path.join(target_values_path, '%s__%03d__TargetValues.h5' %(training_set_id, file_list_index))

            param = {}
            param['processed_dataset_fullfilename'] = processed_dataset_fullfilename
            param['target_values_fullfilename'] = target_values_fullfilename
            param['filtered_features_dataset_fullfilename'] = filtered_features_dataset_fullfilename
            param['best_features_fullfilename'] = feature_selection_fullfilename
            param['number_acquisitions_save'] = number_acquisitions_save
            param['acquisitions_skip'] = acquisitions_skip
            param['detector_index'] = 0

            param['worker_no'] = job_number
            job_number += 1

            arguments_list.append(param)

            print(param)

    result = pool.map(training_dataset_processor.CreateFilteredFeaturesFile, arguments_list)


    pool.close()
    pool.join()
    print('Time Elapsed: %3.3f' %(time.time() - t_start))






#
#
#
# for dataset_index in xrange(dataset_index_start,dataset_index_stop):
#
#     processed_dataset_fullfilename = os.path.join(processed_dataset_path, '%s__%03d__kS_%02d__kB_%02d__gap_%02d__ProcessedDataset.h5'\
#                                                   %(training_set_id, dataset_index, kS, kB, gap))
#
#     processed_dataset = h5py.File(processed_dataset_fullfilename, 'r')
#
#     # get the dimensions of the matrices
#     dimensions = processed_dataset['SNR_matrix'].shape
#     # number of samples (acquisitions) in the pass-by
#     number_instances = dimensions[0]
#     # number of detectors
#     number_detectors = dimensions[1]
#     # number of samples (acquisitions) in the pass-by
#     number_acquisitions = dimensions[2]
#     # number of wavelet bins
#     number_wavelet_bins = dimensions[3]
#
#
#     # focus on a single detector
#
#     # flatten the matrix
#     SNR_matrix = processed_dataset['SNR_matrix'].value
#
#
#     # target_values_fullfilename = target_values_fullfilename_list[dataset_index]
#
#     target_values_fullfilename = os.path.join(target_values_path, '%s__%03d__TargetValues.h5'\
#                                                   %(training_set_id, dataset_index))
#
#     target_values = h5py.File(target_values_fullfilename, 'r')
#
#     source_signal_total_counts_all_detectors_matrix = target_values['source_signal_total_counts_all_detectors_matrix'].value
#
#     source_index = target_values['source_index']
#     source_name_list = target_values['source_name_list']
#
#     number_sources = len(source_name_list)
#
#     source_signal_matrix_all = np.zeros((number_instances * number_acquisitions_save,source_signal_total_counts_all_detectors_matrix.shape[1]))
#
#     # reshaping the matrix
#     # SNR_matrix_all = np.zeros((number_instances * number_acquisitions,number_wavelet_bins))
#     SNR_matrix_all = np.zeros((number_instances * number_acquisitions_save,number_wavelet_bins))
#
#     for instance_index in xrange(number_instances):
#         if instance_index % 10 == 0:
#             print('{}/{}'.format(instance_index, number_instances))
#         start0 = acquisitions_skip
#         stop0 = acquisitions_skip + number_acquisitions_save
#
#         start = instance_index * number_acquisitions_save
#         stop = (instance_index + 1) * number_acquisitions_save
#
#
#         SNR_matrix_all[start:stop,:] = SNR_matrix[instance_index,detector_index,start0:stop0,:]
#
#         source_signal_matrix_all[start:stop,:] = source_signal_total_counts_all_detectors_matrix[instance_index, :,start0:stop0].T
#
#
#     filtered_features_dataset_fullfilename = os.path.join(filtered_features_dataset_path, '%s__%03d__kS_%02d__FilteredFeaturesDataset.h5'\
#                                                   %(training_set_id, dataset_index, kS))
#
#     X = SNR_matrix_all[:, feature_selection_ch2_multiclass['top_features_indices']]
#
#     y = source_signal_matrix_all[:, :]
#
#     with h5py.File(filtered_features_dataset_fullfilename, 'w') as f:
#         f.create_dataset('y', data=y, compression = 'gzip')
#         f.create_dataset('X', data=X, compression = 'gzip')
#
#     print('Wrote: {}'.format(filtered_features_dataset_fullfilename))
#
