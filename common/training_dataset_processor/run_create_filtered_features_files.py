""" Examine the wavelet features and perform feature selection.
Also create rough

Arguments
    training_set_id uuid
    training_set_id = str(sys.argv[1])
    number_threads = int(sys.argv[2])
    kS_list = ast.literal_eval(sys.argv[3])
    kB = ast.literal_eval(sys.argv[4])
    gap = ast.literal_eval(sys.argv[5])
    acquisitions_skip = ast.literal_eval(sys.argv[6])
    number_acquisitions_save = ast.literal_eval(sys.argv[7])
    file_index_start = int(sys.argv[8])
    file_index_stop = int(sys.argv[9])+1

"""
#

import os, sys, glob, time
import h5py, ast
import cPickle
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
    kS_list = ast.literal_eval(sys.argv[3])
    kB = ast.literal_eval(sys.argv[4])
    gap = ast.literal_eval(sys.argv[5])
    acquisitions_skip = ast.literal_eval(sys.argv[6])
    number_acquisitions_save = ast.literal_eval(sys.argv[7])
    file_index_start = int(sys.argv[8])
    file_index_stop = int(sys.argv[9])+1

    feature_indices_name = sys.argv[10]

    pool = Pool(processes=number_threads)

    arguments_list = []

    job_number = 0

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

            # this is the output file name
            filtered_features_dataset_fullfilename = os.path.join(filtered_features_dataset_path,
                                                                  '%s__%03d__kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' \
                                                                  % (training_set_id, file_list_index, kS, kB, gap, feature_indices_name))

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
            param['mask_filtered_features'] = feature_indices_name

            param['detector_index'] = 0

            param['worker_no'] = job_number
            job_number += 1

            arguments_list.append(param)

            print(param)

    result = pool.map(training_dataset_processor.CreateFilteredFeaturesFile, arguments_list)


    pool.close()
    pool.join()
    print('Time Elapsed: %3.3f' %(time.time() - t_start))

