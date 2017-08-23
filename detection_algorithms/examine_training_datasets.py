
import os, sys, glob, time
import h5py


import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# does not work with multidimensional y: mutual_info_regression, mutual_info_classif, f_classif, f_regression
# works with multidimensional y: chi2

from sklearn.ensemble import RandomForestClassifier


from scipy.stats import pearsonr

# open a couple files and append the results


if 'INJECTION_RESOURCES' in os.environ:
    base_dir = os.environ['INJECTION_RESOURCES']
else:
    base_dir = os.path.join(os.environ['HOME'], 'injection_resources')

plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')

filtered_features_dataset_root_path = os.path.join(base_dir, 'filtered_features_datasets')


# parameters
kS = 2
# the uuid of the training dataset
# training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'
training_set_id = '5b178c11-a4e4-4b19-a925-96f27c49491b'


filtered_features_dataset_path = os.path.join(filtered_features_dataset_root_path, training_set_id)

# get the training dataset files
training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_fullfilename_list = glob.glob(os.path.join(training_dataset_path, '*.h5'))
training_dataset_fullfilename_list.sort()
training_dataset_filename_list = [os.path.split(f)[-1] for f in training_dataset_fullfilename_list]
training_dataset_index_list = [int(f.split('__')[1]) for f in training_dataset_filename_list]



training_dataset_index = 0

training_dataset_fullfilename = training_dataset_fullfilename_list[training_dataset_index]


dat = h5py.File(training_dataset_fullfilename, 'r')

#
# # get the processed dataset files
#
# glob_filter = 'kS_%02d__ProcessedDataset.h5' %(kS)
#
# processed_dataset_path = os.path.join(processed_datasets_root_path, training_set_id)
# processed_dataset_fullfilename_list = glob.glob(os.path.join(processed_dataset_path, '*ProcessedDataset.h5'))
# processed_dataset_filename_list = [os.path.split(f)[-1] for f in processed_dataset_fullfilename_list]
#
# processed_dataset_narrow_fullfilename_list = [f for f in processed_dataset_fullfilename_list if glob_filter in f]
#
# # This is the file index
# processed_dataset_narrow_index_list = [int(f.split('__')[1]) for f in processed_dataset_narrow_fullfilename_list]
#
# target_values_path = os.path.join(processed_datasets_root_path, training_set_id)
# target_values_fullfilename_list = glob.glob(os.path.join(processed_dataset_path, '*TargetValues.h5'))
# target_values_index_list = [int(f.split('__')[1]) for f in target_values_fullfilename_list]
#
# # '%s__%03d__kS_%02d__ProcessedDataset.h5' %(training_set_id, file_index, kS)
# 9a1be8d8-c573-4a68-acf8-d7c7e2f9830f__000__kS_01__ProcessedDataset.h5

# number of files to load

# focus on a single detector

detector_index = 0

dataset_index_list = np.arange(5)


# load several datasets
for dataset_index_index, dataset_index in enumerate(dataset_index_list):

    filename = processed_dataset_narrow_fullfilename_list[dataset_index]

    processed_dataset = h5py.File(filename, 'r')

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

    # define the arrays to store in
    if dataset_index_index == 0:
        SNR_matrix = np.zeros((number_instances * len(dataset_index_list), 1, number_acquisitions, number_wavelet_bins))
        SNR_background_matrix = np.zeros((number_instances * len(dataset_index_list), 1, number_acquisitions, number_wavelet_bins))

    # flatten the matrix

    start_instance_index = dataset_index_index * number_instances
    stop_instance_index = (dataset_index_index + 1) * number_instances

    SNR_matrix[start_instance_index:stop_instance_index,:,:,:] = processed_dataset['SNR_matrix'][:,detector_index,:,:]
    SNR_background_matrix[start_instance_index:stop_instance_index,:,:,:] = processed_dataset['SNR_background_matrix'][:,detector_index,:,:]
    #
    # target_values_fullfilename = target_values_fullfilename_list[dataset_index]
    #
    # target_values = h5py.File(target_values_fullfilename, 'r')
    #
    # if dataset_index_index == 0
    #     source_signal_total_counts_all_detectors_matrix = target_values['source_signal_total_counts_all_detectors_matrix'][:,detector_index,:]
    #
    # source_signal_total_counts_all_detectors_matrix = target_values['source_signal_total_counts_all_detectors_matrix'].value
    #
    # source_index = target_values['source_index'].value
    # source_name_list = target_values['source_name_list'].value
    #
    # number_sources = len(source_name_list)