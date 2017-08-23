""" Examine the wavelet features"""
#

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
training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'




filtered_features_dataset_path = os.path.join(filtered_features_dataset_root_path, training_set_id)

# get the training dataset files
training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_fullfilename_list = glob.glob(os.path.join(training_dataset_path, '*.h5'))
training_dataset_filename_list = [os.path.split(f)[-1] for f in training_dataset_fullfilename_list]
training_dataset_index_list = [int(f.split('__')[1]) for f in training_dataset_filename_list]


# get the processed dataset files

glob_filter = 'kS_%02d__ProcessedDataset.h5' %(kS)

processed_dataset_path = os.path.join(processed_datasets_root_path, training_set_id)
processed_dataset_fullfilename_list = glob.glob(os.path.join(processed_dataset_path, '*ProcessedDataset.h5'))
processed_dataset_filename_list = [os.path.split(f)[-1] for f in processed_dataset_fullfilename_list]

processed_dataset_narrow_fullfilename_list = [f for f in processed_dataset_fullfilename_list if glob_filter in f]

# This is the file index
processed_dataset_narrow_index_list = [int(f.split('__')[1]) for f in processed_dataset_narrow_fullfilename_list]

target_values_path = os.path.join(processed_datasets_root_path, training_set_id)
target_values_fullfilename_list = glob.glob(os.path.join(processed_dataset_path, '*TargetValues.h5'))
target_values_index_list = [int(f.split('__')[1]) for f in target_values_fullfilename_list]

# '%s__%03d__kS_%02d__ProcessedDataset.h5' %(training_set_id, file_index, kS)
# 9a1be8d8-c573-4a68-acf8-d7c7e2f9830f__000__kS_01__ProcessedDataset.h5

# number of files to load

# focus on a single detector

detector_index = 1

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

    target_values_fullfilename = target_values_fullfilename_list[dataset_index]

    target_values = h5py.File(target_values_fullfilename, 'r')

    if dataset_index_index == 0
        source_signal_total_counts_all_detectors_matrix = target_values['source_signal_total_counts_all_detectors_matrix'][:,detector_index,:]

    source_signal_total_counts_all_detectors_matrix = target_values['source_signal_total_counts_all_detectors_matrix'].value

    source_index = target_values['source_index'].value
    source_name_list = target_values['source_name_list'].value

    number_sources = len(source_name_list)



# 2d plots of the wavelet vs acquisition index
# skip every other wavelet value

detector_index = 0
for i in xrange(0, 100, 5):

    source_name = source_name_list[int(source_index[i])]

    plt.figure(figsize = [20, 20])

    plt.subplot(2, 1, 1)

    plt.pcolor(SNR_matrix[i, detector_index, ::1,::2])
    plt.colorbar()
    plt.clim((-40, 90))
    plt.title('instance %d, signal+background, %s' %(i, source_name))


    plt.subplot(2, 1, 2)

    plt.pcolor(SNR_background_matrix[i, detector_index, ::1,::2])
    plt.colorbar()
    plt.clim((-40, 90))

    plt.title('instance %d, background, %s' %(i, source_name))

   # break



# feature selection


# Condense the feature and target matrices, removeing the transient section at the start of the drive by
number_acquisitions_save = 20
acquisitions_skip = 35

# source_signal_matrix_all = np.zeros((number_instances * number_acquisitions,source_signal_total_counts_all_detectors_matrix.shape[1]))
source_signal_matrix_all = np.zeros((number_instances * number_acquisitions_save,source_signal_total_counts_all_detectors_matrix.shape[1]))


# reshaping the matrix
# SNR_matrix_all = np.zeros((number_instances * number_acquisitions,number_wavelet_bins))
SNR_matrix_all = np.zeros((number_instances * number_acquisitions_save,number_wavelet_bins))

for instance_index in xrange(number_instances):
    if instance_index % 10 == 0:
        print('{}/{}'.format(instance_index, number_instances))
    # start = instance_index * number_acquisitions
    # stop = (instance_index + 1) * number_acquisitions

    start0 = acquisitions_skip
    stop0 = acquisitions_skip + number_acquisitions_save

    start = instance_index * number_acquisitions_save
    stop = (instance_index + 1) * number_acquisitions_save

    SNR_matrix_all[start:stop,:] = SNR_matrix[instance_index,detector_index,start0:stop0,:]

    source_signal_matrix_all[start:stop,:] = source_signal_total_counts_all_detectors_matrix[instance_index, :,start0:stop0].T


# trying to figure out what part to keep for


plt.figure()
plt.grid()


plt.plot(SNR_matrix[:,0,:,2051].T)
plt.plot(source_signal_total_counts_all_detectors_matrix[:,:,:].sum(0).sum(0)/800000.*20, linewidth = 3)


plt.figure()
plt.grid()

plt.plot(SNR_matrix[:,0,35:55,2051].T)
plt.plot(SNR_matrix[:,0,35:55,2051].mean(0), linewidth = 2)

x = source_signal_total_counts_all_detectors_matrix[:,:,35:55].mean(0).mean(0)
plt.plot(x * 20 / max(x), linewidth = 3)



# plot pcolor plot

plt.figure()
plt.pcolor(SNR_matrix_all[:400:1, ::5])
plt.colorbar()
plt.clim((-40, 90))
plt.title('instance %d, signal+background, %s' % (i, source_name))



# let's make some plots to make sure that things are okay.

plt.figure()

plt.grid()
plt.plot(source_signal_matrix_all)
plt.plot(SNR_matrix_all.sum(1))



training_instance_index = 15

plt.figure()

plt.grid()
plt.plot(source_signal_matrix_all[training_instance_index,:])
plt.plot(SNR_matrix_all[training_instance_index,:])


# use chi2 and the multidimensional y

number_features_keep = 200
feature_selection_ch2_multiclass = {}
feature_selection_ch2_multiclass['ch2'] = SelectKBest(chi2, k=number_features_keep)
feature_selection_ch2_multiclass['ch2'].fit(np.abs(SNR_matrix_all), source_signal_matrix_all>0)
feature_selection_ch2_multiclass['top_features_indices'] = feature_selection_ch2_multiclass['ch2'].get_support(indices = True)
feature_selection_ch2_multiclass['top_features_scores'] = np.array(feature_selection_ch2_multiclass['ch2'].scores_)[feature_selection_ch2_multiclass['top_features_indices']]



# feature_selection_dict[source_index]['ch2'] = SelectKBest(f_regression, k=number_features_keep)

# f_regression, mutual_info_regression


scoring_functions_dict = {}
scoring_functions_dict['chi2'] = chi2
scoring_functions_dict['f_classif'] = f_classif
scoring_functions_dict['mutual_info_classif'] = mutual_info_classif
scoring_functions_dict['f_regression'] = f_regression
scoring_functions_dict['mutual_info_regression'] = mutual_info_regression

# needs absolute values for features: chi2
# does not: f_classif, mutual_info_classif

# mutual_info_classif is slow

# use f_regression



number_features_keep = 100

feature_selection_dict = {}

ch2_dict = {}
topFeatureScores_dict = {}
topFeaturesIndices = {}

# loop over scoring functions
for scoring_function_key, scoring_function in scoring_functions_dict.iteritems():

    if scoring_function_key in ['mutual_info_regression', 'mutual_info_classif']:
        continue

    feature_selection_dict[scoring_function_key] = {}

    for source_index in xrange(number_sources):
        print('Working on {}, {}/{}'.format(scoring_function_key, source_index, number_sources))

        feature_selection_dict[scoring_function_key][source_index] = {}

        feature_selection_dict[scoring_function_key][source_index]['ch2'] = SelectKBest(scoring_function, k=number_features_keep)

        if scoring_function_key in ['chi2']:
            feature_selection_dict[scoring_function_key][source_index]['ch2'].fit(np.abs(SNR_matrix_all), source_signal_matrix_all[:,0])
        else:
            feature_selection_dict[scoring_function_key][source_index]['ch2'].fit(SNR_matrix_all, source_signal_matrix_all[:,0])
        feature_selection_dict[scoring_function_key][source_index]['top_features_indices'] =\
            feature_selection_dict[scoring_function_key][source_index]['ch2'].get_support(indices=True)

        feature_selection_dict[scoring_function_key][source_index]['top_features_scores'] = \
            np.array(feature_selection_dict[scoring_function_key][source_index]['ch2'].scores_)[feature_selection_dict[scoring_function_key][source_index]['top_features_indices']]


# get list of set of
feature_count_array = np.zeros((len(feature_selection_dict)+1, number_wavelet_bins))
feature_score_array = np.zeros((len(feature_selection_dict)+1, number_wavelet_bins))

indexindex = 0
for scoring_function_key, scoring_function in scoring_functions_dict.iteritems():

    if scoring_function_key in ['mutual_info_regression', 'mutual_info_classif']:
        continue

    print('Working on {}'.format(scoring_function_key))

    feature_selection_dict[scoring_function_key]['tally'] = Counter()

    for source_index in xrange(number_sources):
        feature_selection_dict[scoring_function_key]['tally'].update(feature_selection_dict[scoring_function_key][source_index]['top_features_indices'])

        feature_score_array[indexindex,feature_selection_dict[scoring_function_key][source_index]['top_features_indices']] +=\
            feature_selection_dict[scoring_function_key][source_index]['top_features_scores']

    indexindex +=1



for scoring_function_index, scoring_function_key in enumerate(feature_selection_dict.keys()):

    for feature_index, feature_count in feature_selection_dict[scoring_function_key]['tally'].iteritems():
        feature_count_array[scoring_function_index, int(feature_index)] = feature_count

feature_count_array[-1, feature_selection_ch2_multiclass['top_features_indices']] = 17
feature_score_array[-1, feature_selection_ch2_multiclass['top_features_indices']] = feature_selection_ch2_multiclass['top_features_scores']


feature_score_array[indexindex, feature_selection_dict[scoring_function_key][source_index]['top_features_indices']] += \
    feature_selection_dict[scoring_function_key][source_index]['top_features_scores']





with open('top_features.h5', 'w'):



# things to save
feature_score_array

