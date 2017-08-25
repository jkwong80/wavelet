""" Examine the wavelet features and perform feature selection.
Also create rough
"""
#

import os, sys, glob, time
import h5py
import cPickle

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
kB = 16
gap = 4

# the uuid of the training dataset
# training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'
training_set_id = '5b178c11-a4e4-4b19-a925-96f27c49491b'


filtered_features_dataset_path = os.path.join(filtered_features_dataset_root_path, training_set_id)

# get the training dataset files
training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_fullfilename_list = glob.glob(os.path.join(training_dataset_path, '*.h5'))
training_dataset_filename_list = [os.path.split(f)[-1] for f in training_dataset_fullfilename_list]
training_dataset_index_list = [int(f.split('__')[1]) for f in training_dataset_filename_list]


# get the processed dataset files
glob_filter = 'kS_%02d__kB_%02d__gap_%02d__ProcessedDataset' % (kS, kB, gap)
processed_dataset_path = os.path.join(processed_datasets_root_path, training_set_id)
processed_dataset_fullfilename_list = glob.glob(os.path.join(processed_dataset_path, '*ProcessedDataset.h5'))
processed_dataset_filename_list = [os.path.split(f)[-1] for f in processed_dataset_fullfilename_list]
processed_dataset_narrow_fullfilename_list = [f for f in processed_dataset_fullfilename_list if glob_filter in f]
# This is the file index
processed_dataset_narrow_index_list = [int(f.split('__')[1]) for f in processed_dataset_narrow_fullfilename_list]

# get the target files
target_values_path = os.path.join(processed_datasets_root_path, training_set_id)
target_values_fullfilename_list = glob.glob(os.path.join(processed_dataset_path, '*TargetValues.h5'))
target_values_index_list = [int(f.split('__')[1]) for f in target_values_fullfilename_list]


# focus on a single detector

detector_indices_load_list = np.array([0])

dataset_index_list = np.arange(2)

load_background = False

# load several datasets
for dataset_index_index, dataset_index in enumerate(dataset_index_list):

    filename = processed_dataset_narrow_fullfilename_list[dataset_index]

    print('Loading: {}'.format(filename))

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
        SNR_matrix = np.zeros((number_instances * len(dataset_index_list), len(detector_indices_load_list), number_acquisitions, number_wavelet_bins))
        if load_background:
            SNR_background_matrix = np.zeros((number_instances * len(dataset_index_list), len(detector_indices_load_list), number_acquisitions, number_wavelet_bins))

    # flatten the matrix

    start_instance_index = dataset_index_index * number_instances
    stop_instance_index = (dataset_index_index + 1) * number_instances

    SNR_matrix_temp = processed_dataset['SNR_matrix'].value
    if load_background:
        SNR_background_matrix_temp = processed_dataset['SNR_background_matrix'].value

    number_samples_save = processed_dataset['number_samples_save'].value
    number_samples_skip = processed_dataset['number_samples_skip'].value


    SNR_matrix[start_instance_index:stop_instance_index,detector_indices_load_list,:,:] = SNR_matrix_temp[:,detector_indices_load_list,:,:]
    if load_background:
        SNR_background_matrix[start_instance_index:stop_instance_index,detector_indices_load_list,:,:] = SNR_background_matrix_temp[:,detector_indices_load_list,:,:]

    target_values_fullfilename = target_values_fullfilename_list[dataset_index]

    target_values = h5py.File(target_values_fullfilename, 'r')

    if dataset_index_index == 0:
        # only need to the load the source list once
        source_name_list = target_values['source_name_list'].value
        number_sources = len(source_name_list)

        source_signal_total_counts_all_detectors_matrix = np.zeros((number_instances * len(dataset_index_list), number_sources, number_samples_save))
        source_index = np.zeros(number_instances * len(dataset_index_list))

    source_signal_total_counts_all_detectors_matrix[start_instance_index:stop_instance_index,:,:] =\
        target_values['source_signal_total_counts_all_detectors_matrix'][:,:,number_samples_skip:(number_samples_skip+number_samples_save)]
    source_index[start_instance_index:stop_instance_index] = target_values['source_index'].value



# 2d plots of the wavelet vs acquisition index
# skip every other wavelet value

detector_index = 0
for i in xrange(9, 10, 1):

    source_name = source_name_list[int(source_index[i])]

    plt.figure(figsize = [20, 20])

    if load_background:
        plt.subplot(2, 1, 1)
        plt.pcolor(SNR_matrix[i, detector_index, ::1,::2].T)
        plt.colorbar()
        plt.clim((-40, 90))
        plt.xlabel('Acquisition Index')
        plt.ylabel('Wavelet Index')

        plt.title('instance %d, signal+background, %s' %(i, source_name))

        plt.subplot(2, 1, 2)
        plt.pcolor(SNR_background_matrix[i, detector_index, ::1,::2].T)
        plt.colorbar()
        plt.clim((-40, 90))
        plt.xlabel('Acquisition Index')
        plt.ylabel('Wavelet Index')

        plt.title('instance %d, background, %s' %(i, source_name))
    else:
        plt.pcolor(SNR_matrix[i, detector_index, ::1, ::1].T)
        plt.colorbar()
        plt.clim((-40, 90))
        plt.title('instance %d, signal+background, %s' % (i, source_name))
        plt.xlabel('Acquisition Index')
        plt.ylabel('Wavelet Index')

        # break




# Condense the feature and target matrices, removing the transient section at the start of the drive by

number_acquisitions_save = 25
acquisitions_skip = 10

# source_signal_matrix_all = np.zeros((number_instances * number_acquisitions,source_signal_total_counts_all_detectors_matrix.shape[1]))
source_signal_matrix_all = np.zeros((number_instances * number_acquisitions_save,source_signal_total_counts_all_detectors_matrix.shape[1]))

# reshaping the matrix
# SNR_matrix_all = np.zeros((number_instances * number_acquisitions,number_wavelet_bins))
SNR_matrix_all = np.zeros((number_instances * number_acquisitions_save,number_wavelet_bins))

detector_index = 0

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

# plot of the snr vs time for all
# also plotted is ht ecounts
wavelet_index = 2051
plt.figure()
plt.grid()
plt.plot(SNR_matrix[:200,0,:,2051].T)
temp = source_signal_total_counts_all_detectors_matrix[:,:,:].mean(0).mean(0)
plt.plot(temp/ np.max(temp) * np.max(SNR_matrix[:200,0,:,2051].mean(0)), '-k', linewidth = 4)
plt.title('wavelet_index: %d, ' %(wavelet_index))


plt.figure()
plt.grid()
plt.plot(SNR_matrix[:,0,10:35,2051].T)
plt.plot(SNR_matrix[:,0,10:35,2051].mean(0), linewidth = 2)
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



# PERFORM FEATURES SELECTION

# use chi2 and the multidimensional y

number_features_keep = 100
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

number_features_keep = 50

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
            feature_selection_dict[scoring_function_key][source_index]['ch2'].fit(np.abs(SNR_matrix_all), source_signal_matrix_all[:,source_index])
        else:
            feature_selection_dict[scoring_function_key][source_index]['ch2'].fit(SNR_matrix_all, source_signal_matrix_all[:,source_index])
        feature_selection_dict[scoring_function_key][source_index]['top_features_indices'] =\
            feature_selection_dict[scoring_function_key][source_index]['ch2'].get_support(indices=True)

        feature_selection_dict[scoring_function_key][source_index]['top_features_scores'] = \
            np.array(feature_selection_dict[scoring_function_key][source_index]['ch2'].scores_)[feature_selection_dict[scoring_function_key][source_index]['top_features_indices']]


# create a tally of the numger of times a features is select
# also sum up the scores but not sure how this scales for scores of different targets (sources)
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

feature_count_array[-1, feature_selection_ch2_multiclass['top_features_indices']] = 1
feature_score_array[-1, feature_selection_ch2_multiclass['top_features_indices']] = feature_selection_ch2_multiclass['top_features_scores']

feature_score_array[indexindex, feature_selection_dict[scoring_function_key][source_index]['top_features_indices']] += \
    feature_selection_dict[scoring_function_key][source_index]['top_features_scores']


mask_filtered_features_3 = feature_count_array.sum(0) > 3
mask_filtered_features_2 = feature_count_array.sum(0) > 2
mask_filtered_features_1 = feature_count_array.sum(0) > 1
mask_filtered_features_0 = feature_count_array.sum(0) > 0

mask_filtered_features = feature_count_array.sum(0) > 3

indices_filtered_features = np.where(mask_filtered_features)[0]

print('Number of features to keep: {}'.format(sum(mask_filtered_features)))

# print overlap with each
for scoring_function_key, scoring_function in scoring_functions_dict.iteritems():
    if scoring_function_key in ['mutual_info_regression', 'mutual_info_classif']:
        continue

    for source_index in xrange(number_sources):
        top_features_indices = feature_selection_dict[scoring_function_key][source_index]['top_features_indices']

        features_intersection = list(set(indices_filtered_features).intersection(set(top_features_indices)))

        print('{}, {}: {}'.format(scoring_function_key, source_index, len(features_intersection)))



# get the processed dataset files
filename_suffix = 'kS_%02d__kB_%02d__gap_%02d' % (kS, kB, gap)

with h5py.File('%s__%s__top_features.h5' %(training_set_id, filename_suffix), 'w') as f:
    f.create_dataset('mask_filtered_features_0', data = mask_filtered_features_0)
    f.create_dataset('mask_filtered_features_1', data = mask_filtered_features_1)
    f.create_dataset('mask_filtered_features_2', data = mask_filtered_features_2)
    f.create_dataset('mask_filtered_features_3', data = mask_filtered_features_3)
    f.create_dataset('mask_filtered_features', data = mask_filtered_features_2)
    f.create_dataset('feature_count_array', data = feature_count_array)
    f.create_dataset('feature_score_array', data = feature_score_array)

with open('%s__%s__top_features.pkl' %(training_set_id, filename_suffix), 'wb') as fid:
    output = {}
    output['mask_filtered_features_0'] = mask_filtered_features_0
    output['mask_filtered_features_1'] = mask_filtered_features_1
    output['mask_filtered_features_2'] = mask_filtered_features_2
    output['mask_filtered_features_3'] = mask_filtered_features_3
    output['mask_filtered_features'] = mask_filtered_features_2

    output['feature_count_array'] = feature_count_array
    output['feature_score_array'] = feature_score_array

    output['feature_selection_ch2_multiclass'] = feature_selection_ch2_multiclass
    output['feature_selection_dict'] = feature_selection_dict

    cPickle.dump(output, fid, 2)




# things to save
# feature_score_array

# plot of the tally count of from separate scoring algorithms

plt.figure()
plt.grid()
for scoring_function_index, scoring_function_key in enumerate(feature_selection_dict.keys()):
    plt.plot(feature_count_array[scoring_function_index,:], alpha = 0.3, linewidth = 3, label = scoring_function_key)
plt.plot(feature_count_array[-1,:], '.-k', alpha = 0.3, linewidth = 3, label = 'chi2, all features at once')
plt.xlabel('Wavelet Feature Index')
plt.ylabel('Count')
plt.legend(loc = 2)
plt.title()


# plot of the score of from separate scoring algorithms

plt.figure()
plt.grid()
for scoring_function_index, scoring_function_key in enumerate(feature_selection_dict.keys()):
    plt.plot(feature_score_array[scoring_function_index,:], alpha = 0.3, linewidth = 3, label = scoring_function_key)

plt.plot(feature_score_array[-1,:], '-k', alpha = 0.3, linewidth = 3, label = 'chi2, all features at once')

plt.xlabel('Wavelet Feature Index', fontsize = 16)
plt.ylabel('Scores', fontsize = 16)
plt.legend()
plt.yscale('log')


# plot in separate subplots
plt.figure()
plt.plot(feature_count_array.sum(0), '.-k', alpha = 0.5, linewidth = 3, markersize = 12, label = 'chi2, all features at once')
plt.plot(feature_count_array.sum(0), '-k', alpha = 0.5, linewidth = 3)


# save the features to file


# # set the arrays
# X = SNR_matrix_all[:, feature_selection_ch2_multiclass['top_features_indices']]
# y = source_signal_matrix_all[:, :]
#
# y_argmax_index = np.argmax(y, axis = 1)
#
#
# # r, p = pearsonr(SNR_matrix_all, source_signal_matrix_all)
#
# # random forest
# clf = RandomForestClassifier(random_state=0, verbose = 1, n_jobs = 4)
# clf.fit(X, y)
# prediction = clf.predict(X)
#
# prediction_argmax_index = np.argmax(prediction, axis=1)
#
#
# # plot
# plt.figure()
# plt.grid()
# plt.plot(y_argmax_index, prediction_argmax_index, '.k', alpha = 0.02, markersize = 25)
#
# plt.xlabel('Truth', fontsize = 16)
# plt.ylabel('Prediction', fontsize = 16)
# plt.legend()
#
#
# # plot of truth and prediction values vs instance
#
# plt.figure()
# plt.grid()
#
# plt.plot(y_argmax_index, '-k', linewidth = 2, alpha = 0.4, label = 'Truth')
# plt.plot(prediction_argmax_index, '-r', linewidth = 2, alpha = 0.4, label = 'Prediction')
#
# plt.xlabel('Training Instance', fontsize = 16)
# plt.ylabel('Truth/Prediction', fontsize = 16)
# plt.legend()
#

# # false positives
#
# cutt = (y_argmax_index == 0) & (y_argmax_index != prediction_argmax_index)
