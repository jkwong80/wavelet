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

dataset_index = 0


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


# focus on a single detector


detector_index = 0


# flatten the matrix
SNR_matrix = processed_dataset['SNR_matrix'].value
SNR_background_matrix = processed_dataset['SNR_background_matrix'].value


target_values_fullfilename = target_values_fullfilename_list[dataset_index]

target_values = h5py.File(target_values_fullfilename, 'r')

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





# plot of the tally count of from separate scoring algorithms

plt.figure()
plt.grid()
for scoring_function_index, scoring_function_key in enumerate(feature_selection_dict.keys()):
    plt.plot(feature_count_array[scoring_function_index,:], alpha = 0.3, linewidth = 3, label = scoring_function_key)

plt.plot(feature_count_array[-1,:], '.-k', alpha = 0.3, linewidth = 3, label = 'chi2, all features at once')

plt.xlabel('Wavelet Feature Index')
plt.ylabel('Count')
plt.legend()



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
#

# ######################

# create the condensed datasets

detector_index = 0


training_dataset_path

kS = 2

number_acquisitions_save = 20
acquisitions_skip = 35
dataset_index_start = 20
dataset_index_stop = 50

dataset_index_start = 32
dataset_index_stop = 60


for dataset_index in xrange(dataset_index_start,dataset_index_stop):


    processed_dataset_fullfilename = os.path.join(processed_dataset_path, '%s__%03d__kS_%02d__ProcessedDataset.h5'\
                                                  %(training_set_id, dataset_index, kS))

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


    # focus on a single detector


    # flatten the matrix
    SNR_matrix = processed_dataset['SNR_matrix'].value


    # target_values_fullfilename = target_values_fullfilename_list[dataset_index]

    target_values_fullfilename = os.path.join(target_values_path, '%s__%03d__TargetValues.h5'\
                                                  %(training_set_id, dataset_index))

    target_values = h5py.File(target_values_fullfilename, 'r')

    source_signal_total_counts_all_detectors_matrix = target_values['source_signal_total_counts_all_detectors_matrix'].value

    source_index = target_values['source_index']
    source_name_list = target_values['source_name_list']

    number_sources = len(source_name_list)


    source_signal_matrix_all = np.zeros((number_instances * number_acquisitions_save,source_signal_total_counts_all_detectors_matrix.shape[1]))


    # reshaping the matrix
    # SNR_matrix_all = np.zeros((number_instances * number_acquisitions,number_wavelet_bins))
    SNR_matrix_all = np.zeros((number_instances * number_acquisitions_save,number_wavelet_bins))

    for instance_index in xrange(number_instances):
        if instance_index % 10 == 0:
            print('{}/{}'.format(instance_index, number_instances))
        start0 = acquisitions_skip
        stop0 = acquisitions_skip + number_acquisitions_save

        start = instance_index * number_acquisitions_save
        stop = (instance_index + 1) * number_acquisitions_save


        SNR_matrix_all[start:stop,:] = SNR_matrix[instance_index,detector_index,start0:stop0,:]

        source_signal_matrix_all[start:stop,:] = source_signal_total_counts_all_detectors_matrix[instance_index, :,start0:stop0].T


    filtered_features_dataset_fullfilename = os.path.join(filtered_features_dataset_path, '%s__%03d__kS_%02d__FilteredFeaturesDataset.h5'\
                                                  %(training_set_id, dataset_index, kS))

    X = SNR_matrix_all[:, feature_selection_ch2_multiclass['top_features_indices']]

    y = source_signal_matrix_all[:, :]

    with h5py.File(filtered_features_dataset_fullfilename, 'w') as f:
        f.create_dataset('y', data=y, compression = 'gzip')
        f.create_dataset('X', data=X, compression = 'gzip')

    print('Wrote: {}'.format(filtered_features_dataset_fullfilename))


#  ############## concatenate separate files  #######################
number_files = 30
for dataset_index in xrange(0,number_files):


    filtered_features_dataset_fullfilename = os.path.join(filtered_features_dataset_path, '%s__%03d__kS_%02d__FilteredFeaturesDataset.h5'\
                                                  %(training_set_id, dataset_index, kS))

    with h5py.File(filtered_features_dataset_fullfilename, 'r') as f:

        if dataset_index == 0:
            X_dimensions = f['X'].shape
            y_dimensions = f['y'].shape

            print(X_dimensions)

            X = np.zeros((X_dimensions[0] * number_files, X_dimensions[1]))

            y = np.zeros((y_dimensions[0] * number_files, y_dimensions[1]))
        start_index = dataset_index * X_dimensions[0]
        stop_index = (dataset_index+1) * X_dimensions[0]

        X[start_index:stop_index,:] = f['X']
        y[start_index:stop_index,:] = f['y']

    print('read: {}'.format(filtered_features_dataset_fullfilename))


# write to file
filtered_features_dataset_all_fullfilename = os.path.join(filtered_features_dataset_path,
                                                      '%s__all__kS_%02d__FilteredFeaturesDataset.h5' \
                                                      % (training_set_id, kS))

with h5py.File(filtered_features_dataset_all_fullfilename, 'w') as f:
    f.create_dataset('y', data=y, compression='gzip')
    f.create_dataset('X', data=X, compression='gzip')

##################### #########################################

# read in the file

with h5py.File(filtered_features_dataset_all_fullfilename, 'r') as f:
    X = f['X'].value
    y = f['y'].value

number_resample = 5
# number_acquisitions_save

X_no_resample = np.zeros((X.shape[0]/number_resample, X.shape[1]))
y_no_resample = np.zeros((y.shape[0]/number_resample, y.shape[1]))


for i in xrange(X_no_resample.shape[0]):

    start_0 = i * number_acquisitions_save * number_resample
    stop_0 = start_0 + number_acquisitions_save

    start_1 = i * number_acquisitions_save
    stop_1 = start_1 + number_acquisitions_save

    X_no_resample[start_1:stop_1] = X[start_0:stop_0, :]
    y_no_resample[start_1:stop_1] = y[start_0:stop_0, :]



# need a y where the no-isotope is index 0 and group together isotopes
isotope_string_list = []

for source_name in source_name_list:
    index_underscore = source_name.find('_')
    isotope_string_list.append(source_name[:index_underscore])

isotope_string_list = np.array(isotope_string_list)
isotope_string_set = np.array(list(set(isotope_string_list)))
isotope_string_set.sort()


# map from isotope string to y value
isotope_mapping = {}
for isotope_string_index, isotope_string in enumerate(isotope_string_list):
    isotope_mapping[isotope_string_index] = np.where(isotope_string_set == isotope_string)[0][0]+1

y_new = np.zeros(y.shape[0])

count_threshold = 1500

# count_threshold_fraction_max_counts = 0.10

for instance_index in xrange(y.shape[0]):

    # if y[instance_index,:].sum() == 0:
    #     y_new[instance_index] = 0
    # else:
    #     y_new[instance_index] = isotope_mapping[np.argmax(y[instance_index, :])]

    if y[instance_index,:].sum() > count_threshold:
        y_new[instance_index] = isotope_mapping[np.argmax(y[instance_index, :])]




# plot the mean subset of wavelet for each source

indices = np.argmax(y, axis= 1)

indices[y.sum(1) == 0] = y.shape[1]

for i in xrange(y.shape[1]+1):

    cutt = indices == i
    plt.plot(X[cutt,:].mean(0), label = '{}'.format(i))

plt.xlabel('Subwavelet Index')
plt.ylabel('Mean Count')
plt.legend()


plt.figure()
plt.pcolor(X[:2125,:])
plt.colorbar()




plt.figure()
plt.plot(y_new, label = 'New Target')
plt.plot(np.argmax(y, axis = 1), label = 'Old Target')
#
plt.legend()

# for source_index, source_name in enumerate(source_name_list):
#
#     index_underscore = source_name.find('_')
#     isotope_string = source_name[:index_underscore]
#
#     index = np.where(isotope_string_set == isotope_string)[0][0]
#     print(index, isotope_string)
#     y_new[:, index+1] = y_new[:, index+1] + y[:,source_index]
#

# set the arrays
skf = StratifiedKFold(n_splits=2)
for indices_test, indices_train in skf.split(X,y[:,0]):
    print(indices_test)
    print(indices_train)

y_argmax_index = np.argmax(y, axis = 1)

# r, p = pearsonr(SNR_matrix_all, source_signal_matrix_all)

# random forest
print('Training random forest classifier')
clf = RandomForestClassifier(random_state=0, verbose = 1, n_jobs = 4)
clf.fit(X[indices_train,:], y_new[indices_train])


prediction = clf.predict(X)

# prediction_argmax_index = np.argmax(prediction, axis=1)

prediction_test = prediction[indices_test]
prediction_train = prediction[indices_train]

y_new_test = y_new[indices_test]
y_new_train = y_new[indices_train]

# prediction_test_argmax_index = np.argmax(prediction_test, axis=1)
# prediction_train_argmax_index = np.argmax(prediction_train, axis=1)


# Truth vs prediction
plt.figure()
plt.plot(y_new_test, prediction_test, '.k', alpha = 0.02, markersize = 25)
plt.xlabel('Truth', fontsize = 16)
plt.ylabel('Prediction', fontsize = 16)
plt.legend()



# calculate the performance of the


# y_argmax_index = np.argmax(y, axis = 1)

# y_test_argmax_index = y_argmax_index[indices_test]
# y_train_argmax_index = y_argmax_index[indices_train]

cutt= (prediction_test_argmax_index >0 ) & (y_test_argmax_index == prediction_test_argmax_index)


plt.figure()
plt.plot(y_test.sum(1))
plt.plot(prediction_test.sum(1))



plt.figure()
plt.grid()
plt.plot(y_test_argmax_index, prediction_test, '.k', '')


# plot
plt.figure()
plt.grid()
plt.plot(y_test_argmax_index, prediction_test_argmax_index, '.k', alpha = 0.02, markersize = 25)

plt.xlabel('Truth', fontsize = 16)
plt.ylabel('Prediction', fontsize = 16)
plt.legend()


# plot of truth and prediction values vs instance

plt.figure()
plt.grid()

plt.plot(y_test_argmax_index, '-k', linewidth = 2, alpha = 0.4, label = 'Truth')
plt.plot(prediction_test_argmax_index, '-r', linewidth = 2, alpha = 0.4, label = 'Prediction')

plt.xlabel('Training Instance', fontsize = 16)
plt.ylabel('Truth/Prediction', fontsize = 16)
plt.legend()



# keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



lb = preprocessing.LabelBinarizer()
lb.fit(y_new)

lb.classes_

y_matrix = lb.transform(y_new)



seed = 7

np.random.seed(seed)

model = Sequential()

model.add(Dense(X.shape[1], input_dim=X.shape[1], init='uniform', activation='relu'))
model.add(Dense(100, init='uniform', activation='relu'))
# model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
model.add(Dense(y_matrix.shape[1], init='uniform', activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X, y_matrix, nb_epoch=25, batch_size=200, validation_split = 0.33)

prediction = model.predict(X)

predication_arg_max = np.argmax(prediction, axis =1)


# plot
plt.figure()
plt.grid()
plt.plot(y_new, predication_arg_max, '.k', alpha = 0.02, markersize = 25)

plt.xlabel('Truth', fontsize = 16)
plt.ylabel('Prediction', fontsize = 16)
plt.legend()