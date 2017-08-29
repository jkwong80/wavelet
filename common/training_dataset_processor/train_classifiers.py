
import os, sys, glob, time
import h5py, cPickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# does not work with multidimensional y: mutual_info_regression, mutual_info_classif, f_classif, f_regression
# works with multidimensional y: chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
from keras import metrics

from sklearn.utils import class_weight

plot_markers = ''


def create_neural_network_5layer_model(input_size, output_size):
    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))

    model = Sequential()

    model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(400, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model

def create_neural_network_4layer_model(input_size, output_size):
    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))
    model = Sequential()

    model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model

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

models_dataset_root_path = os.path.join(base_dir, 'models')
models_dataset_path = os.path.join(models_dataset_root_path, '0')


if not os.path.exists(models_dataset_root_path):
    os.mkdir(models_dataset_root_path)


if not os.path.exists(models_dataset_path):
    os.mkdir(models_dataset_path)


kS_list = [2, 4, 8]

kB_list = [16]
gap_list = [4]
feature_indices_name_list = ['mask_filtered_features_2', 'mask_filtered_features_3']

# # parameters
# kS = 2
# kB = 16
# gap = 4

number_instance_load = 500000

# number of values to use in the cross validation
number_training_instances_cross_val = 10000
# the uuid of the training dataset
# training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'
# training_set_id = '5b178c11-a4e4-4b19-a925-96f27c49491b'
training_set_id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'

training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_filename = '%s__%03d__TrainingDataset.h5' % (training_set_id, 0)
training_dataset_fullfilename = os.path.join(training_dataset_path, training_dataset_filename)

for kS_index, kS in enumerate(kS_list):
    for kB_index, kB in enumerate(kB_list):
        for gap_index, gap in enumerate(gap_list):
            for feature_indices_name_index, feature_indices_name in enumerate(feature_indices_name_list):

                print('Working on %s, kS_%02d, kB_%02d, gap_%02d, %s' % (training_set_id, kS, kB, gap, feature_indices_name))

                # processed_dataset_path = os.path.join(processed_datasets_root_path, training_set_id)
                # processed_dataset_filename = '%s__%03d__kS_%02d__kB_%02d__gap_%02d__ProcessedDataset.h5' % (training_set_id, 0, kS, kB, gap)
                # processed_dataset_fullfilename = os.path.join(processed_dataset_path, processed_dataset_filename)

                filtered_features_dataset_root_path = os.path.join(base_dir, 'filtered_features_datasets')
                filtered_features_dataset_path  = os.path.join(filtered_features_dataset_root_path, training_set_id)
                filtered_features_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' % (training_set_id, kS, kB, gap, feature_indices_name)
                filtered_features_fullfilename = os.path.join(filtered_features_dataset_path, filtered_features_filename)

                # read in the file in the filtered features file
                # does doesn't have the final target values files

                with h5py.File(filtered_features_fullfilename, 'r') as f:
                    # X = f['X'][:number_instance_load,:]
                    # y = f['y'][:number_instance_load,:]
                    X = f['X'].value
                    y = f['y'].value

                print(X.shape)
                print(y.shape)

                # get source_name_list
                with h5py.File(training_dataset_fullfilename, 'r') as fid:
                    source_name_list = fid['source_name_list'].value
                # need a y where the no-isotope is index 0 and group together isotopes
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
                    isotope_mapping[isotope_string_index] = np.where(isotope_string_set == isotope_string)[0][0]+1
                # this the
                y_new = np.zeros(y.shape[0], dtype = np.int16)

                count_threshold = 50
                # count_threshold_fraction_max_counts = 0.10
                for instance_index in xrange(y.shape[0]):
                    if y[instance_index,:].sum() > count_threshold:
                        y_new[instance_index] = isotope_mapping[np.argmax(y[instance_index, :])]

                # Balance the classes (less background)
                number_background_instances_include = 100 * int(np.bincount(y_new)[1:].mean()/100)

                # find all instances that are class is 0 (background)
                cutt = y_new == 0
                indices = np.where(cutt)[0]
                cutt = np.zeros(len(y_new)) == 1
                cutt[indices[number_background_instances_include:]] = True
                mask_include = ~cutt

                print('Original Count of classes:')
                print(np.bincount(y_new))
                print('total number of instances: {}'.format(len(y_new)))

                print('Count of classes after removeing a lot of background instances:')
                print(np.bincount(y_new[mask_include]))
                print('total number of instances: {}'.format(len(y_new[mask_include])))

                lb = preprocessing.LabelBinarizer()
                lb.fit(y_new)
                lb.classes_

                y_matrix = lb.transform(y_new)

                # Create a dictionary with all the models we will be training
                clf_dict = {}

                # random forest
                # clf_dict['rf_0'] = RandomForestClassifier(n_estimators = 100, random_state=0, verbose = 1, n_jobs = 6)
                # simple neural networks with some dropoff (0.2)
                # clf_dict['nn_4layer'] = KerasClassifier(build_fn = create_neural_network_4layer_model, input_size = X.shape[1], output_size = y_matrix.shape[1], nb_epoch = 100, batch_size = 500, verbose = 1)
                # clf_dict['nn_5layer'] = KerasClassifier(build_fn = create_neural_network_5layer_model, input_size = X.shape[1], output_size = y_matrix.shape[1], nb_epoch = 100, batch_size = 500, verbose = 1)

                clf_dict['lr'] = LogisticRegression(n_jobs = 6)
                clf_dict['lda'] = LinearDiscriminantAnalysis()
                # clf_dict['gb'] = GradientBoostingClassifier(n_estimators = 50, verbose = 1)

                kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)

                results_dict = {}

                # 2d matrix
                y_matrix_mask_include = y_matrix[mask_include, :]
                # this is 1d matrix
                y_index_mask_include = np.argmax(y_matrix[mask_include, :], axis = 1)

                # print(y_matrix_mask_include.shape)
                # print(y_index_mask_include.shape)


                for model_index, model_name in enumerate(clf_dict.keys()):
                    print('Model Name: {}'.format(model_name))

                    # Does not work with neural network
                    if 'nn' not in model_name:
                        # results = cross_val_score(clf_dict[model_name], X[mask_include,:], y_matrix[mask_include,:], cv=kfold)
                        results_dict[model_name] = cross_val_score(clf_dict[model_name], X[mask_include, :][:number_training_instances_cross_val, :], \
                                                                   y_index_mask_include[:number_training_instances_cross_val],\
                                                                   cv=kfold)

                # set test and training split
                skf = StratifiedKFold(n_splits=2)
                for indices_test, indices_train in skf.split(X[mask_include, :], y[mask_include, 0]):
                    print(indices_test_mask_include)
                    print(indices_train_mask_include)

                # now let's train the models
                for model_index, model_name in enumerate(clf_dict.keys()):
                    print('Model Name: {}'.format(model_name))

                    if 'nn' not in model_name:
                        # results = cross_val_score(clf_dict[model_name], X[mask_include,:], y_matrix[mask_include,:], cv=kfold)
                        clf_dict[model_name].fit( X[mask_include, :][:number_training_instances_cross_val, :], \
                                                  y_index_mask_include[:number_training_instances_cross_val])

                prediction_dict = {}

                for model_index, model_name in enumerate(clf_dict.keys()):
                    print('Model Name: {}'.format(model_name))
                    if 'nn' not in model_name:
                        prediction_dict[model_name] = clf_dict[model_name].predict(X[mask_include,:])

                metrics = {}

                for model_index, model_name in enumerate(prediction_dict.keys()):
                    print('Model Name: {}'.format(model_name))

                    metrics[model_name] = {}
                    metrics[model_name]['accuracy'] = accuracy_score(y_index_mask_include, prediction_dict[model_name])
                    metrics[model_name]['f1'] = f1_score(y_index_mask_include, prediction_dict[model_name], average = 'micro')
                    metrics[model_name]['recall'] = recall_score(y_index_mask_include, prediction_dict[model_name], average = 'micro')
                    metrics[model_name]['precision'] = precision_score(y_index_mask_include, prediction_dict[model_name], average = 'micro')

                    for metric_name in metrics[model_name].keys():
                        print('{}: {}'.format(metric_name, metrics[model_name][metric_name]))

                del X
                del y
                # save stuff to file

                with open(os.path.join(models_dataset_path, filtered_features_filename.replace('FilteredFeaturesDataset', 'Models')), 'wb') as fid:
                    output = {}
                    output['metrics'] = metrics
                    output['cross_val_results'] = results_dict
                    output['models'] = clf_dict
                    output['prediction_dict'] = prediction_dict
                    output['prediction_dict'] = prediction_dict
                    output['mask_include'] = mask_include


                    cPickle.dump(output, fid, 2)

#
#
#
# plt.figure()
#
# plt.plot(np.argmax(y_matrix[mask_include, :], axis = 1)[:1000], alpha = 0.3, marker = '.', label = 'Truth')
#
# for model_index, model_name in enumerate(prediction_dict.keys()):
#     print('Model Name: {}'.format(model_name))
#     plt.plot(prediction_dict[model_name][:1000], alpha = 0.3, marker = matplotlib.markers.MarkerStyle.filled_markers[model_index], label = model_name)
#
# plt.xlabel('Instance Index')
# plt.ylabel("Truth and Prediction Class Index")
# plt.legend()
#
#
# prediction = clf.predict(X)
#
# # prediction_argmax_index = np.argmax(prediction, axis=1)
#
# prediction_test = prediction[indices_test]
# prediction_train = prediction[indices_train]
#
# y_new_test = y_new[indices_test]
# y_new_train = y_new[indices_train]
#
# # prediction_test_argmax_index = np.argmax(prediction_test, axis=1)
# # prediction_train_argmax_index = np.argmax(prediction_train, axis=1)
#
# # print confusion matrix
# confusion_mat = confusion_matrix(y_new_test, prediction_test)
#
# print(confusion_mat)
#
# # calculate recall, precision, etc
# recall = recall_score(y_new_test, prediction_test, average = 'micro')
# precision = precision_score(y_new_test, prediction_test, average = 'micro')
# f1 = f1_score(y_new_test, prediction_test, average = 'micro')
# accuracy = accuracy_score(y_new_test, prediction_test)
#
# print('Recall: {}'.format(recall))
# print('Precision: {}'.format(precision))
# print('f1: {}'.format(f1))
# print('accuracy: {}'.format(accuracy))
#
# # calculate the actual drive performance
# # detector source
# # misidentified source
# # fall positive (in section leading up to the source
# number_acquisitions_save = 25
#
# negative_window = [0,10]
#
# positive_window = [15,24]
#
# number_driveby_instances = y_new.shape[0]/number_acquisitions_save
#
# # prediction_driveby = np.zeros(number_driveby_instances)
# prediction_driveby_positive_window = []
# truth_driveby_positive_window = np.zeros(number_driveby_instances)
#
# prediction_driveby_negative_window = []
# truth_driveby_negative_window = np.zeros(number_driveby_instances)
#
# true_positive_driveby = np.zeros(number_driveby_instances)
#
# true_positive_driveby_positive_window = np.zeros(number_driveby_instances).astype(bool)
# false_positive_driveby_positive_window = np.zeros(number_driveby_instances).astype(bool)
# false_positive_driveby_negative_window = np.zeros(number_driveby_instances).astype(bool)
#
# for drive_by_index in xrange(number_driveby_instances):
#
#     start_1 = drive_by_index * number_acquisitions_save
#     stop_1 = start_1 + number_acquisitions_save
#
#     truth_driveby_positive_window[drive_by_index] = np.max(y_new[start_1:stop_1][positive_window[0]:positive_window[1]]  )
#     prediction_driveby_positive_window.append(Counter(prediction[start_1:stop_1][positive_window[0]:positive_window[1]]))
#
#     truth_driveby_negative_window[drive_by_index] = np.max(y_new[start_1:stop_1][negative_window[0]:negative_window[1]])
#     prediction_driveby_negative_window.append(Counter(prediction[start_1:stop_1][negative_window[0]:negative_window[1]]))
#
#     prediction_keys = prediction_driveby_positive_window[drive_by_index].keys()
#
#     true_positive_driveby_positive_window[drive_by_index] = truth_driveby_positive_window[drive_by_index] in prediction_keys
#
#     # set of prediction labels
#     temp1 = set(prediction_keys)
#     # set of truth labels
#     # - append 0 because we don't consider predictions as no-source to be necessarily wrong
#     temp2 = set([truth_driveby_positive_window[drive_by_index], 0.0])
#
#     incorrect_predictions_positive_windows = list(temp1.difference(temp2))
#
#     false_positive_driveby_positive_window[drive_by_index] = len(incorrect_predictions_positive_windows) > 0
#
#     # let's examine the negative window - before the source
#
#     prediction_keys_negative_window = prediction_driveby_negative_window[drive_by_index].keys()
#
#
#     # set of prediction labels
#     temp1 = set(prediction_keys_negative_window)
#     # set of truth labels
#     # - append 0 because we don't consider predictions as no-source to be necessarily wrong
#     temp2 = set([truth_driveby_negative_window[drive_by_index], 0.0])
#
#     incorrect_predictions_negative_windows = list(temp1.difference(temp2))
#
#     false_positive_driveby_negative_window[drive_by_index] = len(incorrect_predictions_negative_windows) > 0
#
#
#
# recall_positive_window = true_positive_driveby_positive_window.sum() / float(len(true_positive_driveby_positive_window))
#
# incorrect_prediction_rate_positive_window = false_positive_driveby_positive_window.sum() / float(len(false_positive_driveby_positive_window))
#
# false_positive_rate_negative_window = false_positive_driveby_negative_window.sum() / float(len(false_positive_driveby_negative_window))
#
#
#
# # Truth vs prediction
# plt.figure()
# plt.plot(y_new_test, prediction_test, '.k', alpha = 0.01, markersize = 25)
# plt.xlabel('Truth', fontsize = 16)
# plt.ylabel('Prediction', fontsize = 16)
# plt.legend()
#
#
#
# # plot of truth and prediction values vs instance
#
# plt.figure()
# plt.grid()
#
# test_indices = np.arange(len(y_new_test))
#
# mask_correct = y_new_test == prediction_test
#
# plt.plot(y_new_test, '-k', linewidth = 2, alpha = 0.4, label = 'Truth')
# plt.plot(test_indices[mask_correct], prediction_test[mask_correct], '*-g', linewidth = 2, alpha = 0.4, label = 'Prediction Correct')
# plt.plot(test_indices[~mask_correct], prediction_test[~mask_correct], '*r', linewidth = 2, alpha = 0.4, label = 'Prediction Incorrect')
#
# plt.xlabel('Training Instance', fontsize = 16)
# plt.ylabel('Truth/Prediction', fontsize = 16)
# plt.legend()
#
#
# # calculate the performance of the
#
# # y_argmax_index = np.argmax(y, axis = 1)
#
# # y_test_argmax_index = y_argmax_index[indices_test]
# # y_train_argmax_index = y_argmax_index[indices_train]
#
# cutt = (prediction_test_argmax_index >0 ) & (y_test_argmax_index == prediction_test_argmax_index)
#
#
# plt.figure()
# plt.plot(y_test.sum(1))
# plt.plot(prediction_test.sum(1))
#
#
#
# plt.figure()
# plt.grid()
# plt.plot(y_test_argmax_index, prediction_test, '.k', '')
#
#
# # plot
# plt.figure()
# plt.grid()
# plt.plot(y_test_argmax_index, prediction_test_argmax_index, '.k', alpha = 0.02, markersize = 25)
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
# plt.plot(y_test_argmax_index, '-k', linewidth = 2, alpha = 0.4, label = 'Truth')
# plt.plot(prediction_test_argmax_index, '-r', linewidth = 2, alpha = 0.4, label = 'Prediction')
#
# plt.xlabel('Training Instance', fontsize = 16)
# plt.ylabel('Truth/Prediction', fontsize = 16)
# plt.legend()
#
#
#
#
#
#
# # keras
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.constraints import maxnorm
# from keras.optimizers import SGD
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from keras import metrics
#
# from sklearn.utils import class_weight
#
#
#
#
#
# lb = preprocessing.LabelBinarizer()
# lb.fit(y_new)
#
# lb.classes_
#
#
#
# y_matrix = lb.transform(y_new)
#
#
# cutt = y_new == 0
#
# indices = np.where(cutt)[0]
#
# cutt = np.zeros(len(y_new)) == 1
#
# cutt[indices[6000:]] = True
#
# mask_include = ~cutt
#
# seed = 7
#
# np.random.seed(seed)
#
# #
# # class_weight = class_weight.compute_class_weight('balanced', np.unique(y_new), y_new)
#
# # opt = SGD(lr=0.001)
# # model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
#
# # model.fit(X[indices_train,:], y_matrix[indices_train,:], nb_epoch=25, batch_size=2000, validation_split = 0.33, class_weight = class_weight)
# model.fit(X[mask_include,:], y_matrix[mask_include,:], nb_epoch=200, batch_size=10000, validation_split = 0.25)
#
#
# prediction_matrix = model.predict(X)
#
# prediction = np.argmax(prediction_matrix, axis =1)
#
# # prediction_argmax_index = np.argmax(prediction, axis=1)
#
# prediction_test = prediction[indices_test]
# prediction_train = prediction[indices_train]
#
# y_new_test = y_new[indices_test]
# y_new_train = y_new[indices_train]
#
# # prediction_test_argmax_index = np.argmax(prediction_test, axis=1)
# # prediction_train_argmax_index = np.argmax(prediction_train, axis=1)
#
# # print confusion matrix
# confusion_mat = confusion_matrix(y_new_test, prediction_test)
#
# print(confusion_mat)
#
# # calculate recall, precision, etc
# recall = recall_score(y_new_test, prediction_test, average = 'micro')
# precision = precision_score(y_new_test, prediction_test, average = 'micro')
# f1 = f1_score(y_new_test, prediction_test, average = 'micro')
# accuracy = accuracy_score(y_new_test, prediction_test)
#
#
# print('Recall: {}'.format(recall))
# print('Precision: {}'.format(precision))
# print('f1: {}'.format(f1))
# print('accuracy: {}'.format(accuracy))
#
# # plot of truth and prediction values vs instance
#
# plt.figure()
# plt.grid()
#
# test_indices = np.arange(len(y_new_test))
#
# mask_correct = y_new_test == prediction_test
#
# plt.plot(y_new_test, '-k', linewidth = 2, alpha = 0.4, label = 'Truth')
# plt.plot(test_indices[mask_correct], prediction_test[mask_correct], '*-g', linewidth = 2, alpha = 0.4, label = 'Prediction Correct')
# plt.plot(test_indices[~mask_correct], prediction_test[~mask_correct], '*r', linewidth = 2, alpha = 0.4, label = 'Prediction Incorrect')
#
# plt.xlabel('Training Instance', fontsize = 16)
# plt.ylabel('Truth/Prediction', fontsize = 16)
# plt.legend()
#
