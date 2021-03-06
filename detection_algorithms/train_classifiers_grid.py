
import cPickle
import copy
import sys
import time
from collections import Counter

import h5py
import numpy as np
import os
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# does not work with multidimensional y: mutual_info_regression, mutual_info_classif, f_classif, f_regression
# works with multidimensional y: chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

plot_markers = ''

sys.setrecursionlimit(10000)

from detection_algorithms.nn_models import create_neural_network_2layer_model, create_neural_network_3layer_model, create_neural_network_4layer_model,\
    create_neural_network_4layer_no_dropout_model, create_neural_network_5layer_model

# Define the paths to things in $INJECTION_RESOURCES
if 'INJECTION_RESOURCES' in os.environ:
    base_dir = os.environ['INJECTION_RESOURCES']
else:
    base_dir = os.path.join(os.environ['HOME'], 'injection_resources')

plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')
filtered_features_dataset_root_path = os.path.join(base_dir, 'filtered_features_datasets')
models_dataset_root_path = os.path.join(base_dir, 'models')
if not os.path.exists(models_dataset_root_path):
    os.mkdir(models_dataset_root_path)



###########  PARAMETERS  #####################

# subpath in $INJECTION_RESOURCES/models to save models
models_dataset_subpath = '10'

# the uuid of the training dataset
training_set_id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'

# lost of parameters to cycle through
kS_list = [4]
kB_list = [16]
gap_list = [4]

# kS_list = [2, 4]
# kB_list = [16, 32]
# gap_list = [4]

# this is the name of the list of features in the file created by select_best_features_first_pass.py
feature_indices_name_list = ['mask_filtered_features_2']

# List of algorithms to train
# lda - linear discriminant analysis
# rf - random forest
# nn_Xlayer - neural network with X layers with 0.2 dropout regularization
# nn_Xlayer_no_dropout - neural network with X layers with no dropout

# model_keys = ['lr', 'lda']
# model_keys = ['lda', 'nn_2layer', 'nn_3layer', 'nn_4layer', 'nn_5layer', 'nn_4layer_no_dropout']
# model_keys = ['lda', 'rf_0', 'lr', 'nn_2layer', 'nn_3layer', 'nn_4layer', 'nn_5layer', 'nn_4layer_no_dropout']
model_keys = ['lda', 'rf_0', 'nn_2layer', 'nn_3layer', 'nn_4layer', 'nn_5layer', 'nn_4layer_no_dropout']

# the number of epochs to train the neural networks
nn_epochs = 100

# Define how much data to load from the consolidated filtered features file
load_subset = False
number_instance_load = 1000000

# the target value is positive if the number of counts from the source injection is above this amount
# (the background is usually about 1000 counts/second)
count_threshold = 100

# The snr wavelet values become negative shortly after passing by the source.  I didn't want to train on this section
# of the data. I only want to train only on the (postive) peak before it becomes negative.  The time width of the peak
# in the SNR depends on the kS.  acq_buffer_size is the map from kS to the number of acquisitions to include in the peak
acq_buffer_size = {2: 4, 4: 6, 8: 6}

# The number of background acquisitions grealy outnumbers the acquisitions with source injection counts;
# there is a step to balance things out.  The number of background = background_ratio * the average number of other instances
background_ratio = 20

# create classifiers trained on these speed ranges
# for example for [5,10], it will
speed_bounds_list = [[5, 10], [10, 15], [15, 20], [20, 25], [0, 25]]


# ########### DANGER - these values should be passed on somehow from file
# This is the number of acquisitions that were saved to the filtered features files.
number_acquisitions_save = 25
# These are the acquisitions before we get close to the source - these are almost always background (counts from source
# is usually zero or nearly zero)
negative_window = [0, 10]

# when the vehicle is close to the source; there should be non-zero counts.
positive_window = [15, 24]

#############  END of PARAMETERS  #################



# Train the classifiers

models_dataset_path = os.path.join(models_dataset_root_path, models_dataset_subpath)
if not os.path.exists(models_dataset_path):
    os.mkdir(models_dataset_path)

training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_filename = '%s__%03d__TrainingDataset.h5' % (training_set_id, 0)
training_dataset_fullfilename = os.path.join(training_dataset_path, training_dataset_filename)


# loop in kS, kB, gap and feature set name
for kS_index, kS in enumerate(kS_list):
    for kB_index, kB in enumerate(kB_list):
        for gap_index, gap in enumerate(gap_list):
            for feature_indices_name_index, feature_indices_name in enumerate(feature_indices_name_list):
                print(' ')
                print(' ')
                print('**********************************************')
                print(' ')
                print('Working on %s, kS_%02d, kB_%02d, gap_%02d, %s' % (training_set_id, kS, kB, gap, feature_indices_name))

                # Build consolidated filtered features dataset filename

                filtered_features_dataset_root_path = os.path.join(base_dir, 'filtered_features_datasets')
                filtered_features_dataset_path  = os.path.join(filtered_features_dataset_root_path, training_set_id)
                filtered_features_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' % (training_set_id, kS, kB, gap, feature_indices_name)
                filtered_features_fullfilename = os.path.join(filtered_features_dataset_path, filtered_features_filename)

                # read in the file in the filtered features file
                # does doesn't have the final target values files

                if not os.path.exists(filtered_features_fullfilename):
                    print('Does not exist. Skipping: {}'.format(filtered_features_fullfilename))
                    continue

                with h5py.File(filtered_features_fullfilename, 'r') as f:
                    if load_subset:
                        X = f['X'][:number_instance_load,:]
                        y = f['y'][:number_instance_load,:]
                        speed = f['speed'][:number_instance_load]
                        distance_closest_approach = f['distance_closest_approach'][:number_instance_load]
                        signal_sigma = f['signal_sigma'][:number_instance_load]

                    else:
                        X = f['X'].value
                        y = f['y'].value
                        speed = f['speed'].value
                        distance_closest_approach = f['distance_closest_approach'].value
                        signal_sigma = f['signal_sigma'].value

                # input_size = X.shape[1]
                # output_size = y_matrix.shape[1]
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

                # this the y with the new isotope mapping
                y_new = np.zeros(y.shape[0], dtype = np.int16)

                # count_threshold_fraction_max_counts = 0.10
                for instance_index in xrange(y.shape[0]):
                    if y[instance_index,:].sum() > count_threshold:
                        y_new[instance_index] = isotope_mapping[np.argmax(y[instance_index, :])]

                indices = np.where(np.diff(y_new.astype(int)) > 0)[0]

                y_new_buffered = copy.deepcopy(y_new)
                for index in indices:
                    y_new_buffered[(index + 1):(index + 1 + acq_buffer_size[kS])] = y_new_buffered[index + 1]
                # now use y_new_buffered for almost everything

                # temporary - go back to it for comparison
                y_new_buffered = y_new

                # Balance the classes (less background)

                # determine the number of
                number_background_instances_include = 100 * int(np.bincount(y_new_buffered)[1:].mean()/100) * background_ratio
                # find all instances that are class is 0 (background)
                cutt = y_new_buffered == 0
                indices = np.where(cutt)[0]

                # randomly shuffle the indices so that is random which ones we keep and throw out.  Before it was
                # keep the ones in the beginning
                np.random.shuffle(indices)

                cutt = np.zeros(len(y_new_buffered)) == 1
                cutt[indices[number_background_instances_include:]] = True

                # this is the maske of all instances that we will use for training and testing
                masks = {}

                masks['balanced'] = ~cutt

                # designate half of the training set for drive by testing
                # find the halfway point and round up to number_acquisitions_save so that it doesn't cut into the middle of
                # a drive by.
                middle_index = X.shape[0]/ 2
                masks['training'] = np.ones(X.shape[0]) == 0
                masks['training'][0:middle_index] = True
                masks['test'] = ~masks['training']

                for speed_index, speed_bounds in enumerate(speed_bounds_list):

                    masks['speed'] = (speed > speed_bounds[0]) & (speed <= speed_bounds[1])

                    print('Original Count of classes:')
                    print(np.bincount(y_new_buffered))
                    print('total number of instances: {}'.format(len(y_new_buffered)))

                    print('Count of classes after removing a lot of background instances:')
                    print(np.bincount(y_new_buffered[masks['balanced']]))
                    print('total number of instances: {}'.format(len(y_new_buffered[masks['balanced']])))

                    print('Count of classes in Training Set after removing a lot of background instances:')
                    print(np.bincount(y_new_buffered[masks['balanced'] & masks['training']]))
                    print('total number of instances: {}'.format(len(y_new_buffered[masks['balanced'] & masks['training']])))

                    print('Count of classes in Test Set after removing a lot of background instances:')
                    print(np.bincount(y_new_buffered[masks['balanced'] & masks['test']]))
                    print('total number of instances: {}'.format(len(y_new_buffered[masks['balanced'] & masks['test']])))


                    print('Count of classes in Training Set and speed range after removing a lot of background instances:')
                    print(np.bincount(y_new_buffered[masks['balanced'] & masks['training'] & masks['speed']]))
                    print('total number of instances: {}'.format(len(y_new_buffered[masks['balanced'] & masks['training'] & masks['speed']])))

                    print('Count of classes in Test Set and speed range after removing a lot of background instances:')
                    print(np.bincount(y_new_buffered[masks['balanced'] & masks['test'] & masks['speed']]))
                    print('total number of instances: {}'.format(len(y_new_buffered[masks['balanced'] & masks['test'] & masks['speed']])))

                    lb = preprocessing.LabelBinarizer()
                    lb.fit(y_new_buffered)
                    lb.classes_

                    y_matrix = lb.transform(y_new_buffered)

                    results_dict = {}

                    clf_all_dict = {key:[] for key in model_keys}
                    metrics_all_dict = {key:[] for key in model_keys}

                    # prediction for the test
                    prediction_cross_val_dict = {key:[] for key in model_keys}
                    # prediction for all
                    prediction_all_dict =  {key:[] for key in model_keys}

                    # prediction for all
                    background_threshold_dict =  {key:[] for key in model_keys}

                    prediction_prob_cross_val_dict =  {key:[] for key in model_keys}

                    # prediction for all
                    prediction_prob_all_dict =  {key:[] for key in model_keys}

                    true_positive_driveby_positive_window_dict = {key:[] for key in model_keys}
                    false_positive_driveby_positive_window_dict = {key:[] for key in model_keys}
                    false_positive_driveby_negative_window_dict = {key:[] for key in model_keys}

                    recall_positive_window_dict = {key:[] for key in model_keys}
                    incorrect_prediction_rate_positive_window_dict = {key:[] for key in model_keys}
                    false_positive_rate_negative_window_dict = {key:[] for key in model_keys}

                    prediction_driveby_positive_window_dict = {key:[] for key in model_keys}
                    prediction_driveby_negative_window_dict = {key:[] for key in model_keys}

                    confusion_matrix_dict = {key:[] for key in model_keys}

                    # the test and training portion of the non-driveby portion of the dataset
                    indices = {'test':[], 'train':[]}

                    # set test and training split
                    skf = StratifiedKFold(n_splits=2)

                    # this the portion that will be use for cross validation.  It does not include the drive by portion
                    masks['cross_val'] = masks['balanced'] & masks['training'] & masks['speed']

                    # Create 2d and 1d matrices including only instances in the subset with more balanced classes
                    # 2d matrix
                    y_matrix_cross_val = y_matrix[masks['cross_val'], :]

                    # this is 1d matrix
                    y_index_cross_val = np.argmax(y_matrix[masks['cross_val'], :], axis = 1)

                    split_index = 0

                    # cross validation indice generator generates indices in the subspace of 'cross_val'
                    for indices_test, indices_train in skf.split(X[masks['cross_val'], :], y_new_buffered[masks['cross_val']]):

                        indices['test'].append(indices_test)
                        indices['train'].append(indices_train)

                        for model_index, model_name in enumerate( model_keys ):

                            print(' ')
                            print(' ')

                            # random forest
                            if model_name == 'rf_0':
                                model = RandomForestClassifier(n_estimators = 100, random_state=0, verbose = 1, n_jobs = 4)

                            elif model_name == 'nn_2layer':
                                model = KerasClassifier(build_fn = create_neural_network_2layer_model,\
                                                        input_size = X.shape[1], output_size = y_matrix.shape[1], \
                                                        epochs = nn_epochs, batch_size = 10000, verbose = 1)

                            elif model_name == 'nn_3layer':
                                model = KerasClassifier(build_fn = create_neural_network_3layer_model,\
                                                        input_size = X.shape[1], output_size = y_matrix.shape[1], \
                                                        epochs = nn_epochs, batch_size = 10000, verbose = 1)

                            elif model_name == 'nn_4layer':
                                model = KerasClassifier(build_fn = create_neural_network_4layer_model,\
                                                        input_size = X.shape[1], output_size = y_matrix.shape[1], \
                                                        epochs = nn_epochs, batch_size = 10000, verbose = 1)

                            elif model_name == 'nn_4layer_no_dropout':
                                model = KerasClassifier(build_fn = create_neural_network_4layer_no_dropout_model,\
                                                        input_size = X.shape[1], output_size = y_matrix.shape[1], \
                                                        epochs = nn_epochs, batch_size = 10000, verbose = 1)

                            elif model_name == 'nn_5layer':
                                model = KerasClassifier(build_fn = create_neural_network_5layer_model, \
                                                        input_size = X.shape[1], output_size = y_matrix.shape[1], \
                                                        epochs = nn_epochs, batch_size = 10000, verbose = 1)

                            elif model_name == 'lr':
                                model = LogisticRegression()
                            elif model_name == 'lda':
                                model = LinearDiscriminantAnalysis()
                            elif model_name == 'gb':
                                model = GradientBoostingClassifier(n_estimators = 100, verbose = 1)
                            else:
                                continue
                            clf_all_dict[model_name].append(model)

                            print('Model Name: {}'.format(model_name))

                            # The cross_val_score does not work with neural network and also doesn't save models so
                            # don't use this for now.

                            # NOT neural network
                            if 'nn' not in model_name:

                                t0 = time.time()
                                # fit to the training indices
                                # y_index_cross_val is already in the 'cross_val' subspace
                                clf_all_dict[model_name][split_index].fit(X[masks['cross_val'], :][indices_train, :], \
                                                         y_index_cross_val[indices_train])
                                print('Time to fit: {}'.format(time.time() - t0))

                                t0 = time.time()

                                # predict only for the stuff in the balanced dataset
                                prediction_cross_val_dict[model_name].append( clf_all_dict[model_name][split_index].predict(X[masks['cross_val'],:]))
                                # predict for all acquisitions
                                prediction_all_dict[model_name].append( clf_all_dict[model_name][split_index].predict(X))

                                # prediction probabilities
                                prediction_prob_all_dict[model_name].append(clf_all_dict[model_name][split_index].predict_proba(X))

                                print('Time to predict: {}'.format(time.time() - t0))

                            # Neural Network
                            else:
                                t0 = time.time()
                                clf_all_dict[model_name][split_index].fit(X[masks['cross_val'], :][indices_train, :], \
                                                             y_matrix_cross_val[indices_train,:].astype(float))
                                print('Time to fit: {}'.format(time.time() - t0))

                                t0 = time.time()

                                # run predictions

                                # prediction probabilities
                                prediction_prob_all_dict[model_name].append(clf_all_dict[model_name][split_index].model.predict(X))
                                prediction_prob_cross_val_dict[model_name].append(clf_all_dict[model_name][split_index].model.predict(X[masks['cross_val'],:]))

                                # set predictions based on max probability value
                                prediction_all_dict[model_name].append( np.argmax(prediction_prob_all_dict[model_name][split_index], axis = 1) )

                                # set predictions based on max probability value
                                prediction_cross_val_dict[model_name].append( np.argmax(prediction_prob_cross_val_dict[model_name][split_index], axis = 1) )

                                print('Time to predict: {}'.format(time.time() - t0))

                            metrics = {}

                            metrics['accuracy'] = accuracy_score(y_index_cross_val[indices_test], prediction_cross_val_dict[model_name][split_index][indices_test])
                            metrics['f1'] = f1_score(y_index_cross_val[indices_test], prediction_cross_val_dict[model_name][split_index][indices_test], average = 'micro')
                            metrics['recall'] = recall_score(y_index_cross_val[indices_test], prediction_cross_val_dict[model_name][split_index][indices_test], average = 'micro')
                            metrics['precision'] = precision_score(y_index_cross_val[indices_test], prediction_cross_val_dict[model_name][split_index][indices_test],  average = 'micro')

                            metrics_all_dict[model_name].append(metrics)
                            # print out the metrics
                            for key in ['accuracy', 'recall', 'precision', 'f1']:
                                print('{}: {}'.format(key, metrics[key]))

                            # print confusion matrix
                            confusion_matrix_dict[model_name].append(  confusion_matrix( y_index_cross_val[indices_test], prediction_cross_val_dict[model_name][split_index][indices_test]  ) )

                            # print(confusion_matrix_dict[model_name][split_index])

                            # calculate the actual drive performance
                            # detector source
                            # misidentified source
                            # fall positive (in section leading up to the source

                            # inputs
                            # number_acquisitions_save
                            # negative_window
                            # positive_window

                            # Work on the drive by instances
                            # 9/14/2017
                            # It makes sense to work only on the stuff that hasn't been trained on which is why we
                            # are now going to work on the masks['test'] portion.

                            # only examine for instances in the speed range
                            masks['driveby'] = masks['test'] & masks['speed']


                            # the number of drive bys is the number of acquisitions in the test divided by the number of
                            # acquisitions saved per drive by
                            number_driveby_instances = masks['driveby'].sum()/number_acquisitions_save

                            # prediction_driveby = np.zeros(number_driveby_instances)
                            prediction_driveby_positive_window = []
                            truth_driveby_positive_window = np.zeros(number_driveby_instances)

                            prediction_driveby_negative_window = []
                            truth_driveby_negative_window = np.zeros(number_driveby_instances)

                            true_positive_driveby = np.zeros(number_driveby_instances)

                            true_positive_driveby_positive_window = np.zeros(number_driveby_instances).astype(bool)
                            false_positive_driveby_positive_window = np.zeros(number_driveby_instances).astype(bool)
                            false_positive_driveby_negative_window = np.zeros(number_driveby_instances).astype(bool)

                            # the prediction in the test drive bys

                            # get particular trained model of particular cross validation split
                            prediction_arg_max = prediction_all_dict[model_name][masks['driveby'][split_index]]

                            # truth in the test drive by
                            y_new_buffered_test = y_new_buffered[masks['driveby']]


                            for drive_by_index in xrange(number_driveby_instances):

                                # start and stop of the drive by, in the test drive by subspace
                                start_1 = drive_by_index * number_acquisitions_save
                                stop_1 = start_1 + number_acquisitions_save

                                # get the truth and prediction in the window near the source (positive window)
                                truth_driveby_positive_window[drive_by_index] = np.max(y_new_buffered_test[start_1:stop_1][positive_window[0]:positive_window[1]] )
                                prediction_driveby_positive_window.append(Counter(prediction_arg_max[start_1:stop_1][positive_window[0]:positive_window[1]]))

                                # get the truth and prediction in the window before the source (negative window)
                                truth_driveby_negative_window[drive_by_index] = np.max(y_new_buffered_test[start_1:stop_1][negative_window[0]:negative_window[1]])
                                prediction_driveby_negative_window.append(Counter(prediction_arg_max[start_1:stop_1][negative_window[0]:negative_window[1]]))

                                # get the isotope index of isotopes predicted in the positive window
                                prediction_keys = prediction_driveby_positive_window[drive_by_index].keys()
                                # It's true positive if we detect the source at least once
                                true_positive_driveby_positive_window[drive_by_index] = truth_driveby_positive_window[drive_by_index] in prediction_keys

                                # set of prediction labels
                                temp1 = set(prediction_keys)
                                # set of truth labels
                                # - append 0 because we don't consider predictions as no-source to be necessarily wrong
                                temp2 = set([truth_driveby_positive_window[drive_by_index], 0.0])

                                # temp1.difference(temp2) = elements in temp1 but not in temp2
                                # if we predict isotopes that are not in the truth (temp2) then it is considered a
                                # false prediction
                                incorrect_predictions_positive_windows = list(temp1.difference(temp2))

                                # boolean on whether or not there was a false positive in the source window
                                false_positive_driveby_positive_window[drive_by_index] = len(incorrect_predictions_positive_windows) > 0

                                # let's examine the negative window - before the source
                                prediction_keys_negative_window = prediction_driveby_negative_window[drive_by_index].keys()

                                # set of prediction labels
                                temp1 = set(prediction_keys_negative_window)
                                # set of truth labels
                                # - append 0 because we don't consider predictions as no-source to be necessarily wrong
                                temp2 = set([truth_driveby_negative_window[drive_by_index], 0.0])

                                incorrect_predictions_negative_windows = list(temp1.difference(temp2))

                                false_positive_driveby_negative_window[drive_by_index] = len(incorrect_predictions_negative_windows) > 0

                            recall_positive_window = true_positive_driveby_positive_window.sum() / float(len(true_positive_driveby_positive_window))
                            incorrect_prediction_rate_positive_window = false_positive_driveby_positive_window.sum() / float(len(false_positive_driveby_positive_window))
                            false_positive_rate_negative_window = false_positive_driveby_negative_window.sum() / float(len(false_positive_driveby_negative_window))

                            print('drive by recall_positive_window: {}'.format(recall_positive_window))
                            print('drive by incorrect_prediction_rate_positive_window: {}'.format(incorrect_prediction_rate_positive_window))
                            print('drive by false_positive_rate_negative_window: {}'.format(false_positive_rate_negative_window))

                            # save the values
                            true_positive_driveby_positive_window_dict[model_name].append(true_positive_driveby_positive_window)
                            false_positive_driveby_positive_window_dict[model_name].append(false_positive_driveby_positive_window)
                            false_positive_driveby_negative_window_dict[model_name].append(false_positive_driveby_negative_window)

                            recall_positive_window_dict[model_name].append(recall_positive_window)
                            incorrect_prediction_rate_positive_window_dict[model_name].append(incorrect_prediction_rate_positive_window)
                            false_positive_rate_negative_window_dict[model_name].append(false_positive_rate_negative_window)

                            prediction_driveby_positive_window_dict[model_name].append(prediction_driveby_positive_window)
                            prediction_driveby_negative_window_dict[model_name].append(prediction_driveby_negative_window)
                        split_index += 1

                    # del X
                    # del y
                    # save stuff to file after training and testing each speed.

                    model_output_fullfilename = os.path.join(models_dataset_path,
                                                             filtered_features_filename.replace('FilteredFeaturesDataset', 'speed_%02d_%02d__Models' \
                                                                                                %(speed_bounds[0], speed_bounds[1])).replace('h5', 'pkl'))
                    with open(model_output_fullfilename, 'wb') as fid:
                        output = {}
                        output['models'] = clf_all_dict
                        cPickle.dump(output, fid, 2)

                    print('Wrote:{}'.format(model_output_fullfilename))

                    # for the neural networks, write each one separately
                    for model_name in clf_all_dict.keys():
                        if 'nn' in model_name:
                            # save just the neural networks

                            for model_index in xrange(len(clf_all_dict[model_name])):
                                nn_output_fullfilename = os.path.join(models_dataset_path,
                                                                         filtered_features_filename.replace('FilteredFeaturesDataset', 'speed_%02d_%02d__%s__%02d_%02d' \
                                                                            %(speed_bounds[0], speed_bounds[1], model_name, model_index, len(clf_all_dict[model_name]))) )
                                clf_all_dict[model_name][model_index].model.save(nn_output_fullfilename)
                                print('Wrote:{}'.format(nn_output_fullfilename))

                    modelmetrics_output_fullfilename = os.path.join(models_dataset_path,
                                                                    filtered_features_filename.replace('.h5', '.pkl').replace('FilteredFeaturesDataset', 'speed_%02d_%02d__ModelMetrics' \
                                                                                                %(speed_bounds[0], speed_bounds[1]))  )
                    with open(modelmetrics_output_fullfilename, 'wb') as fid:
                        output = {}
                        output['speed_bounds'] = speed_bounds
                        output['metrics'] = metrics_all_dict
                        output['prediction_cross_val_dict'] = prediction_cross_val_dict
                        output['prediction_all_dict'] = prediction_all_dict
                        output['prediction_prob_all_dict'] = prediction_prob_all_dict

                        output['indices'] = indices

                        output['masks'] = masks

                        output['y_new'] = y_new
                        output['y_new_buffered'] = y_new_buffered

                        output['y_matrix_cross_val'] = y_matrix_cross_val
                        output['y_index_cross_val'] = y_index_cross_val

                        output['true_positive_driveby_positive_window_dict'] = true_positive_driveby_positive_window_dict
                        output['false_positive_driveby_positive_window_dict'] = false_positive_driveby_positive_window_dict
                        output['false_positive_driveby_negative_window_dict'] = false_positive_driveby_negative_window_dict

                        output['recall_positive_window_dict'] = recall_positive_window_dict
                        output['incorrect_prediction_rate_positive_window_dict'] = incorrect_prediction_rate_positive_window_dict
                        output['false_positive_rate_negative_window_dict'] = false_positive_rate_negative_window_dict
                        output['truth_driveby_positive_window'] = truth_driveby_positive_window

                        output['prediction_driveby_positive_window_dict'] = prediction_driveby_positive_window_dict
                        output['prediction_driveby_negative_window_dict'] = prediction_driveby_negative_window_dict

                        # this is same for all models since truth value
                        output['truth_driveby_positive_window'] = truth_driveby_positive_window
                        output['truth_driveby_negative_window'] = truth_driveby_negative_window

                        output['number_acquisitions_save'] = number_acquisitions_save
                        output['negative_window'] = negative_window
                        output['positive_window'] = positive_window

                        cPickle.dump(output, fid, 2)
                    print('Wrote:{}'.format(modelmetrics_output_fullfilename))

                # save the file containing stuff at


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
