
import os, sys, glob, time
import h5py, cPickle
import copy
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
from scipy.stats import pearsonr

plot_markers = ''


def create_neural_network_5layer_model(input_size, output_size):
# def create_neural_network_5layer_model():

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

# def create_neural_network_4layer_model():
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
models_dataset_path = os.path.join(models_dataset_root_path, '2')


if not os.path.exists(models_dataset_root_path):
    os.mkdir(models_dataset_root_path)
if not os.path.exists(models_dataset_path):
    os.mkdir(models_dataset_path)


# lost of parameters to cycle through
kS_list = [2, 4, 8]

kB_list = [16]
gap_list = [4]
feature_indices_name_list = ['mask_filtered_features_2', 'mask_filtered_features_3']

# list of algorithms
# model_keys = ['lr', 'lda']
model_keys = ['lda', 'nn_4layer']



# # number to load from file
load_subset = True
# load_subset = False
number_instance_load = 2000

# number of values to use in the cross validation
# number_training_instances_cross_val = 10000

count_threshold = 100

# the uuid of the training dataset
training_set_id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'

training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_filename = '%s__%03d__TrainingDataset.h5' % (training_set_id, 0)
training_dataset_fullfilename = os.path.join(training_dataset_path, training_dataset_filename)

positive_buffer_value = 0


X_dict = {ks:{ kB:{ gap:{} for gap in gap_list      } for kB in kB_list    } for ks in kS_list }
y_dict = {ks:{ kB:{ gap:{} for gap in gap_list      } for kB in kB_list    } for ks in kS_list }
y_matrix_dict = {ks:{ kB:{ gap:{} for gap in gap_list      } for kB in kB_list    } for ks in kS_list }


for kS_index, kS in enumerate(kS_list):
    for kB_index, kB in enumerate(kB_list):
        for gap_index, gap in enumerate(gap_list):
            for feature_indices_name_index, feature_indices_name in enumerate(feature_indices_name_list[:1]):

                print(' ')
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
                    if load_subset:
                        X = f['X'][:number_instance_load,:]
                        y = f['y'][:number_instance_load,:]
                    else:
                        X = f['X'].value
                        y = f['y'].value

                X_dict[kS][kB][gap] = X+0
                y_dict[kS][kB][gap] = y+0

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

                # Balance the classes (less background)

                # determine the number of
                number_background_instances_include = 100 * int(np.bincount(y_new)[1:].mean()/100)
                # find all instances that are class is 0 (background)
                cutt = y_new == 0
                indices = np.where(cutt)[0]
                cutt = np.zeros(len(y_new)) == 1
                cutt[indices[number_background_instances_include:]] = True

                # this is the maske of all instances that we will use for training and testing
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

                y_matrix_dict[kS][kB][gap] = lb.transform(y_new)



                #
                # # kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)
                #
                # results_dict = {}
                #
                # # Create 2d and 1d matrices including only instances in the subset with more balanced classes
                # # 2d matrix
                # y_matrix_mask_include = y_matrix[mask_include, :]
                # # this is 1d matrix
                # y_index_mask_include = np.argmax(y_matrix[mask_include, :], axis = 1)
                #
                # clf_all_dict = {key:[] for key in model_keys}
                # metrics_all_dict = {key:[] for key in model_keys}
                # prediction_dict =  {key:[] for key in model_keys}
                #
                # prediction_all_dict =  {key:[] for key in model_keys}
                #
                # true_positive_driveby_positive_window_dict = {key:[] for key in model_keys}
                # false_positive_driveby_positive_window_dict = {key:[] for key in model_keys}
                # false_positive_driveby_negative_window_dict = {key:[] for key in model_keys}
                #
                # recall_positive_window_dict = {key:[] for key in model_keys}
                # incorrect_prediction_rate_positive_window_dict = {key:[] for key in model_keys}
                # false_positive_rate_negative_window_dict = {key:[] for key in model_keys}
                #
                # confusion_matrix_dict =  {key:[] for key in model_keys}
                #
                # indices = {'test':[], 'train':[]}

# figure out which wavelet bins to plot
plt.figure()

plt.pcolor(X_dict[kS][kB][gap][:300,:])

# create new y_new's

y_new_dict = {}
buffer_size = {2:4, 4:6, 8:6}
indices = np.where(np.diff(y_new.astype(int)) > 0)[0]
for kS in kS_list:
    y_new_dict[kS] = copy.deepcopy(y_new)
    for index in indices:
        y_new_dict[kS][(index+1):(index+1 + buffer_size[kS])] = y_new_dict[kS][index+1]


# plot the old and new y
plt.figure();
plt.grid()
plt.plot(y_new, label = "Original")
for kS in kS_list:
    plt.plot(y_new_dict[kS], alpha = kS, label = 'kS: {}'.format(kS))

plt.xlabel('Instance Index')
plt.ylabel('Isotope Index')
plt.legend()

#  plot the old and new y over the

wavelet_bounds = [210, 250]
plt.figure()
plt.grid()
for kS in np.sort(X_dict.keys()):
    for kB in np.sort( X_dict[kS].keys()):
        for gap in np.sort(X_dict[kS][kB].keys()):

            plt.plot(X_dict[kS][kB][gap][:,wavelet_bounds[0]:wavelet_bounds[1]].sum(1), label = 'kS: {}, kB: {}, gap: {}'.format(kS, kB, gap))

plt.plot(y[0:2000,:].sum(1), label = 'Target Count Value')

plt.plot(y_new*10, label = "Original")
for kS in kS_list:
    plt.plot(y_new_dict[kS]*10, alpha = kS, label = 'kS: {}'.format(kS))


plt.plot(np.argmax(y_matrix_dict[kS][kB][gap], axis=1))
plt.xlabel('Instance Index')
plt.ylabel('Wavelegth Subset Sum')
plt.legend()




wavelet_bounds = [210, 215]
plt.figure()
plt.grid()
for kS in np.sort(X_dict.keys()):
    for kB in np.sort( X_dict[kS].keys()):
        for gap in np.sort(X_dict[kS][kB].keys()):

            plt.plot(X_dict[kS][kB][gap][:,wavelet_bounds[0]:wavelet_bounds[1]].sum(1), label = 'kS: {}, kB: {}, gap: {}'.format(kS, kB, gap))

plt.plot(y[0:2000,:].sum(1), label = 'Target Count Value')

plt.plot(np.argmax(y_matrix_dict[kS][kB][gap], axis=1))
plt.xlabel('Instance Index')
plt.ylabel('Wavelegth Subset Sum')
plt.legend()