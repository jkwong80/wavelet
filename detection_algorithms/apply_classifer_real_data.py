

import os, sys, glob, time
import h5py, cPickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

# assumes that you are at the base of the repo
sys.path.append('common')


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

lineStyles = ['-', '--', '-.', ':']
markerTypes = ['.', '*', 'o', 'd', 'h', 'p', 's', 'v', 'x']
plotColors = ['k', 'r', 'g', 'b', 'm', 'y', 'c'] * 10

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


snr_root_path = os.path.join(base_dir, 'snr_functions')

if not os.path.exists(snr_root_path):
    os.mkdir(snr_root_path)

snr_path = os.path.join(snr_root_path, '20170824')

real_data_root_path = os.path.join(base_dir, 'real_data')
real_data_processed_root_path = os.path.join(base_dir, 'real_data_processed')

if not os.path.exists(real_data_processed_root_path):
    os.mkdir(real_data_processed_root_path)

print(real_data_processed_root_path)




# lost of parameters to cycle through
kS_list = [2, 4, 8]
kB_list = [16, 32]
gap_list = [4, 8, 16]
feature_indices_name_list = ['mask_filtered_features_2', 'mask_filtered_features_3']

model_keys = ['lda', 'nn_4layer', 'lr_0']


kS = 4
kB = 16
gap = 8
feature_indices_name = feature_indices_name_list[0]
# list of algorithms
# model_keys = ['lr', 'lda']

model_name = 'rf_0'

# # number to load from file
load_subset = True
# load_subset = False
number_instance_load = 500000

# number of values to use in the cross validation
# number_training_instances_cross_val = 10000

count_threshold = 100

# the uuid of the training dataset
training_set_id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'

training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)
training_dataset_filename = '%s__%03d__TrainingDataset.h5' % (training_set_id, 0)
training_dataset_fullfilename = os.path.join(training_dataset_path, training_dataset_filename)

positive_buffer_value = 0

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

if not os.path.exists(filtered_features_fullfilename):
    print('Does not exist. Skipping: {}'.format(filtered_features_fullfilename))

with h5py.File(filtered_features_fullfilename, 'r') as f:
    if load_subset:
        X = f['X'][:number_instance_load,:]
        y = f['y'][:number_instance_load,:]
    else:
        X = f['X'].value
        y = f['y'].value
#
# X_dict[kS][kB][gap] = X+0
# y_dict[kS][kB][gap] = y+0

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

snr_name_suffix = 'kS_%02d__kB_%02d__gap_%02d' % (kS, kB, gap)

snr_fullfilename = os.path.join(snr_path, 'f_snr__%s.pkl' % snr_name_suffix)

print('Loadings snr: {}'.format(snr_fullfilename))
with open(snr_fullfilename, 'rb') as fid:
    f_snr = cPickle.load(fid)



# let's load the snr function for this


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

# y_matrix_dict[kS][kB][gap] = lb.transform(y_new)

model_name = 'rf_0'

# load the classifier
output_fullfilename = os.path.join(models_dataset_path,
                                   filtered_features_filename.replace('FilteredFeaturesDataset', 'Models'))
with open(output_fullfilename, 'rb') as fid:
    model = cPickle.load(fid)


# let's look at the random forest

new_prediction= model['models'][model_name][0].predict_proba(X)

new_prediction1 = np.argmax(new_prediction, axis =1)

background_probability_threshold = 0.05
new_prediction1[new_prediction[:,0]>background_probability_threshold] = 0

old_prediction = copy.deepcopy(model['prediction_all_dict'][model_name][0])

# example the random forest performance


display_limit = [0,5000]
# display_limit = [10000,15000]
display_limit = [0,50000]

plt.figure()
plt.grid()
mask_error = model['y_new_buffered'] != model['prediction_all_dict'][model_name][0]
plt.plot(np.arange(display_limit[0], display_limit[1]), model['y_new_buffered'][display_limit[0]:display_limit[1]], label = 'Truth')
plt.plot(np.arange(display_limit[0], display_limit[1]), model['prediction_all_dict'][model_name][0][display_limit[0]:display_limit[1]], label = '{} Prediction'.format(model_name))

plt.plot(np.arange(display_limit[0], display_limit[1])[mask_error[display_limit[0]:display_limit[1]]],\
         mask_error[display_limit[0]:display_limit[1]][mask_error[display_limit[0]:display_limit[1]]], 'r*')

plt.xlabel('Acquisition Index')
plt.ylabel("Prediction, Truth")
plt.legend()


# plot of the old prediction and new one

plt.figure()
plt.grid()
plt.plot(np.arange(display_limit[0], display_limit[1]), model['y_new_buffered'][display_limit[0]:display_limit[1]], label = 'Truth')
# plt.plot(np.arange(display_limit[0], display_limit[1]), new_prediction1[display_limit[0]:display_limit[1]], label = 'New Prediction')
plt.plot(np.arange(display_limit[0], display_limit[1]), old_prediction[display_limit[0]:display_limit[1]], label = '{} Prediction'.format(model_name))

plt.plot(np.arange(display_limit[0], display_limit[1])[mask_error[display_limit[0]:display_limit[1]]],\
         mask_error[display_limit[0]:display_limit[1]][mask_error[display_limit[0]:display_limit[1]]], 'r*')

plt.xlabel('Acquisition Index')
plt.ylabel("Prediction, Truth")
plt.legend()

plt.xlim((13000, 14000))


# plot the truth and new prediction values
plt.figure()
plt.grid()
plt.plot(np.arange(display_limit[0], display_limit[1]), model['y_new_buffered'][display_limit[0]:display_limit[1]], '-b', label = 'Truth', linewidth = 2, alpha = 0.7)
plt.plot(np.arange(display_limit[0], display_limit[1]), new_prediction1[display_limit[0]:display_limit[1]], '--r', label = 'New Prediction, {}'.format(model_name), linewidth = 2, alpha = 0.7)
# plt.plot(np.arange(display_limit[0], display_limit[1]), old_prediction[display_limit[0]:display_limit[1]], label = '{} Prediction'.format(model_name))

plt.plot(np.arange(display_limit[0], display_limit[1])[mask_error[display_limit[0]:display_limit[1]]],\
         mask_error[display_limit[0]:display_limit[1]][mask_error[display_limit[0]:display_limit[1]]], 'r*')

plt.xlabel('Acquisition Index', fontsize = 18)
plt.ylabel("Prediction, Truth", fontsize = 18)
plt.legend()

plt.xlim((13000, 14000))


# false_positive_rate

model_name = 'nn_4layer'
nn_output_old = copy.deepcopy(model['prediction_all_dict'][model_name][0])
nn_prediction_old = np.argmax(nn_output_old, axis =1)
nn_prediction_new = np.argmax(nn_output_old, axis =1)
truth = model['y_new_buffered']

background_probability_threshold = 0.04
nn_prediction_new[nn_output_old[:,0]>background_probability_threshold] = 0



fp_old = (nn_prediction_old != truth) & (truth == 0)
fp_new = (nn_prediction_new != truth) & (truth == 0)

print('Number of fp old: {}'.format(fp_old.sum()))

print('Number of fp new: {}'.format(fp_new.sum()))


display_limit = [0,5000]
display_limit = [10000,15000]


plt.figure(figsize = [30, 10])
plt.grid()

mask_error = truth != nn_prediction_old

plt.plot(np.arange(display_limit[0], display_limit[1]), truth[display_limit[0]:display_limit[1]], label = 'Truth')
plt.plot(np.arange(display_limit[0], display_limit[1]), nn_prediction_old[display_limit[0]:display_limit[1]], label = '{} Prediction'.format(model_name))

plt.plot(np.arange(display_limit[0], display_limit[1])[mask_error[display_limit[0]:display_limit[1]]],\
         mask_error[display_limit[0]:display_limit[1]][mask_error[display_limit[0]:display_limit[1]]], 'r*')

plt.xlabel('Acquisition Index')
plt.ylabel("Prediction, Truth")
plt.legend()
plt.title(model_name)

plt.figure(figsize = [30, 10])
plt.grid()

mask_error = truth != nn_prediction_old

plt.plot(np.arange(display_limit[0], display_limit[1]), truth[display_limit[0]:display_limit[1]], label = 'Truth')
plt.plot(np.arange(display_limit[0], display_limit[1]), nn_prediction_new[display_limit[0]:display_limit[1]], label = '{} Prediction'.format(model_name))

plt.plot(np.arange(display_limit[0], display_limit[1])[mask_error[display_limit[0]:display_limit[1]]],\
         mask_error[display_limit[0]:display_limit[1]][mask_error[display_limit[0]:display_limit[1]]], 'r*')

plt.xlabel('Acquisition Index')
plt.ylabel("Prediction, Truth")
plt.legend()
plt.title(model_name)



##########################

number_bins = 512
number_instances_process = 50000

filename_header_list = ['RebinReadings_2017-04-14T19-55-54Z', 'RebinReadings_2017-04-25T15-32-24Z', 'RebinReadings_2017-04-26T14-29-40Z']

filename_header = filename_header_list[1]

fullfilename = os.path.join(real_data_root_path, '{}.h5'.format(filename_header))

with h5py.File(fullfilename, 'r') as dat:
    try:
        spectra = dat['sensor_150_degrees']['spectra'].value
    except:
        spectra = dat['sensor_150_degrees']['adc_channel_counts'].value

mask_filtered_features_name = 'mask_filtered_features_2'
# load the feature_list =

# # this is the dict key of the feature indices you wan from the feature selection file
#
# feature_selection_path = os.path.join(base_dir, 'feature_selection')

# feature_selection_filename = '5b178c11-a4e4-4b19-a925-96f27c49491b__kS_02__kB_16__gap_04__top_features.h5'
# feature_selection_fullfilename = os.path.join(feature_selection_path, feature_selection_filename)

# with h5py.File(feature_selection_fullfilename, 'r') as f:
#     mask_filtered_features = f[mask_filtered_features_name].value
# print('Keeping {} features'.format(sum(mask_filtered_features)))

kS = 2
kB = 16
gap = 4

filename = '%s__kS_%02d__kB_%02d__gap_%02d__%s.h5' % (filename_header, kS, kB, gap, mask_filtered_features_name)

fullfilename = os.path.join(real_data_processed_root_path, filename)

print('Loading: {}'.format(fullfilename))
with h5py.File(fullfilename, 'r') as f:
    SNR_filtered_matrix = f['SNR_filtered'].value
    t = f['t'].value

# load the file

prediction = {}

cross_val_index = 0


for model_name in model['models'].keys():
    prediction[model_name] = {}

    if model_name == 'nn_4layer':
        prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
        prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis =1)
        prediction[model_name]['background_probability_threshold'] = 0.02
        prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis =1)
        prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

    elif model_name == 'rf_0':
        prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].predict_proba(SNR_filtered_matrix)
        prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis =1)
        prediction[model_name]['background_probability_threshold'] = 0.03
        prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis =1)
        prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

    elif model_name == 'lda':
        prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].predict_proba(SNR_filtered_matrix)
        prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis =1)
        # prediction[model_name]['background_probability_threshold'] = 0.03
        prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis =1)
        # prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

#
#
#
#
# model_name = 'nn_4layer'
#
# nn_output_new_data = model['models']['nn_4layer'][0].model.predict(SNR_filtered_matrix)
# nn_prediction_new_data= np.argmax(nn_output_new_data, axis =1)
# lda_prediction = model['models']['lda'][0].predict(SNR_filtered_matrix)
# background_probability_threshold = 0.02
# nn_prediction_new_data[nn_output_new_data[:,0]>background_probability_threshold] = 0
#
#
# model_name = 'rf_0'
# rf_new_prediction= model['models'][model_name][0].predict_proba(SNR_filtered_matrix)
# rf_new_prediction1 = np.argmax(rf_new_prediction, axis = 1)
# rf_background_probability_threshold = 0.03
# rf_new_prediction1[rf_new_prediction[:,0]>rf_background_probability_threshold] = 0
#
#
# model_name = 'lda'
# lda_new_prediction = model['models'][model_name][0].predict_proba(SNR_filtered_matrix)
# lda_new_prediction1 = np.argmax(rf_new_prediction, axis = 1)
# # lda_background_probability_threshold = 0.03
# # lda_new_prediction1[rf_new_prediction[:,0]>rf_background_probability_threshold] = 0


# plot of the random fore

plt.figure()
plt.grid()
plt.plot(spectra.sum(1), label = 'Total Counts')
# plt.plot(nn_prediction_new_data*100, '--b', label = 'Prediction, nn_4layer', alpha = 0.6)

marker_types = ['.', 's', 'd']

for model_name_index, model_name in enumerate(model['models'].keys()):

    tally = Counter(prediction[model_name]['prediction'])
    instance_index = np.arange( prediction[model_name]['prediction'].shape[0])

    plot_index = 0
    for index, key in enumerate(tally.keys()):

        if key == 0:
            continue
        cutt = prediction[model_name]['prediction'] == key

        if key == 0:
            isotope = 'background'
        else:
            isotope = isotope_string_set[key-1]

        if plot_index >= 7:
            markeredgecolor = 'k'
        else:
            markeredgecolor = plotColors[plot_index]

        plt.plot(instance_index[cutt], 1100*np.ones_like(instance_index)[cutt] + plot_index*10 + model_name_index*200, \
                 linestyle = 'none', marker = marker_types[model_name_index], \
                 markersize = 10, color = plotColors[plot_index], markeredgecolor= markeredgecolor,
                 alpha = 0.4, label = '{} Prediction {}'.format(model_name, isotope), mew = 5)
        plot_index += 1
plt.legend()
plt.xlabel('Acquisition Index')



isotope_string_with_background_set = np.array(['background'] + list(isotope_string_set))

# plot the probabilities

isotopes_plot = ['background', 'Am241', 'Ba133', 'Co57', 'I131', 'RGPu']

for model_name_index, model_name in enumerate(model['models'].keys()):

    plt.figure()
    plt.grid()
    plot_index = 0
    for isotope_index,isotope_name in enumerate(isotopes_plot):

        isotope_name_index = np.where(isotope_name == isotope_string_with_background_set)[0][0]

        isotope = isotope_string_with_background_set[isotope_name_index]

        if plot_index >= 7:
            markeredgecolor = 'k'
        else:
            markeredgecolor = plotColors[plot_index]

        plt.plot(prediction[model_name]['prob'][:,isotope_index], linestyle = '-',\
                 color = plotColors[plot_index],
                 alpha = 0.7, label = '{} Prediction {}'.format(model_name, isotope))
        plot_index += 1
    plt.legend()
    plt.xlabel('Acquisition Index')
    plt.ylabel('Probability')
    plt.title(model_name)
    plt.xlim(a)


plt.figure()
plt.pcolor(nn_output_new_data)
plt.colorbar()


# plot
plt.figure()
plt.grid()
plt.plot(spectra.sum(1), label = 'Total Counts')

plt.plot(rf_new_prediction1*100, '--r', label = 'Prediction, rf_0', alpha = 0.6)
plt.legend()
plt.xlabel('Acquisition Index')

