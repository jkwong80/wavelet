
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

sys.path.append('detection_algorithms')

# from training_dataset_processor.training_dataset_processor import GetInjectionResourcePaths, GetSourceMapping
import training_dataset_processor.training_dataset_processor
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
# from scipy.stats import pearsonr


from nn_models import create_neural_network_2layer_model, create_neural_network_3layer_model, create_neural_network_4layer_model,\
    create_neural_network_4layer_no_dropout_model, create_neural_network_5layer_model


plot_markers = ''

lineStyles = ['-', '--', '-.', ':']*5
markerTypes = ['.', '*', 'o', 'd', 'h', 'p', 's', 'v', 'x']*20
plotColors = ['k', 'r', 'g', 'b', 'm', 'y', 'c'] * 10

# def create_neural_network_5layer_model(input_size, output_size):
# # def create_neural_network_5layer_model():
#
#     print('input_size: '.format(input_size))
#     print('output_size: '.format(output_size))
#
#     model = Sequential()
#
#     model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(400, init='uniform', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(200, init='uniform', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(100, init='uniform', activation='relu'))
#
#     # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
#     model.add(Dense(output_size, init='uniform', activation='relu'))
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
#     return model
#
# # def create_neural_network_4layer_model():
# def create_neural_network_4layer_model(input_size, output_size):
#
#     print('input_size: '.format(input_size))
#     print('output_size: '.format(output_size))
#     model = Sequential()
#
#     model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(200, init='uniform', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(100, init='uniform', activation='relu'))
#
#     # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
#     model.add(Dense(output_size, init='uniform', activation='relu'))
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
#     return model


# open a couple files and append the results

paths = training_dataset_processor.training_dataset_processor.GetInjectionResourcePaths()
base_dir = paths['base']
if 'snr' not in paths:
    paths['snr'] = os.path.join(paths['snr_root'], '20170824')


real_data_subpath = '20170908'
real_data_processed_subpath = '20170908_8'
paths['real_data'] = os.path.join(paths['real_data_root'], real_data_subpath)
paths['real_data_processed'] = os.path.join(paths['real_data_processed_root'], real_data_processed_subpath)

# # lost of parameters to cycle through
# kS_list = [2, 4, 8]
# kB_list = [16, 32]
# gap_list = [4, 8, 16]
# feature_indices_name_list = ['mask_filtered_features_2', 'mask_filtered_features_3']

# get source_name_list from one the recent training datasets
id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'

sources = training_dataset_processor.training_dataset_processor.GetSourceMapping(id)

#  Specify which model to load

mask_filtered_features_name = 'mask_filtered_features_3'
# kS = 4
# kB = 32
# gap = 8

kS = 2
kB = 16
gap = 4


models_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__Models.pkl' % (id, kS, kB, gap, mask_filtered_features_name)

# load the classifier

model_subpath = '8'
paths['models'] = os.path.join(paths['models_root'], model_subpath)
output_fullfilename = os.path.join(paths['models'],models_filename)

with open(output_fullfilename, 'rb') as fid:
    model = cPickle.load(fid)

##########################

save_plots = True

number_bins = 512


spectra_fullfilename_list = glob.glob(os.path.join(paths['real_data'], '*.hdf5'))
spectra_fullfilename_list.sort()
spectra_filename_list = [os.path.split(f)[-1] for f in spectra_fullfilename_list]

spectra_filename_header_list = [f.replace('.hdf5', '') for f in spectra_filename_list]

snr_filename_list = ['%s__kS_%02d__kB_%02d__gap_%02d__%s.h5' \
                     % (f, kS, kB, gap, mask_filtered_features_name)  for f in spectra_filename_header_list]

# fullfilename = os.path.join(paths['real_data_processed'], filename)
# snr_fullfilename_list = glob.glob(os.path.join(paths['real_data_processed'], '*.h5'))
# snr_filename_list = [os.path.split(f)[-1] for f in snr_fullfilename_list]

snr_fullfilename_list = [os.path.join(paths['real_data_processed'], f) for f in snr_filename_list]

snr_filename_header_list = [f.replace('.h5', '') for f in snr_filename_list]


# filename_header = filename_header_list[0]
# fullfilename = os.path.join(paths['real_data'], '{}.hdf5'.format(filename_header))

for fullfilename_index,  snr_fullfilename in enumerate(snr_fullfilename_list):

    snr_filename_header = snr_filename_header_list[fullfilename_index]
    snr_filename = snr_filename_list[fullfilename_index]
    spectra_fullfilename = spectra_fullfilename_list[fullfilename_index]

    # load the spectra
    with h5py.File(spectra_fullfilename, 'r') as dat:
        try:
            print(dat.keys())
            spectra = dat['88']['rebinned_spectra'].value
        except:
            spectra = dat['88']['adc_channel_counts'].value

    # read in the snr wavelet dataset
    print('Loading: {}'.format(snr_fullfilename))
    with h5py.File(snr_fullfilename, 'r') as f:
        SNR_filtered_matrix = f['SNR_filtered'].value
        t = f['t'].value

    # load the file
    prediction = {}
    cross_val_index = 0

    for model_name in model['models'].keys():
        prediction[model_name] = {}

        if model_name == 'nn_4layer':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 0.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

        elif model_name == 'nn_4layer_no_dropout':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 0.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0
        elif model_name == 'nn_2layer':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 0.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0
        elif model_name == 'nn_3layer':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 0.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0


        elif model_name == 'rf_0':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].predict_proba(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.03
            prediction[model_name]['background_probability_threshold'] = 0.2
            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

        elif model_name == 'lda':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].predict_proba(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.03
            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0


    # plot of the random f

    plt.figure(figsize = [20, 20])
    plt.grid()
    plt.plot(spectra.sum(1), label = 'Total Counts')
    # plt.plot(nn_prediction_new_data*100, '--b', label = 'Prediction, nn_4layer', alpha = 0.6)

    marker_types = ['.', 's', 'd', 'o', '<', '>', '^']

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
                isotope = sources['isotope_string_set'][key-1]

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
    plt.title(snr_filename)

    if save_plots:
        plt.savefig(os.path.join(paths['plot'], '%s__%s.png' %(snr_filename_header, 'prediction') ))
        plt.close()

    isotope_string_with_background_set = np.array(['background'] + list(sources['isotope_string_set']))

    # plot the probabilities

    isotopes_plot = ['background', 'Am241', 'Ba133', 'Co57', 'I131', 'RGPu', 'Na22', 'Cs137', 'Eu152']

    plt.figure(figsize=[30, 10])
    for model_name_index, model_name in enumerate(model['models'].keys()):
        plt.subplot(len(model['models'].keys()), 1, model_name_index+1)
        plt.grid()
        plot_index = 0
        for isotope_index,isotope_name in enumerate(isotopes_plot):

            isotope_name_index = np.where(isotope_name == isotope_string_with_background_set)[0][0]

            isotope = isotope_string_with_background_set[isotope_name_index]

            if plot_index >= 7:
                markeredgecolor = 'k'
            else:
                markeredgecolor = plotColors[plot_index]

            plt.plot(prediction[model_name]['prob'][:,isotope_name_index], linestyle = lineStyles[plot_index / 7],\
                     color = plotColors[plot_index],
                     alpha = 0.7, label = '{} Prediction {}'.format(model_name, isotope))
            plot_index += 1
        plt.legend(fontsize = 10)
        plt.xlabel('Acquisition Index')
        plt.ylabel('Probability')
        plt.title('{}, Probability Values'.format(snr_filename))
        # plt.xlim(a)

    if save_plots:
        plt.savefig(os.path.join(paths['plot'], '%s__%s.png' %(snr_filename_header, 'probability') ))
        plt.close()

