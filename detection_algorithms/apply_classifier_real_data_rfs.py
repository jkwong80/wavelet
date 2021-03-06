
from collections import Counter

import cPickle
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# assumes that you are at the base of the repo
sys.path.append('common')

sys.path.append('detection_algorithms')

# from training_dataset_processor.training_dataset_processor import GetInjectionResourcePaths, GetSourceMapping
import training_dataset_processor.training_dataset_processor

plot_markers = ''

lineStyles = ['-', '--', '-.', ':']*5
markerTypes = ['.', '*', 'o', 'd', 'h', 'p', 's', 'v', 'x']*20
plotColors = ['k', 'r', 'g', 'b', 'm', 'y', 'c'] * 10
marker_types = ['.', 's', 'd', 'o', '<', '>', '^']

# open a couple files and append the results

paths = training_dataset_processor.training_dataset_processor.GetInjectionResourcePaths()
base_dir = paths['base']
if 'snr' not in paths:
    paths['snr'] = os.path.join(paths['snr_root'], '20170824')

real_data_subpath = 'rfs'
real_data_processed_subpath = 'rfs'
paths['real_data'] = os.path.join(paths['real_data_root'], real_data_subpath)
paths['real_data_processed'] = os.path.join(paths['real_data_processed_root'], real_data_processed_subpath)

# # lost of parameters to cycle through
# kS_list = [2, 4, 8]
# kB_list = [16, 32]
# gap_list = [4, 8, 16]
# feature_indices_name_list = ['mask_filtered_features_2', 'mask_filtered_features_3']

# get source_name_list from one the recent training datasets
model_id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'

sources = training_dataset_processor.training_dataset_processor.GetSourceMapping(model_id)
#  Specify which model to load
mask_filtered_features_name = 'mask_filtered_features_3'
# kS = 4
# kB = 32
# gap = 8

kS = 2
kB = 16
gap = 4

# load the classifier
model_subpath = '8'

models_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__Models.pkl' % (model_id, kS, kB, gap, mask_filtered_features_name)
paths['models'] = os.path.join(paths['models_root'], model_subpath)
models_fullfilename = os.path.join(paths['models'],models_filename)
with open(models_fullfilename, 'rb') as fid:
    model = cPickle.load(fid)

##########################

save_plots = True

number_bins = 512

spectra_fullfilename_list = glob.glob(os.path.join(paths['real_data'], '*.hdf5')) + glob.glob(os.path.join(paths['real_data'], '*.h5'))

spectra_fullfilename_list.sort()
spectra_filename_list = [os.path.split(f)[-1] for f in spectra_fullfilename_list]

spectra_filename_header_list = [f.replace('.hdf5', '').replace('.h5', '') for f in spectra_filename_list]

snr_filename_list = ['%s__kS_%02d__kB_%02d__gap_%02d__%s.h5' \
                     % (f, kS, kB, gap, mask_filtered_features_name)  for f in spectra_filename_header_list]

# fullfilename = os.path.join(paths['real_data_processed'], filename)
# snr_fullfilename_list = glob.glob(os.path.join(paths['real_data_processed'], '*.h5'))
# snr_filename_list = [os.path.split(f)[-1] for f in snr_fullfilename_list]

snr_fullfilename_list = [os.path.join(paths['real_data_processed'], f) for f in snr_filename_list]

snr_filename_header_list = [f.replace('.h5', '') for f in snr_filename_list]


# filename_header = filename_header_list[0]
# fullfilename = os.path.join(paths['real_data'], '{}.hdf5'.format(filename_header))

detector_name = 'sensor_150_degrees'

for fullfilename_index,  snr_fullfilename in enumerate(snr_fullfilename_list):

    snr_filename_header = snr_filename_header_list[fullfilename_index]
    snr_filename = snr_filename_list[fullfilename_index]
    spectra_fullfilename = spectra_fullfilename_list[fullfilename_index]

    # load the spectra
    with h5py.File(spectra_fullfilename, 'r') as dat:
        try:
            print(dat.keys())
            if 'rebinned_spectra' in dat[detector_name]:
                spectra = dat[detector_name]['rebinned_spectra'].value
            elif 'spectrum' in dat[detector_name]:
                spectra = dat[detector_name]['spectrum'].value
            elif 'spectra' in dat[detector_name]:
                spectra = dat[detector_name]['spectra'].value

            else:
                print(dat[detector_name].keys())
        except:
            spectra = dat[detector_name]['adc_channel_counts'].value

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
            prediction[model_name]['background_probability_threshold'] = 1.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

        elif model_name == 'nn_4layer_no_dropout':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 1.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0
        elif model_name == 'nn_2layer':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 1.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0
        elif model_name == 'nn_3layer':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 1.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0
        elif model_name == 'nn_5layer':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].model.predict(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.02
            prediction[model_name]['background_probability_threshold'] = 1.20

            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

        elif model_name == 'rf_0':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].predict_proba(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.03
            prediction[model_name]['background_probability_threshold'] = 1.2
            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

        elif model_name == 'lda':
            prediction[model_name]['prob'] = model['models'][model_name][cross_val_index].predict_proba(SNR_filtered_matrix)
            prediction[model_name]['prediction0'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['background_probability_threshold'] = 0.03
            prediction[model_name]['prediction'] = np.argmax(prediction[model_name]['prob'], axis = 1)
            # prediction[model_name]['prediction'][prediction[model_name]['prob'][:,0] > prediction[model_name]['background_probability_threshold']] = 0

    # plot of the random f
    plt.figure(figsize = [35, 15])
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
        plt.savefig(os.path.join(paths['plot'], '%s__%s.pdf' %(snr_filename_header, 'prediction') ))
        plt.close()

    isotope_string_with_background_set = np.array(['background'] + list(sources['isotope_string_set']))

    # plot the probabilities

    isotopes_plot = ['background', 'Am241', 'Ba133', 'Co57', 'I131', 'RGPu', 'Na22', 'Cs137', 'Eu152']

    plt.figure(figsize=[35, 15])
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
        plt.savefig(os.path.join(paths['plot'], '%s__%s.pdf' %(snr_filename_header, 'probability') ))
        plt.close()

    # save segments
    segment_length = 5000
    segments = np.arange(0, spectra.shape[0], segment_length)
    if segments[-1] != (segment_length-1):
        segments = np.hstack((segments, [spectra.shape[0]]))

    for segmentIndex in xrange(len(segments)-1):
        segment_start = segments[segmentIndex]
        segment_stop = segments[segmentIndex+1]

        # plot of the random f
        plt.figure(figsize=[35, 15])
        plt.grid()
        plt.plot(spectra.sum(1), label='Total Counts')
        # plt.plot(nn_prediction_new_data*100, '--b', label = 'Prediction, nn_4layer', alpha = 0.6)


        for model_name_index, model_name in enumerate(model['models'].keys()):
            tally = Counter(prediction[model_name]['prediction'])
            instance_index = np.arange(prediction[model_name]['prediction'].shape[0])

            plot_index = 0
            for index, key in enumerate(tally.keys()):

                if key == 0:
                    continue
                cutt = prediction[model_name]['prediction'] == key

                if key == 0:
                    isotope = 'background'
                else:
                    isotope = sources['isotope_string_set'][key - 1]

                if plot_index >= 7:
                    markeredgecolor = 'k'
                else:
                    markeredgecolor = plotColors[plot_index]

                plt.plot(instance_index[cutt],
                         1100 * np.ones_like(instance_index)[cutt] + plot_index * 10 + model_name_index * 200, \
                         linestyle='none', marker=marker_types[model_name_index], \
                         markersize=10, color=plotColors[plot_index], markeredgecolor=markeredgecolor,
                         alpha=0.4, label='{} Prediction {}'.format(model_name, isotope), mew=5)
                plot_index += 1
        plt.legend()
        plt.xlabel('Acquisition Index')
        plt.title(snr_filename)
        plt.xlim((segment_start, segment_stop))

        if save_plots:
            plt.savefig(os.path.join(paths['plot'], '%s__%s__Segment_%06d_%06d.pdf' % (snr_filename_header, 'prediction', segment_start, segment_stop)))
            plt.close()

        isotope_string_with_background_set = np.array(['background'] + list(sources['isotope_string_set']))

        # plot the probabilities

        isotopes_plot = ['background', 'Am241', 'Ba133', 'Co57', 'I131', 'RGPu', 'Na22', 'Cs137', 'Eu152']

        plt.figure(figsize=[35, 15])
        for model_name_index, model_name in enumerate(model['models'].keys()):
            plt.subplot(len(model['models'].keys()), 1, model_name_index + 1)
            plt.grid()
            plot_index = 0
            for isotope_index, isotope_name in enumerate(isotopes_plot):

                isotope_name_index = np.where(isotope_name == isotope_string_with_background_set)[0][0]

                isotope = isotope_string_with_background_set[isotope_name_index]

                if plot_index >= 7:
                    markeredgecolor = 'k'
                else:
                    markeredgecolor = plotColors[plot_index]

                plt.plot(prediction[model_name]['prob'][:, isotope_name_index], linestyle=lineStyles[plot_index / 7], \
                         color=plotColors[plot_index],
                         alpha=0.7, label='{} Prediction {}'.format(model_name, isotope))
                plot_index += 1
            plt.legend(fontsize=10)
            plt.xlabel('Acquisition Index')
            plt.ylabel('Probability')
            plt.title('{}, Probability Values'.format(snr_filename))
            plt.xlim((segment_start, segment_stop))

        if save_plots:
            plt.savefig(os.path.join(paths['plot'], '%s__%s__Segment_%06d_%06d.pdf' % (snr_filename_header, 'probability', segment_start, segment_stop)))
            plt.close()



