"""Examine the performance of classifiers on training data.
Looking at the values in the file saved by train_classifiers_grid.py


"""

import os, sys, glob, time
sys.setrecursionlimit(10000)


import h5py, cPickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression

from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, accuracy_score

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

# assumes that you are at the base of the repo
sys.path.append('common')
sys.path.append('detection_algorithm')

# from training_dataset_processor.training_dataset_processor import GetInjectionResourcePaths, GetSourceMapping
import training_dataset_processor.training_dataset_processor


from detection_algorithms.nn_models import create_neural_network_2layer_model, create_neural_network_3layer_model, create_neural_network_4layer_model,\
    create_neural_network_4layer_no_dropout_model, create_neural_network_5layer_model

lineStyles = ['-', '--', '-.', ':']
markerTypes = ['.', '*', 'o', 'd', 'h', 'p', 's', 'v', 'x']
plotColors = ['k', 'r', 'g', 'b', 'm', 'y', 'c'] * 10

id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'


paths = training_dataset_processor.training_dataset_processor.GetInjectionResourcePaths()
base_dir = paths['base']
if 'snr' not in paths:
    paths['snr'] = os.path.join(paths['snr_root'], '20170824')
real_data_subpath = '20170908'
paths['real_data'] = os.path.join(paths['real_data_root'], real_data_subpath)
paths['real_data_processed'] = os.path.join(paths['real_data_processed_root'], real_data_subpath)

paths['filtered_features_datasets'] = os.path.join(paths['filtered_features_datasets_root'], id)

model_subpath = '9'
paths['models'] = os.path.join(paths['models_root'], model_subpath)


sources = training_dataset_processor.training_dataset_processor.GetSourceMapping(id)

# build sets

load_parameter_list = []
load_parameter_list.append( {'id':id, 'kS':2, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[5, 10]} )
load_parameter_list.append( {'id':id, 'kS':2, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[10, 15]} )
load_parameter_list.append( {'id':id, 'kS':2, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[15, 20]} )
load_parameter_list.append( {'id':id, 'kS':2, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[20, 25]} )
load_parameter_list.append( {'id':id, 'kS':2, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[0, 25]} )

load_parameter_list.append( {'id':id, 'kS':4, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[5, 10]} )
load_parameter_list.append( {'id':id, 'kS':4, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[10, 15]} )
load_parameter_list.append( {'id':id, 'kS':4, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[15, 20]} )
load_parameter_list.append( {'id':id, 'kS':4, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[20, 25]} )
load_parameter_list.append( {'id':id, 'kS':4, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2', 'speed_bounds':[0, 25]} )

parameters_dict = {}
for parameter_name in load_parameter_list[0].keys():
    parameters_dict[parameter_name] = np.array([f[parameter_name] for f in load_parameter_list  ])

id_array = np.array([f['id'] for f in load_parameter_list  ])
# id_array = np.array([f['id'] for f in load_parameter_list  ])


model_filename_list  = []
for param in load_parameter_list:
    model_filename_list.append( '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__speed_%02d_%02d__ModelMetrics.pkl' \
                                %(param['id'], param['kS'], param['kB'], param['gap'], param['feature'], param['speed_bounds'][0], param['speed_bounds'][1]) )

# model_filename_list.sort()

model_fullfilename_list = [os.path.join(paths['models'], f) for f in model_filename_list]


models = []
for model_fullfilename in model_fullfilename_list:

    with open(model_fullfilename, 'rb') as fid:
        models.append(cPickle.load(fid))

filtered_features_datasets_filename_list = [f.replace('__Models.pkl', '__FilteredFeaturesDataset.h5') for f in
                                            model_filename_list]
filtered_features_datasets_fullfilename_list = [os.path.join(paths['filtered_features_datasets'], f) for f in
                                                filtered_features_datasets_filename_list]

# get the speed and distance of closest approach for these models
sim_parameters = []
for index, filtered_features_datasets_fullfilename in enumerate(filtered_features_datasets_fullfilename_list):

    indices = np.where(models[index]['mask_include'])[0]

    with h5py.File(filtered_features_datasets_fullfilename, 'r') as f:

        mask = np.zeros(f['speed'].shape[0]).astype(bool)
        mask[indices] = True

        sim_parameters.append({'speed':f['speed'][mask], 'distance_closest_approach':f['distance_closest_approach'][mask]})

plt.figure()
plt.hist(sim_parameters[1]['speed'], bins=100)
plt.xlabel("Speed (m/s)")
plt.ylabel('Count')

plt.figure()
plt.plot(sim_parameters[1]['speed'], sim_parameters[1]['distance_closest_approach'], '.k', markersize = 10, alpha = 0.5)
plt.xlabel("Speed (m/s)")
plt.ylabel('Distance of Closest Approach (m)')


# print the accuracy values as a function of speed for the different (ks,kb,gap,feature_set)

parameter_set_list =[{'kS':2, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2'},
        {'kS':4, 'kB':16, 'gap':4, 'feature': 'mask_filtered_features_2'}]


speed_bounds_list_list = []
speed_mean_list_list = []
accuracy_dict_list = []
for parameter_set in parameter_set_list:

    cutt = (parameters_dict['kS'] == parameter_set['kS']) & \
           (parameters_dict['kB'] == parameter_set['kB']) & \
           (parameters_dict['gap'] == parameter_set['gap']) & \
           (parameters_dict['feature'] == parameter_set['feature'])
    indices = np.where(cutt)[0]
    speed_bounds_list = []
    speed_mean_list = []
    accuracy_dict = {model_name:[] for model_name in models[0]['metrics'].keys()}
    for index in indices:
        speed_bounds_list.append(models[index]['speed_bounds'])
        speed_mean_list.append(np.mean(models[index]['speed_bounds']))
        accuracy_temp = []
        for model_name in models[index]['metrics']: # loop over algorithms
            temp = [f['accuracy'] for f in models[index]['metrics'][model_name]]
            accuracy_dict[model_name].append(np.mean(temp))

    accuracy_dict_list.append(accuracy_dict)
    speed_mean_list_list.append(speed_mean_list)
    speed_bounds_list_list.append(speed_bounds_list)


plt.figure(figsize = [20, 10])
plt.grid()

for model_name_index, model_name in enumerate(models[0]['metrics'].keys()):
    for parameter_set_index, parameter_set in enumerate(parameter_set_list):

        plt.plot(speed_mean_list_list[parameter_set_index][:-1], accuracy_dict_list[parameter_set_index][model_name][:-1],
                 label = '%s kS_%02d__kB_%02d__gap_%02d__%s'\
                         %(model_name, parameter_set['kS'], parameter_set['kB'], parameter_set['gap'], parameter_set['feature'],\
                           ),color = plotColors[model_name_index], marker = markerTypes[parameter_set_index],
                 linestyle = lineStyles[parameter_set_index],
                 markersize = 10)

        plt.plot([5, 25], accuracy_dict_list[parameter_set_index][model_name][-1] * np.ones(2),
                 label = '%s kS_%02d__kB_%02d__gap_%02d__%s, trained on all'\
                         %(model_name, parameter_set['kS'], parameter_set['kB'], parameter_set['gap'], parameter_set['feature'],\
                           ),color = plotColors[model_name_index], marker = markerTypes[parameter_set_index],
                 linestyle=lineStyles[parameter_set_index],
                 linewidth = 2,
                 markersize = 16, alpha = 0.6)

plt.legend()
plt.xlabel('Mean Speed (m/s)', fontsize = 16)
plt.ylabel('Accuracy', fontsize = 16)
plt.title('%s' %(model_name))



plt.figure(figsize = [20, 10])
plt.grid()

model_name = 'nn_2layer'
# for model_name_index, model_name in enumerate(models[0]['metrics'].keys()):

bins = np.linspace(0, 1, 501)
bins_center = (bins[1:] + bins[:-1])/2.0
cross_val_index = 0

plot_index = 0
for class_index in xrange(models[0]['prediction_prob_all_dict'][model_name][cross_val_index].shape[1]):

    if class_index == 0:
        isotope_name = 'background'
    else:
        isotope_name = sources['isotope_string_set'][class_index-1]

    cutt = models[0]['y_new'] == class_index
    p = models[0]['prediction_prob_all_dict'][model_name][cross_val_index][cutt,class_index]

    counts, bins_ = np.histogram(p, bins = bins)
    plt.plot(bins_center, counts, label = '{}'.format(isotope_name),
             color = plotColors[plot_index], marker = None,
                 linestyle=lineStyles[plot_index/7])
    plot_index +=1

plt.xlabel('Probability', fontsize = 16)
plt.ylabel('Sum', fontsize = 16)
plt.legend(loc = 3)
# plt.yscale("log")



# plot of the cummulative sum on the

model_name = 'nn_3layer'

for model_index in xrange(len(model_filename_list)):
    plt.figure(figsize = [20, 10])
    plt.grid()

    # for model_name_index, model_name in enumerate(models[0]['metrics'].keys()):

    bins = np.linspace(0, 1, 501)
    bins_center = (bins[1:] + bins[:-1])/2.0
    cross_val_index = 0

    cum_sum_fraction_threshold = 0.9

    plot_index = 0


    for class_index in xrange(models[model_index]['prediction_prob_all_dict'][model_name][cross_val_index].shape[1]):

        if class_index == 0:
            isotope_name = 'background'
        else:
            isotope_name = sources['isotope_string_set'][class_index-1]

        cutt = models[model_index]['y_new'] == class_index
        p = models[model_index]['prediction_prob_all_dict'][model_name][cross_val_index][cutt,class_index]

        counts, bins_ = np.histogram(p, bins = bins)

        cum_sum_fraction = np.cumsum(counts[-1::-1])/float(counts.sum())

        cum_sum_fraction = cum_sum_fraction[-1::-1]

        index_095 = np.argmin(np.abs(cum_sum_fraction - cum_sum_fraction_threshold))

        plt.plot(bins_center, cum_sum_fraction,
                 label = '{}, prob threshold for {}: {}, Cumulative Fraction at 0.5: {}'.format(isotope_name, cum_sum_fraction_threshold, bins_center[index_095], cum_sum_fraction[len(bins_)/2]),
                 color = plotColors[plot_index], marker = None,
                     linestyle=lineStyles[plot_index/7])
        plot_index +=1

    plt.xlabel('Probability', fontsize = 16)
    plt.ylabel('Cummulative Fraction', fontsize = 16)
    plt.legend(loc = 3, fontsize = 12)
    plt.yscale('log')
    plt.ylim((0.5, 1.0))
    plt.title(model_filename_list[model_index])



# plot of the cummulative sum on the

model_name = 'nn_3layer'
cross_val_index = 0

drive_by_metrics_by_isotope = []

for model_index, model_filename in enumerate(model_filename_list):

    drive_by_metrics_by_isotope_temp = {'recall':{}, 'precision':{}, 'f1':{}, 'accuracy':{}}

    # if model_index not in [4, 9]:
    #     continue

    # for model_name_index, model_name in enumerate(models[0]['metrics'].keys()):
    prediction = models[model_index]['prediction_driveby_positive_window_dict'][model_name][cross_val_index]

    number_drivebys = len(models[model_index]['truth_driveby_positive_window'])
    number_sources = len(sources['isotope_string_set'])
    number_classes = number_sources + 1

    truth_driveby = models[model_index]['truth_driveby_positive_window']

    # build the prediction matrix
    # first class is background
    prediction_matrix = np.zeros((number_drivebys, number_classes))

    for driveby_index, driveby_counts in enumerate(models[model_index]['prediction_driveby_positive_window_dict'][model_name][cross_val_index]):

        for class_index, class_count in driveby_counts.iteritems():
            prediction_matrix[driveby_index,class_index] = class_count


    for class_index_index, isotope_name in enumerate(sources['isotope_string_set']):

        class_index = class_index_index + 1
        if class_index == 0:
            isotope_name = 'background'
        else:
            isotope_name = sources['isotope_string_set'][class_index-1]

        # cut of all the actual drive by instances
        cutt = models[model_index]['truth_driveby_positive_window'] == class_index

        # recall = recall_score(truth_driveby[cutt] == class_index, prediction_matrix[cutt,class_index]>0)
        recall = recall_score(truth_driveby == class_index, prediction_matrix[:,class_index]>0)
        accuracy = accuracy_score(truth_driveby == class_index, prediction_matrix[:,class_index]>0)
        precision = precision_score(truth_driveby == class_index, prediction_matrix[:,class_index]>0)
        f1 = f1_score(truth_driveby == class_index, prediction_matrix[:,class_index]>0)

        print('{}, {}, recall: {}, precision: {}, f1: {}, accuracy: {}'.format( model_filename, isotope_name, recall, precision, f1, accuracy))

        drive_by_metrics_by_isotope_temp['recall'][isotope_name] = recall
        drive_by_metrics_by_isotope_temp['precision'][isotope_name] = precision
        drive_by_metrics_by_isotope_temp['accuracy'][isotope_name] = accuracy
        drive_by_metrics_by_isotope_temp['f1'][isotope_name] = f1

    drive_by_metrics_by_isotope.append(drive_by_metrics_by_isotope_temp)


# print the results
print("speed range, Recall Set 1, Recall Set 2, Recall Difference")
for class_index_index, isotope_name in enumerate(sources['isotope_string_set']):
    print(' ')
    for model_index, model_filename in enumerate(model_filename_list):
        speed_bounds = parameters_dict['speed_bounds'][model_index]
        if model_index >= (len(model_filename_list)/2):
            continue
        first_set_index = model_index
        second_set_index = model_index + len(model_filename_list)/2
        recall_1 = drive_by_metrics_by_isotope[first_set_index]['recall'][isotope_name]

        recall_2 = drive_by_metrics_by_isotope[second_set_index]['recall'][isotope_name]
        print('{}-{}, {}: {}, {}, {}'.format(speed_bounds[0], speed_bounds[1], isotope_name, recall_1, recall_2, recall_1 - recall_2))


    # if model_index not in [4, 9]:
    #     continue


# print the accuracy values
for model_index, model in enumerate(models):
    print(model_filename_list[model_index])
    model_name_list = model['metrics'].keys()
    model_name_list.sort()

    for model_name in model_name_list:

        temp = [t['accuracy'] for t in model['metrics'][model_name]]
        print('%s : %3.4f' %(model_name, np.mean(temp)))

# print drive by metriplot_index = 0cs




for model_index, model in enumerate(models):
    print(model_filename_list[model_index])

    # for model_name in model_name_list:
    #     recall_positive_window = [np.sum(t) for t in model['recall_positive_window_dict'][model_name]]
    #     incorrect_prediction_rate_positive_window = [np.sum(t) for t in model['incorrect_prediction_rate_positive_window_dict'][model_name]]
    #     false_positive_rate_negative_window = [np.sum(t) for t in model['false_positive_rate_negative_window_dict'][model_name]]

    for metric_name in ['recall_positive_window', 'incorrect_prediction_rate_positive_window', 'false_positive_rate_negative_window']:
        print('metric name: %s' %metric_name)
        for model_name in model_name_list:
            temp = [t for t in model[metric_name + '_dict'][model_name]]
            print('  %s : %3.4f' %(model_name, np.mean(temp)))

# print the roc curves for each isotope, for each model,
cross_val_index = 0

for model_index, model in enumerate(models):
    print(model_filename_list[model_index])

    model_name = 'rf_0'

    plt.figure()
    plt.grid()
    for isotope_index in list(set(model['y_index_mask_include'])):

        fpr, tpr, threshold = roc_curve(model['y_matrix_mask_include'][:,isotope_index],\
                                        model['prediction_prob_all_dict']['lda'][cross_val_index][model['mask_include'],isotope_index])

        plt.plot(fpr, tpr, color = plotColors[isotope_index], linestyle = lineStyles[isotope_index/7],
                 markeredgecolor = plotColors[isotope_index], label = 'class: {}'.format(isotope_index), alpha = 0.7)
    plt.title(model_name)
    plt.legend(loc = 4)
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)

# recall_positive_window = true_positive_driveby_positive_window.sum() / float(len(true_positive_driveby_positive_window))
# incorrect_prediction_rate_positive_window = false_positive_driveby_positive_window.sum() / float(
#     len(false_positive_driveby_positive_window))
# false_positive_rate_negative_window = false_positive_driveby_negative_window.sum() / float(
#     len(false_positive_driveby_negative_window))

print('drive by recall_positive_window: {}'.format(recall_positive_window))
print('drive by incorrect_prediction_rate_positive_window: {}'.format(incorrect_prediction_rate_positive_window))
print('drive by false_positive_rate_negative_window: {}'.format(false_positive_rate_negative_window))

plt.figure()
plt.grid()
plt.plot()