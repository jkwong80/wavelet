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

from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

# assumes that you are at the base of the repo
sys.path.append('common')

# from training_dataset_processor.training_dataset_processor import GetInjectionResourcePaths, GetSourceMapping
import training_dataset_processor.training_dataset_processor


from nn_models import create_neural_network_2layer_model, create_neural_network_3layer_model, create_neural_network_4layer_model,\
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



model_subpath = '8'
paths['models'] = os.path.join(paths['models_root'], model_subpath)

model_filename_list = ['dd70a53c-0598-447c-9b23-ea597ed1704e__all__kS_02__kB_16__gap_04__mask_filtered_features_2__Models.pkl',\
                 'dd70a53c-0598-447c-9b23-ea597ed1704e__all__kS_04__kB_32__gap_08__mask_filtered_features_2__Models.pkl']

filtered_features_datasets_filename_list = [f.replace('__Models.pkl', '__FilteredFeaturesDataset.h5') for f in model_filename_list]

filtered_features_datasets_fullfilename_list = [os.path.join(paths['filtered_features_datasets'], f) for f in filtered_features_datasets_filename_list]

models = []
for model_filename in model_filename_list:
    model_fullfilename = os.path.join(paths['models'], model_filename)

    with open(model_fullfilename, 'rb') as fid:
        models.append(cPickle.load(fid))

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


# print the accuracy values
for model_index, model in enumerate(models):
    print(model_filename_list[model_index])
    model_name_list = model['metrics'].keys()
    model_name_list.sort()

    for model_name in model_name_list:

        temp = [t['accuracy'] for t in model['metrics'][model_name]]
        print('%s : %3.4f' %(model_name, np.mean(temp)))

# print drive by metrics
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