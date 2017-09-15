
import os, sys, glob, time
import h5py, cPickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from process_real_data_to_wavelet import ProcessRebinReadingsFile

# assumes that you are at the base of the repo
sys.path.append('common')

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
models_dataset_path = os.path.join(models_dataset_root_path, '8')

if not os.path.exists(models_dataset_root_path):
    os.mkdir(models_dataset_root_path)
if not os.path.exists(models_dataset_path):
    os.mkdir(models_dataset_path)

snr_root_path = os.path.join(base_dir, 'snr_functions')

if not os.path.exists(snr_root_path):
    os.mkdir(snr_root_path)
snr_path = os.path.join(snr_root_path, '20170824')

real_data_root_path = os.path.join(base_dir, 'real_data')

real_data_subpath = '20170908'
real_data_processed_subpath = '20170908_8'


real_data_processed_root_path = os.path.join(base_dir, 'real_data_processed')
real_data_processed_path = os.path.join(real_data_processed_root_path, real_data_processed_subpath)
real_data_path = os.path.join(real_data_root_path, real_data_subpath)

if not os.path.exists(real_data_processed_root_path):
    os.mkdir(real_data_processed_root_path)
if not os.path.exists(real_data_processed_path):
    os.mkdir(real_data_processed_path)

print("real_data_path: {}".format(real_data_path))
print("real_data_processed_path: {}".format(real_data_processed_path))

feature_selection_path = os.path.join(base_dir, 'feature_selection')

# lost of parameters to cycle through
# kS_list = [2, 4, 8]
# kB_list = [16, 32]
# gap_list = [4, 8, 16]

kS_list = [2,]
kB_list = [16,]
gap_list = [4,]

feature_indices_name_list = ['mask_filtered_features_2']
feature_selection_filename ='5b178c11-a4e4-4b19-a925-96f27c49491b__kS_02__kB_16__gap_04__top_features.h5'

input_fullfilename_list = glob.glob(os.path.join(real_data_path, '*.hdf5'))
print(input_fullfilename_list)

input_filename_list = [os.path.split(f)[-1] for f in input_fullfilename_list]

detector_name = '88'
number_bins = 512

for input_fullfilename_index, input_fullfilename in enumerate(input_fullfilename_list):

    input_filename = input_filename_list[input_fullfilename_index]

    for kS_index, kS in enumerate(kS_list):
        for kB_index, kB in enumerate(kB_list):
            for gap_index, gap in enumerate(gap_list):
                for feature_indices_name_index, feature_indices_name in enumerate(feature_indices_name_list):

                    snr_name_suffix = 'kS_%02d__kB_%02d__gap_%02d' % (kS, kB, gap)
                    snr_fullfilename = os.path.join(snr_path, 'f_snr__%s.pkl' % snr_name_suffix)

                    if not os.path.exists(snr_fullfilename):
                        print('Skipping, does not exist: {}'.format(snr_fullfilename))
                        continue

                    # feature_selection_filename = '5b178c11-a4e4-4b19-a925-96f27c49491b__%s__top_features.h5' %snr_name_suffix

                    feature_selection_fullfilename = os.path.join(feature_selection_path, feature_selection_filename)

                    if not os.path.exists(feature_selection_fullfilename):
                        print('Skipping, does not exist: {}'.format(feature_selection_fullfilename))
                        continue

                    if 'h5' in input_filename:
                        output_filename = '{}__{}__{}.h5'.format(input_filename.replace('.h5', ''),  snr_name_suffix, feature_indices_name)
                    else:
                        output_filename = '{}__{}__{}.h5'.format(input_filename.replace('.hdf5', ''),  snr_name_suffix, feature_indices_name)

                    output_fullfilename = os.path.join(real_data_processed_path, output_filename)

                    print('**')
                    print(output_filename)
                    print(os.path.join(real_data_processed_path, output_filename))
                    print(real_data_processed_path)
                    print(output_fullfilename)

                    if os.path.exists(output_fullfilename):
                        print('Skipping, already exists: {}'.format(output_fullfilename))
                        continue

                    ProcessRebinReadingsFile(input_fullfilename, feature_selection_fullfilename,
                                             feature_indices_name, snr_fullfilename, detector_name,
                                             output_fullfilename, True, [5, 20])
