from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import glob
import numpy as np

from multiprocessing import Pool

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(file_directory)

sys.path.append(os.path.join(file_directory, ".."))


import training_dataset_processor


plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']


# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'
base_dir = os.path.join(os.environ['HOME'], 'injection_resources')
plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')



if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# LOAD DATA FROM SIMULATION FILE
training_set_id = '9a1be8d8-c573-4a68-acf8-d7c7e2f9830f'

# get the sim file names
training_dataset_path = os.path.join(training_datasets_root_path, training_set_id)

training_dataset_fullfilename_list = glob.glob(os.path.join(training_dataset_path, '*.h5'))
training_dataset_filename_list = [os.path.split(f)[-1] for f in training_dataset_fullfilename_list]

training_dataset_index_list = [int(f.split('__')[1]) for f in training_dataset_filename_list]

# training_dataset_processor.ProcessTrainingDataset(param)

if __name__ == '__main__':

    # append a time string so that we know all the jobs are part of the same unit

    t_start = time.time()

    number_threads = int(sys.argv[1])

    pool = Pool(processes=number_threads)

    arguments_list = []

    for file_list_index in xrange(len(training_dataset_fullfilename_list)):

        training_dataset_filename = training_dataset_filename_list[file_list_index]
        training_dataset_fullfilename = training_dataset_fullfilename_list[file_list_index]
        training_dataset_index = training_dataset_index_list[file_list_index]
        training_dataset_filename_prefix = '__'.join(training_dataset_filename.split('__')[0:2])

        processed_dataset_path = os.path.join(processed_datasets_root_path, training_set_id)
        if not os.path.exists(processed_dataset_path):
            os.mkdir(processed_dataset_path)

        param = {}
        param['input_filename'] = training_dataset_fullfilename

        param['output_dir'] = processed_dataset_path

        param['gap'] = 3
        param['kS_list'] = [4, 8, 16]
        # param['kS_list'] = [1, 2, 4, 8, 16]

        param['kB'] = 8

        param['worker_no'] = file_list_index

        arguments_list.append(param)

        print(param)


    result = pool.map(training_dataset_processor.ProcessTrainingDataset, arguments_list)
    pool.close()
    pool.join()
    print('Time Elapsed: %3.3f' %(time.time() - t_start))
