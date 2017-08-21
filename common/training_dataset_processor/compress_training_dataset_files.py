
from __future__ import print_function

import h5py
import numpy as np
import os
import sys
import time
import glob
import numpy as np
import ast
import shutil

from multiprocessing import Pool

# file_directory = os.path.dirname(os.path.realpath(__file__))
# print(os.path.realpath(__file__))
# print(file_directory)
# sys.path.append(file_directory)
#
# sys.path.append(os.path.join(file_directory, ".."))



# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'
# base_dir = os.path.join(os.environ['HOME'], 'injection_resources')
if 'INJECTION_RESOURCES' in os.environ:
    base_dir = os.environ['INJECTION_RESOURCES']
else:
    base_dir = os.path.join(os.environ['HOME'], 'injection_resources')

plot_dir = os.path.join(base_dir, 'plots', time.strftime('%Y%m%d'))
training_datasets_root_path = os.path.join(base_dir, 'training_datasets')
processed_datasets_root_path = os.path.join(base_dir, 'processed_datasets')

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)



def worker(working_path, file_suffix):

    # get the sim file names

    print(file_suffix)

    training_dataset_fullfilename_list = glob.glob(os.path.join(working_path, '*' + file_suffix))
    training_dataset_fullfilename_list.sort()
    training_dataset_filename_list = [os.path.split(f)[-1] for f in training_dataset_fullfilename_list]

    print(training_dataset_fullfilename_list)
    # return

    training_dataset_index_list = [int(f.split('__')[1]) for f in training_dataset_filename_list]

    for training_dataset_fullfilename_index, training_dataset_fullfilename in enumerate(training_dataset_fullfilename_list):

        t_start = time.time()

        print('Working on {}'.format(training_dataset_fullfilename))

        temporary_filename = os.path.join(working_path, training_dataset_fullfilename.replace(file_suffix, file_suffix.replace('.h5', '_temp.h5') ))

        # copy to temporary file name
        print('Copying file.  This will take a while.')
        shutil.copy(training_dataset_fullfilename, temporary_filename)

        print('Deleting old file.')
        os.remove(training_dataset_fullfilename)

        # open old file with temporary name
        input_file = h5py.File(temporary_filename, 'r')

        # Create new file with original name
        print('Creating new file: {}'.format(training_dataset_fullfilename))
        output_file = h5py.File(training_dataset_fullfilename, 'w')

        for key in input_file.keys():
            print('Working on {}'.format(key))
            try:
                output_file.create_dataset(key, data=input_file[key].value, compression='gzip')
            except:
                # goes here if not array
                output_file.create_dataset(key, data=input_file[key].value)
        output_file.close()
        input_file.close()

        # Removing old file
        print('Removing temporary file.')
        os.remove(temporary_filename)

        print('Time Elapsed: %3.3f' %(time.time() - t_start))



if __name__ == '__main__':

    worker(sys.argv[1], sys.argv[2])

