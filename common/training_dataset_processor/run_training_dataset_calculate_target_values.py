""" Calculates the target values for training data in a directory
python run_training_dataset_calculate_target_values.py <input_directory> <output_directory>


"""

import os, sys, glob

import training_dataset_processor


def CalculateTargetValuesMultiple(input_directory, output_directory):
    # get the list of files
    fullfilenamelist = glob.glob(os.path.join(input_directory, '*TrainingDataset.h5'))
    fullfilenamelist.sort()

    filenamelist = [os.path.split(f)[-1] for f in fullfilenamelist]

    for fullfilename_index, fullfilename in enumerate(fullfilenamelist):
        output_fullfilename = os.path.join(output_directory, filenamelist[fullfilename_index].replace('TrainingDataset', 'TargetValues'))

        if not os.path.exists(output_fullfilename):
            training_dataset_processor.CalculateTargetValues(fullfilename, output_fullfilename)
        else:
            print('Skipping as already exists: {}'.format(output_fullfilename))

if __name__ == '__main__':
    CalculateTargetValuesMultiple(sys.argv[1], sys.argv[2])


