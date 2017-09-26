"""Calculate the target values for a single file

Usage:
>>python calculate_target_values.py {input_filename} {output_filename}

Arguments
    input_filename - the full filename of the training dataset file
    output_filename - the output file name


*** A better too is probably to use run_training_dataset_calculate_target_values.py as this will process many dataset files.
This only only process a single file.


"""
from __future__ import print_function

import os, sys, time

import training_dataset_processor

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, ".."))

if __name__ == '__main__':
    training_dataset_processor.CalculateTargetValues(sys.argv[1], sys.argv[2])

