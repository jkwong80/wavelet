"""Calculate the target values"""
from __future__ import print_function

import os, sys, time

import training_dataset_processor

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, ".."))

if __name__ == '__main__':
    training_dataset_processor.CalculateTargetValues(sys.argv[1], sys.argv[2])
