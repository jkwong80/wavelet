""" test_isotope_id.py

Demonstrates the
"""

import os, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

lineStyles = ['-', '--', '-.', ':']
markerTypes = ['.', '*', 'o', 'd', 'h', 'p', 's', 'v', 'x']
plotColors = ['k', 'r', 'g', 'b', 'm', 'y', 'c'] * 10


file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, "../common"))
sys.path.append(os.path.join(file_directory, ".."))

import wavelet_core.bins

import training_dataset_processor.training_dataset_processor
import detection_algorithms.isotope_id.isotope_id

paths = training_dataset_processor.training_dataset_processor.GetInjectionResourcePaths()

# real data path
paths['real_data'] = os.path.join(paths['real_data_root'], '20170908')

parameters_file = os.path.join(paths['models_root'], '9', 'model_parameters.h5')
neural_network_file = os.path.join(paths['models_root'], '9', 'model__kS_02__kB_16__gap_04__mask_filtered_features_2__speed_00_25__nn_3layer__00_02.h5')

id = detection_algorithms.isotope_id.isotope_id.isotope_id(parameters_file, neural_network_file)

# open data taken by mike

input_filename = 'Ba133-1_east.hdf5'

input_fullfilename = os.path.join(paths['real_data'], input_filename)

detector_name = '88'
with h5py.File(input_fullfilename, 'r') as dat:
    print(dat.keys())
    if 'rebinned_spectra' in dat[detector_name]:
        spectra = dat[detector_name]['rebinned_spectra'].value
        t = dat[detector_name]['timestamp_us'].value


snr_array = np.zeros((spectra.shape[0], id.snr_dimensions))
prob_array = np.zeros((spectra.shape[0], id.prob_dimensions))

for acquisition_index in xrange(spectra.shape[0]):

    temp = id.ingest(spectra[acquisition_index,:])
    prob_array[acquisition_index,:] = temp[0]
    snr_array[acquisition_index,:] = temp[1]


plt.figure(figsize = [20, 20])
plt.plot(np.argmax(prob_array, axis = 1))

plt.figure()
plt.imshow(prob_array, aspect = 'auto', interpolation = 'nearest')


plt.figure(figsize = [20, 20])
plt.subplot(2, 2, 1)
plt.imshow(snr_array.T, aspect = 'auto', interpolation = 'nearest')
plt.title('SNR')

plt.xlabel('Acquisition Index')

plt.ylabel('SNR')


plt.subplot(2, 2, 2)
plt.imshow(prob_array.T, aspect = 'auto', interpolation = 'nearest')
plt.title('Probability')
plt.xlabel('Acquisition Index')
plt.ylabel('Class Index')

plt.subplot(2, 2, 3)
plt.grid()
for i in xrange(prob_array.shape[1]):
    plt.plot(prob_array[:,i], alpha = 0.5, color = plotColors[i % 7], linestyle = lineStyles[i/7], label = '{}'.format(id.isotopes[i]))
plt.legend(fontsize = 9)
plt.xlabel('Acquisition Index')
plt.ylabel('Probability')

plt.title('Probability of each Class')

plt.subplot(2, 2, 4)
plt.grid()
plt.plot(np.argmax(prob_array, axis =1), alpha = 0.5, label = 'Class Index')
plt.plot(spectra.sum(1)/max(spectra.sum(1)), label = 'Counts (normalized)')

plt.title('Index of Class with Highest Probability')

plt.xlabel('Acquisition Index')
plt.ylabel('Predicted Class Index')
plt.legend()


plt.figure()
plt.grid()
plt.imshow(spectra.T, aspect = 'auto', interpolation = 'nearest')
plt.xlabel('Acquisition Index')
plt.ylabel('Energy Bin')
plt.title('Rebinned Spectra\n{}'.format(input_filename))

