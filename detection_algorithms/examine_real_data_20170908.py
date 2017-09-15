"""
A script for checking the walk by data taken on 9/8/2017.

(everything looks fine as far as I can tell)


"""

import os, sys, time
import json, glob
import h5py
import matplotlib.pyplot as plt
import numpy as np


lineStyles = ['-', '--', '-.', ':']
markerTypes = ['.', '*', 'o', 'd', 'h', 'p', 's', 'v', 'x']
plotColors = ['k', 'r', 'g', 'b', 'm', 'y', 'c'] * 10


# Define all the paths
# base_dir = '/Volumes/Lacie2TB/BAA/Data'mm
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

snr_root_path = os.path.join(base_dir, 'snr_functions')

if not os.path.exists(snr_root_path):
    os.mkdir(snr_root_path)

snr_path = os.path.join(snr_root_path, '20170824')

feature_selection_path = os.path.join(base_dir, 'feature_selection')

feature_selection_filename = '5b178c11-a4e4-4b19-a925-96f27c49491b__kS_02__kB_16__gap_04__top_features.h5'
feature_selection_fullfilename = os.path.join(feature_selection_path, feature_selection_filename)

filtered_features_dataset_root_path = os.path.join(base_dir, 'filtered_features_datasets')

real_data_root_path = os.path.join(base_dir, 'real_data')
real_data_processed_root_path = os.path.join(base_dir, 'real_data_processed')

if not os.path.exists(real_data_processed_root_path):
    os.mkdir(real_data_processed_root_path)
print(real_data_processed_root_path)
real_data_path = os.path.join(real_data_root_path, '20170908')


# read in all the files

fullfilename_list = glob.glob(os.path.join(real_data_path, '*.hdf5'))
fullfilename_list.sort()
filename_list = [os.path.split(f)[-1] for f in fullfilename_list]

# get name of the sources and create set
isotope_name_list = [name.split('-')[0] for name in filename_list]
isotope_name_set = list(set(isotope_name_list))

spectra  = {}
energy_array = {}

for fullfilename_index, fullfilename in enumerate(fullfilename_list):
    print("Reading: {}".format(fullfilename))
    filename = filename_list[fullfilename_index]

    spectra[filename] = {}
    energy_array[filename] = {}
    with h5py.File(fullfilename, 'r') as f:

        for detector_name in f.keys():
            spectra[filename][detector_name] = f[detector_name]['rebinned_spectra'].value
            energy_array[filename][detector_name] = f[detector_name]['energies'][0]



# Analysis




# plot the total counts vs time for all datasets

plt.figure()
plt.grid()

filename_plot_list = spectra.keys()
filename_plot_list.sort()

for filename_index, filename in enumerate(filename_plot_list):

    for detector_name in spectra[filename].keys():
        plt.plot(spectra[filename][detector_name].sum(1), label = '{}, {}'.format(filename, detector_name))

plt.xlabel('Acquisition Index')
plt.ylabel('Total Counts')
plt.legend()



# plot the total counts vs time, one figure for each source


filename_plot_list = spectra.keys()
filename_plot_list.sort()

for isotope_name in isotope_name_set:
    plt.figure()
    plt.grid()


    for filename_index, filename in enumerate(filename_plot_list):

        if isotope_name in filename:
            for detector_name in spectra[filename].keys():
                plt.plot(spectra[filename][detector_name].sum(1), label = '{}, {}'.format(filename, detector_name))


    plt.xlabel('Acquisition Index', fontsize = 16)
    plt.ylabel('Total Counts', fontsize = 16)
    plt.legend()
    plt.title(isotope_name, fontsize = 16)



# the spectrum at maximum counts, for all datasets
plt.figure()
plt.grid()

filename_plot_list = spectra.keys()
filename_plot_list.sort()

for filename_index, filename in enumerate(filename_plot_list):
    for detector_name in spectra[filename].keys():

        # fine the peaks counts acquisition index
        index = np.argmax(spectra[filename][detector_name].sum(1))
        plt.plot(energy_array[filename][detector_name], spectra[filename][detector_name][index,:], label = '{}, {}, index: {}'.format(filename, detector_name, index))

plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.legend()


# the spectrum at maximum counts, for all datasets
filename_plot_list = spectra.keys()
filename_plot_list.sort()

plt.figure(figsize = [14,  10])
plt.grid()

plot_index = 0
for isotope_name in isotope_name_set:
    for filename_index, filename in enumerate(filename_plot_list):
        if isotope_name in filename:
            for detector_name in spectra[filename].keys():
                index = np.argmax(spectra[filename][detector_name].sum(1))
                plt.plot(energy_array[filename][detector_name],
                         spectra[filename][detector_name][(index-1):(index+2), :].mean(0),
                         label='{}, {}, index: {}'.format(filename, detector_name, index),
                         color=plotColors[plot_index], linestyle=lineStyles[plot_index / 7])
                # plt.plot(spectra[filename][detector_name][index, :],
                #          label='{}, {}, index: {}'.format(filename, detector_name, index))
                plot_index += 1
                break
            break
plt.xlabel('Energy (keV)', fontsize = 16)
plt.ylabel('Total Counts', fontsize = 16)
plt.legend()
# plt.title(isotope_name, fontsize = 16)

plt.legend(loc = 1)
plt.xlim((0, 2000))
plt.ylim((0.1, 1e3))
plt.yscale('log')



# the spectrum at maximum counts, for all datasets
filename_plot_list = spectra.keys()
filename_plot_list.sort()
plot_index = 0
for isotope_name in isotope_name_set:
    plt.figure()
    plt.grid()


    for filename_index, filename in enumerate(filename_plot_list):

        if isotope_name in filename:
            for detector_name in spectra[filename].keys():
                index = np.argmax(spectra[filename][detector_name].sum(1))
                plt.plot(spectra[filename][detector_name][index, :],
                         label='{}, {}, index: {}'.format(filename, detector_name, index), \
                         color = plotColors[plot_index], linestyle = lineStyles[plot_index/7])
                plot_index += 1
            break
    plt.xlabel('Acquisition Index', fontsize = 16)
    plt.ylabel('Total Counts', fontsize = 16)
    plt.legend()
    plt.title(isotope_name, fontsize = 16)
