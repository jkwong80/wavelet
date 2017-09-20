"""
A script for checking the walk by data taken on 9/8/2017 (by Mike and Jason).
Save plots to default location (requires specifying INJECTION_RESOURCES env variable.
No plot of classifications results, just plots of the raw data.

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

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, "../common"))

import training_dataset_processor.training_dataset_processor


# define the paths
paths = training_dataset_processor.training_dataset_processor.GetInjectionResourcePaths()
base_dir = paths['base']
if 'snr' not in paths:
    paths['snr'] = os.path.join(paths['snr_root'], '20170824')


real_data_subpath = '20170908'
real_data_processed_subpath = '20170908_8'
paths['real_data'] = os.path.join(paths['real_data_root'], real_data_subpath)
paths['real_data_processed'] = os.path.join(paths['real_data_processed_root'], real_data_processed_subpath)


# change this to whatever path you want
plot_dir = paths['plot']


save_figures = True


# read in all the data files

fullfilename_list = glob.glob(os.path.join(paths['real_data'], '*.hdf5'))
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

# plot the total counts vs time for all detectors of all datasets
plt.figure()
plt.grid()
filename_plot_list = spectra.keys()
filename_plot_list.sort()
for filename_index, filename in enumerate(filename_plot_list):

    for detector_name in spectra[filename].keys():
        plt.plot(spectra[filename][detector_name].sum(1), label = '{}, {}'.format(filename, detector_name))

plt.xlabel('Acquisition Index')
plt.ylabel('Total Counts')
plt.legend(fontsize = 8)

if save_figures:
    plt.savefig(os.path.join(plot_dir, 'Data_%s__total_counts_vs_acquisition__all.pdf' %(real_data_subpath)))
    plt.close()



# plot the total counts vs time, one figure for each source
filename_plot_list = spectra.keys()
filename_plot_list.sort()
for isotope_name in isotope_name_set:
    plt.figure(figsize = [20, 20])
    plt.grid()


    for filename_index, filename in enumerate(filename_plot_list):

        if isotope_name in filename:
            for detector_name in spectra[filename].keys():
                plt.plot(spectra[filename][detector_name].sum(1), label = '{}, {}'.format(filename, detector_name))


    plt.xlabel('Acquisition Index', fontsize = 16)
    plt.ylabel('Total Counts', fontsize = 16)
    plt.legend(fontsize = 8)
    plt.title(isotope_name, fontsize = 16)

    if save_figures:
        plt.savefig(os.path.join(plot_dir, 'Data_%s__total_counts_vs_acquisition__%s.pdf' %(real_data_subpath, isotope_name)))
        plt.close()




# the spectrum at maximum counts, one plot for each isotope

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

plt.legend(loc = 1, fontsize = 8)
plt.xlim((0, 2000))
plt.ylim((0.1, 1e3))
plt.yscale('log')


if save_figures:
    plt.savefig(os.path.join(plot_dir, 'Data_%s__spectra__all.pdf' %(real_data_subpath)))
    plt.close()




# the spectrum at maximum counts, for all datasets, one figure per isotope
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
                plt.plot(energy_array[filename][detector_name], spectra[filename][detector_name][index, :],
                         label='{}, {}, index: {}'.format(filename, detector_name, index), \
                         color = plotColors[plot_index], linestyle = lineStyles[plot_index/7])
                plot_index += 1
            break
        plt.xlabel('Energy (keV)', fontsize=16)
    plt.ylabel('Total Counts', fontsize = 16)
    plt.title(isotope_name, fontsize = 16)

    plt.legend(loc=1, fontsize=8)
    plt.xlim((0, 2000))
    plt.ylim((0.1, 1e3))
    plt.yscale('log')

    if save_figures:
        plt.savefig(os.path.join(plot_dir, 'Data_%s__spectra__%s.pdf' % (real_data_subpath, isotope_name)))
        plt.close()

