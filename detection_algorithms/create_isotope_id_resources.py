import os, sys
import h5py

import matplotlib.pyplot as plt

file_directory = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
print(file_directory)
sys.path.append(os.path.join(file_directory, "../common"))


import wavelet_core.bins

import training_dataset_processor.training_dataset_processor



paths = training_dataset_processor.training_dataset_processor.GetInjectionResourcePaths()

paths['real_data'] = os.path.join(paths['real_data_root'], '20170908')


# load the bin indices
feature_selection_file = '5b178c11-a4e4-4b19-a925-96f27c49491b__kS_02__kB_16__gap_04__top_features.h5'

dat = h5py.File(os.path.join(paths['feature_selection_root'], feature_selection_file), 'r')




id = 'dd70a53c-0598-447c-9b23-ea597ed1704e'
sources = training_dataset_processor.training_dataset_processor.GetSourceMapping(id)

isotope_list = np.hstack(('background', sources['isotope_string_set']))


# Get the bins from the snr thing

feature_selection_set_name = 'mask_filtered_features_2'

bin_indices = dat[feature_selection_set_name].value

kB = 16
gap = 4
kS = 2
number_spectral_bins = 512

bins_512 = wavelet_core.bins.waveletBinTree(512).flatList()

bins_subset = [bins_512[i] for i in xrange(len(bin_indices)) if bin_indices[i]]
bins_subset_first_last_indices = np.zeros((len(bins_subset), 2)).astype(int)

for i in xrange(len(bins_subset)):
    bins_subset_first_last_indices[i,0] = bins_subset[i][0]
    bins_subset_first_last_indices[i,1] = bins_subset[i][-1]


# open data taken by mike

input_filename = 'Ba133-1_east.hdf5'

input_fullfilename = os.path.join(paths['real_data'], input_filename)

detector_name = '88'
with h5py.File(input_fullfilename, 'r') as dat:
    print(dat.keys())
    if 'rebinned_spectra' in dat[detector_name]:
        spectra = dat[detector_name]['rebinned_spectra'].value
        t = dat[detector_name]['timestamp_us'].value

'9', 'dd70a53c-0598-447c-9b23-ea597ed1704e__all__kS_02__kB_16__gap_04__mask_filtered_features_2__speed_00_25__nn_3layer__00_02.h5'


with h5py.File(os.path.join(paths['models_root'], '9', 'model_parameters.h5'), 'w') as f:
    f.create_dataset('kB', data = 16)
    f.create_dataset('kS', data = 2)
    f.create_dataset('gap', data = 4)
    f.create_dataset('number_spectral_bins', data = 512)
    f.create_dataset('bins_subset', data = bins_subset_first_last_indices)
    f.create_dataset('isotopes', data = isotope_list)





# plt.figure()
#
# temp = np.zeros(1024)
# for i in xrange(len(bins_subset)):
#     for j in xrange(bins_subset_first_last_indices[i,0], bins_subset_first_last_indices[i,1]+1):
#         temp[j] += 1
#
# plt.plot(temp)
# plt.xlabel('Energy Bin Index', fontsize = 16)
# plt.ylabel('Count', fontsize = 16)
# plt.title("Wavelet Bin Overlap Count")

# snr_1 = np.zeros((spectra.shape[0], len(bins_subset)))
# snr_2 = np.zeros((spectra.shape[0], len(bins_subset)))
# snr_3 = np.zeros((spectra.shape[0], len(bins_subset)))
#
# # Testing the two to see if they are equal
#
# f_snr_1 = wavelet_core.isoPerceptron.isoSNRFeature(number_spectral_bins, kB, gap, kS, bins = bins_subset)
# f_snr_2 = wavelet_core.isoPerceptron.isoSNRFeature(number_spectral_bins, kB, gap, kS, bins = bins_subset_first_last_indices)
# f_snr_3 = wavelet_core.isoPerceptron.isoSNRFeature(number_spectral_bins, kB, gap, kS, bins = bins_subset)
#
#
# for i in xrange(spectra.shape[0]):
#     snr_1[i, :] = f_snr_1.ingest(spectra[i,:])
#     snr_2[i, :] = f_snr_2.ingest(spectra[i,:])
#     snr_3[i, :] = f_snr_3.ingest(spectra[i,:])
#
#
# for i in xrange(spectra.shape[0]):
#     snr_1[i, :] = f_snr_1.ingest(spectra[i,:])
#     snr_2[i, :] = f_snr_2.ingest(spectra[i,:])
#     snr_3[i, :] = f_snr_3.ingest(spectra[i,:])
#
# plt.figure()
# plt.grid()
# index = 60
# plt.plot(snr_1[index, :], '-k', alpha = 0.6)
# plt.plot(snr_2[index, :], '-r', alpha = 0.6)
# plt.plot(snr_3[index, :], '-b', alpha = 0.6)

# open the features file

# plt.figure()
# plt.imshow(snr_2, aspect = 'auto', interpolation = 'nearest')
