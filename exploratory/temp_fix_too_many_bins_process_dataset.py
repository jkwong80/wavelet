"""
For fixing something related to the SNR_matrix - honestly don't remember. To be removed because was probably just a temporary
Fix for something.
9/17/2017, JK
"""



import os, sys, h5py
import cPickle
import shutil
import glob


input_path = '/Volumes/Lacie2TB/BAA/injection_resources/processed_datasets/5b178c11-a4e4-4b19-a925-96f27c49491b'

fullfilename_list = glob.glob(os.path.join(input_path, '*kS_04__kB_16__gap_04__ProcessedDataset.h5'))


filename_list = [os.path.split(f)[-1] for f in fullfilename_list]


for fullfilename_index, fullfilename in enumerate(fullfilename_list):

    filename = filename_list[fullfilename_index]
    temp_filename = fullfilename.replace('.h5', '_temp.h5')

    shutil.copy(fullfilename, temp_filename)
    with h5py.File(temp_filename, 'r') as f_input:

        with h5py.File(fullfilename, 'w') as f_output:

            f_output.create_dataset('gap', data = f_input['gap'].value)
            f_output.create_dataset('kS', data = f_input['kS'].value)
            f_output.create_dataset('kB', data = f_input['kB'].value)
            f_output.create_dataset('snr_fullfilename', data = f_input['snr_fullfilename'].value)
            f_output.create_dataset('number_samples_save', data = f_input['number_samples_save'].value)
            f_output.create_dataset('number_samples_skip', data = f_input['number_samples_skip'].value)
            f_output.create_dataset('number_bins', data = f_input['number_bins'].value)
            f_output.create_dataset('SNR_matrix', data = f_input['SNR_matrix'][:,:,:50,:])

