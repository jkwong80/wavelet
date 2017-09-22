"""Consolidates the filtered features files



>>python common/training_dataset_processor/run_consolidate_filtered_features_files.py {file path} {job id} {ks} {kB] {gap} {start index} {stop index} {name of the feature index list}

Example
>>python common/training_dataset_processor/run_consolidate_filtered_features_files.py /Volumes/Lacie2TB/BAA/injection_resources/filtered_features_datasets/dd70a53c-0598-447c-9b23-ea597ed1704e dd70a53c-0598-447c-9b23-ea597ed1704e 2 16 4 0 99 mask_filtered_features_3

Arguments

file path - location of the filtered features files
job id - job uuid
kS
kB
gap
start index - start index of files to consolidate
stop index - stop index of files to consolidate
feature set name - feature set name in data structure created by select_best_features_first_pass.py. example "mask_filtered_features_3"



"""

import os, sys, glob

import training_dataset_processor


if __name__ == '__main__':

    print(sys.argv)
    file_path = sys.argv[1]
    # training_data_path = sys.argv[2]
    training_set_id = sys.argv[2]

    kS = int(sys.argv[3])
    kB = int(sys.argv[4])
    gap = int(sys.argv[5])

    start_index = int(sys.argv[6])
    stop_index = int(sys.argv[7])
    feature_indices_name = sys.argv[8]

    temp = glob.glob(os.path.join(file_path, \
                      '*kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' % ( kS, kB, gap, feature_indices_name)))

    # make list of files that are in the file number range
    fullfilename_list = [f for f in temp if (training_set_id in f) and ('all' not in f) and (int(f.split('__')[1]) >= start_index) and (int(f.split('__')[1]) <= stop_index)]
    fullfilename_list.sort()

    # make sure not 'all' and that the index is in range
    # filename_list = [os.path.split(f)[-1] for f in fullfilename_list if ('all' not in f) and (int(f.split('__')[1]) >= start_index) and (int(f.split('__')[1]) <= stop_index)]
    filename_list = [os.path.split(f)[-1] for f in fullfilename_list]

    # print(filename_list)

    output_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' % (training_set_id, kS, kB, gap, feature_indices_name)

    output_fullfilename = os.path.join(file_path, output_filename)

    training_dataset_processor.ConsolidateFilteredFeatturesFiles(fullfilename_list, output_fullfilename)
