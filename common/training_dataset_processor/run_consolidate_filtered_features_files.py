"""Consolidates the filtered features files
Arguments

file path
uuid
kS
kB
gap
start index
stop index
feature set name


"""

import os, sys, glob
z
import training_dataset_processor


if __name__ == '__main__':

    file_path = sys.argv[1]
    training_set_id = sys.argv[2]

    kS = int(sys.argv[3])
    kB = int(sys.argv[4])
    gap = int(sys.argv[5])

    start_index = int(sys.argv[6])
    stop_index = int(sys.argv[7])
    feature_indices_name = sys.argv[8]


    temp = glob.glob(os.path.join(file_path, \
                      '*kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' % ( kS, kB, gap, feature_indices_name)))

    # filtered_features_dataset_fullfilename = os.path.join(filtered_features_dataset_path,
    #                                                       '%s__%03d__kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' \
    #                                                       % (training_set_id, file_list_index, kS, kB, gap,
    #                                                          feature_indices_name))

    fullfilename_list = [f for f in temp if (training_set_id in f) and ('all' not in f) and (int(f.split('__')[1]) >= start_index) and (int(f.split('__')[1]) <= stop_index)]
    fullfilename_list.sort()
    # make sure not 'all' and that the index is in range
    # filename_list = [os.path.split(f)[-1] for f in fullfilename_list if ('all' not in f) and (int(f.split('__')[1]) >= start_index) and (int(f.split('__')[1]) <= stop_index)]
    filename_list = [os.path.split(f)[-1] for f in fullfilename_list]

    # temp = [int(f.split('__')[1]) for f in filename_list if 'all' not in f]
    #
    # dataset_index_list = [f for f in temp if (f >= start_index) and (f <= stop_index)]

    print(filename_list)

    output_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__%s__FilteredFeaturesDataset.h5' % (training_set_id, kS, kB, gap, feature_indices_name)

    output_fullfilename = os.path.join(file_path, output_filename)

    training_dataset_processor.ConsolidateFilteredFeatturesFiles(fullfilename_list, output_fullfilename)

