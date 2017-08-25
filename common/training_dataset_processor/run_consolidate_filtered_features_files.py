import os, sys, glob



import training_dataset_processor


if __name__ == '__main__':

    file_path = sys.argv[1]
    training_set_id = sys.argv[2]

    kS = int(sys.argv[3])
    kB = int(sys.argv[4])
    gap = int(sys.argv[5])


    temp = glob.glob(os.path.join(file_path, \
                      '*kS_%02d__kB_%02d__gap_%02d__FilteredFeaturesDataset.h5' % ( kS, kB, gap)))

    fullfilename_list = [f for f in temp if training_set_id in f]

    fullfilename_list.sort()

    output_filename = '%s__all__kS_%02d__kB_%02d__gap_%02d__FilteredFeaturesDataset.h5' % (training_set_id, kS, kB, gap)

    output_fullfilename = os.path.join(file_path, output_filename)

    training_dataset_processor.ConsolidateFilteredFeatturesFiles(fullfilename_list, output_fullfilename)

