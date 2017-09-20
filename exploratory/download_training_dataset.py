""" For downloading all the training dataset files for a training job

python download_training_data.py {job uuid} {api host} {}

Example
>> python download_training_data.py 5b178c11-a4e4-4b19-a925-96f27c49491b 52.204.80.65:5000 .


8/23/2017, JK
9/19/2017, JK

"""

import sys, os, json
import requests
import urllib


def main(job_id, host, output_path):

    headers = {
        'Content-type': 'application/json'
    }
    r = requests.get('{}/training_dataset/{}'.format(host, job_id), headers=headers)

    training_dataset_info = r.json()['training_datasets']

    download_url_list = [d['download_url'] for d in training_dataset_info[0]['files']]
    filename_list = [os.path.split(d['file_location'])[-1] for d in training_dataset_info[0]['files']]

    testfile = urllib.URLopener()

    for download_url_index, download_url in enumerate(download_url_list):
        output_fullfilename = os.path.join(output_path, filename_list[download_url_index])
        print('Downloading: {}'.format(output_fullfilename))
        testfile.retrieve(download_url, output_fullfilename)

if __name__ == '__main__':

    job_id = sys.argv[1]
    host = sys.argv[2]
    output_path = sys.argv[3]

    main(job_id, host, output_path)