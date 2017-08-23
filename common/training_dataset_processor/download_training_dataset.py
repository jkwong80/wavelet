""" For downloading all the training dataset files for a training job

8/23/2017

"""
import sys, os, json
import requests
import urllib

#################################################
# Edit this

job_id = '5b178c11-a4e4-4b19-a925-96f27c49491b'

# the host is needed for the API for retrieving the information on the job.
host = 'http://127.0.0.1:5000'
# host = 'http://52.204.80.65:5000'

##########################################


if 'INJECTION_RESOURCES' in os.environ:
    base_dir = os.environ['INJECTION_RESOURCES']
else:
    base_dir = os.path.join(os.environ['HOME'], 'injection_resources')

training_datasets_root_path = os.path.join(base_dir, 'training_datasets')

training_dataset_path = os.path.join(training_datasets_root_path, job_id)

if not os.path.exists(training_dataset_path):
    os.mkdir(training_dataset_path)
    print('Created directory: {}'.format(training_dataset_path))

headers = {
    'Content-type': 'application/json'
}
r = requests.get('{}/training_dataset/{}'.format(host, job_id), headers=headers)

training_dataset_info = r.json()['training_datasets']


download_url_list = [d['download_url'] for d in training_dataset_info[0]['files']]
filename_list = [os.path.split(d['file_location'])[-1] for d in training_dataset_info[0]['files']]

testfile = urllib.URLopener()


for download_url_index, download_url in enumerate(download_url_list):
    output_fullfilename = os.path.join(training_dataset_path, filename_list[download_url_index])
    print('Downloading: {}'.format(output_fullfilename))
    testfile.retrieve(download_url, output_fullfilename)

