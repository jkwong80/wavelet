"""
Isotope id algorithm class


"""



from keras.models import load_model
import wavelet_core
import copy
import h5py
from nn_models import create_neural_network_2layer_model, create_neural_network_3layer_model


class isotope_id(object):
    def __init__(self, parameters_file, neural_network_file):

        # self.minimum_ingest_index_before_alert = kB*2
        self.model = load_model(neural_network_file)


        # number_spectral_bins = 512
        # if number_spectral_bins == 512:
        #     number_wavelet_bins = 4107
        # elif number_spectral_bins == 1024:
        #     number_wavelet_bins = 9228

        self.ingest_index = 0

        self.snr_previous = None
        self.snr_recent = None
        self.prob_previous = None
        self.prob_recent = None

        with h5py.File(parameters_file, 'r') as f:
            kB = f['kB'].value
            kS = f['kS'].value
            gap = f['gap'].value
            number_spectral_bins = f['number_spectral_bins'].value
            bins_subset = f['bins_subset'].value
            self.isotopes = f['isotopes'].value

        # create the snr function
        self.f_snr = wavelet_core.isoPerceptron.isoSNRFeature(number_spectral_bins, kB, gap, kS, bins = bins_subset)

        self.snr_dimensions = self.model.get_config()[0]['config']['units']
        self.prob_dimensions =self.model.get_config()[-1]['config']['units']

    def ingest(self, spectrum):
        self.snr_previous = copy.deepcopy(self.snr_recent)
        self.snr_recent = self.f_snr.ingest(spectrum)
        # print(self.snr_recent)
        self.prob_previous = copy.deepcopy(self.prob_recent)
        try:
            self.prob_recent = self.model.predict(self.snr_recent)
        except:
            self.prob_recent = self.model.predict(self.snr_recent.reshape((1, self.snr_recent.shape[0])))

        return self.prob_recent, self.snr_recent

    def clear_state(self):
        pass
