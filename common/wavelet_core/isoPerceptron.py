#! /usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np
from filters import *
from bins import *
import sys
import pickle
from filters import *
from scipy.misc import factorial
from scipy.special import erf
from lld import lld
import time

import numpy as np

class isoSNRFeature:
    def __init__(self, nElem, kB, gap, kS, bins = None):
        w = waveletBinTree(nElem)

        if bins is None:
            self.bins = w.flatList()
        else:
            self.bins = bins

        self.nchan = np.zeros(len(self.bins))
        for i in range(len(self.bins)):
            if type(self.bins) == np.ndarray:
                self.nchan[i] = self.bins[i][-1] - self.bins[i][0] + 1
            else:
                self.nchan[i] = len(self.bins[i])

        self.BEst = backgroundEstimation(kB, gap, len(self.bins))
        self.S = EWMA(kS, len(self.bins))
        self.kS = kS
        # self.S = []
        # if type(kS) is list:
        #     for k in kS:
        #         self.S.append(EWMA(k, len(self.bins)))
        # else:
        #     self.S.append(EWMA(kS, len(self.bins)))
        self.kS = kS
        self.kB = kB
        self.E = np.arange(nElem) * 3.0
        self.SNR = np.zeros(len(self.bins))
        self.cumSNR = 0.0

        # self.bin_matrix = np.zeros((len(self.bins), 1024)).astype(bool)
        #
        # for i in xrange(len(self.bins)):
        #     self.bin_matrix[i, self.bins[i][0]:(self.bins[i][0 - 1] + 1)] = True

    def ingest(self, spec):

        bin_max = 600

        s = spec

        # this is the bottle neck ~130ms
        binnedSpec = np.zeros(len(self.bins))
        # for i in range(len(self.bins)):
        #     binnedSpec[i] = sum(s[self.bins[i]])

        for i in range(len(self.bins)):
            if self.bins[i][0] < bin_max:
                binnedSpec[i] = s[self.bins[i][0]:(self.bins[i][0-1]+1)].sum()

        # for i in range(len(self.bins)):
        #     if self.bins[i][0] > bin_max:
        #         continue
        #     else:
        #         binnedSpec[i] = s[self.bins[i][0]:(self.bins[i][0 - 1] + 1)].sum()

        # print(len(s))

        # this takes 12.5 sec - slower
        # binnedSpec = np.sum(np.tile(s, [len(self.bins), 1]) * self.bin_matrix, 1)



        self.S.inject(binnedSpec)
        Sfil = self.S.get()


        # Sfil = np.zeros(len(binnedSpec))
        # for S in self.S:
        #     S.inject(binnedSpec)
        #     Sfil = np.maximum(Sfil,S.get())

        self.BEst.ingest(binnedSpec)
        Bfil = np.array(self.BEst.Bfil.get())


        self.SNR = np.zeros(len(self.bins))
        gauss_ind = (np.where(Bfil >= 0.1)[0]).astype(int)


        if len(gauss_ind) > 0:
            r = 1/(0.8/1023 * self.nchan + 1.01)  # Normalizing function (done at 100 cps)
            self.SNR[gauss_ind] = r[gauss_ind] * np.sqrt(2*self.kS)*(Sfil[gauss_ind] - Bfil[gauss_ind]) / np.sqrt(Sfil[gauss_ind] + Bfil[gauss_ind])
            self.cumSNR = np.sum((Sfil[gauss_ind] - Bfil[gauss_ind]) / np.sqrt(Bfil[gauss_ind]))
        else:
            self.cumSNR = 0.0

        return self.SNR

    def get(self):
        return self.SNR

    def getB(self):
        return self.BEst.Bfil.get()

    def getS(self):
        return self.S.get()
        # Sfil = np.zeros(len(self.bins))
        # for S in self.S:
        #     Sfil = np.maximum(Sfil, S.get())
        # return Sfil

    def getCumulativeSNR(self):
        return self.cumSNR




class iso_perceptron:

    def __init__(self, nElem):
        w = waveletBinTree(nElem)
        self.bins = w.flatList()
        self.Wt = np.zeros(len(self.bins))
        self.Y = 0.0
        self.SNR = np.zeros(len(self.bins))

    def ingest(self, snr):
        self.SNR = snr
        yi = np.sum(self.Wt * self.SNR)
        self.Y = 1.0 / (1 + np.exp(-1.0 * (yi)))  #Sigmoid

    def train(self, dout, accel=0.001):
        self.Wt = self.Wt + accel * (dout - self.Y) * self.SNR

    def get(self):
        return self.Y

    def getWt(self):
        return self.Wt

    def setWt(self,wts):
        self.Wt = wts

    def getSNR(self):
        return self.SNR






