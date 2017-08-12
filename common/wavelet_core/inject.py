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
import random


class backgroundFile:

    def __init__(self, fname, startIndex=0, dataset='sensor_90_degrees/spectrum'):
        self.backgroundFile = fname
        self.spectrum = None

        try:
            with h5py.File(fname, 'r') as hf:
                self.spectrum = hf[dataset][:]
        except:
            print('Background File not Opened...')

        self.index = startIndex

    def loadSpectrum(self, startIndex=0, dataset='sensor_90_degrees/spectrum'):
        self.spectrum = None
        try:
            with h5py.File(self.backgroundFile, 'r') as hf:
                self.spectrum = hf[dataset][:]
        except:
            print('Spectrum failed to load...')

        self.index = startIndex

    def loadBackgroundFile(self, fname, dataset=None):
        self.backgroundFile = fname

    def setIndex(self,ind):
        self.index = ind

    def getIndex(self):
        return self.index

    def getSpecNumber(self):
        return np.shape(self.spectrum)[0]

    def nextSpectrum(self):
        if self.spectrum is None:
            return None

        sh = np.shape(self.spectrum)

        if self.index < sh[0]:
            s = self.spectrum[self.index][:]
            self.index = self.index + 1
            return s
        else:
            return None

class injectFile:

    def __init__(self, fname, LLD):
        self.injectFile = fname
        self.spectrum = None
        self.injectionStarts = {}
        self.maxInjectionIndex = 0
        self.injectSpectrum = None
        E = np.arange(1024) * 3.0;
        self.FLLD = np.ones(len(E))
        self.FLLD[E < 100] = lld(E[E < 100], LLD, 10)

        try:
            with h5py.File(fname, 'r') as Finj:
                for key in Finj:
                    sIndex = Finj[key].attrs['Start_Index']
                    self.injectionStarts[sIndex] = key + '/Inj_' + key
        except:
            print "ERROR: opening inject file: " + fname

    def getInjectData(self, ind):

        if ind in self.injectionStarts and self.injectSpectrum is None:
            with h5py.File(self.injectFile, 'r') as Finj:
                self.injectSpectrum = Finj[self.injectionStarts[ind]][:]
                print "Injection Started: " + self.injectFile

        if self.injectSpectrum is None:
            return None

        sh = np.shape(self.injectSpectrum)

        spec = self.injectSpectrum[0, :] * self.FLLD

        if sh[0] > 1:
            self.injectSpectrum = self.injectSpectrum[1:]
        else:
            self.injectSpectrum = None

        return spec

    def cooldownActive(self):
        if self.cooldownTimer > 0:
            return True
        else:
            return False


    def injectActive(self):
        if self.injectSpectrum is None:
            return False
        else:
            return True

class injectionRun:

    def __init__(self, bkgFile, cooldownTime, attenuation=1.0):

        self.BFile = backgroundFile(bkgFile)
        self.injectFiles = {}
        self.LLD = 30  #keV
        self.coolDownTime = cooldownTime
        self.cooldownTimer = cooldownTime
        self.attenuation = attenuation

    def addInjectFile(self, iFile, pSample):
        self.injectFiles[iFile] = (pSample, injectFile(iFile,30.0))

    def setSampleInjectProbablility(self, iFile, pSample):
        if iFile in self.injectFiles:
            self.injectFiles[0] = pSample

    def run(self, startIndex=0):
        self.BFile.setIndex(startIndex)
        self.cooldownTimer = self.coolDownTime

    def injectActive(self):
        for key in self.injectFiles:
            if self.injectFiles[key][1].injectActive() is True:
                return key
        return None

    def cooldownActive(self):
        if self.cooldownTimer > 0 and self.injectActive() is None:
            return True
        else:
            return False

    def nextSpectrum(self):

        if self.cooldownTimer > 0:
            self.cooldownTimer = self.cooldownTimer - 1

        # print 'Index: ' + str(self.BFile.getIndex()-1)

        bspec = self.BFile.nextSpectrum()

        if bspec is None:
            return None

        activeInjectKey = self.injectActive()

        if not self.cooldownActive():

            if activeInjectKey is not None:
                print "Injection Active: " + activeInjectKey
                idata = self.injectFiles[activeInjectKey][1].getInjectData(self.BFile.getIndex()-1)
            else:
                rand = random.random()

                # First sample individual files:
                iKeys = []
                for key in self.injectFiles:
                    if rand < self.injectFiles[key][0]:
                        iKeys.append(key)

                if len(iKeys) > 0:
                    ikey = iKeys[random.randint(0,len(iKeys)-1)]  # Random selection
                    idata = self.injectFiles[ikey][1].getInjectData(self.BFile.getIndex()-1)
                else:
                    idata = None

            if idata is not None:
                self.cooldownTimer = self.coolDownTime
        else:
            print 'Cooldown state: ' + str(self.cooldownTimer)
            idata = None

        if idata is not None:
            spec = bspec + idata*self.attenuation
        else:
            spec = bspec

        return spec











