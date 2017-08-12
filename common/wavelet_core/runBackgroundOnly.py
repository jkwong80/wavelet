#! /usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np
from isoPerceptron import *
from inject import *
import pickle
from numpy.random import poisson

if __name__ == '__main__':

    import argparse
    from os import listdir
    from os.path import isfile, join

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename','-f', type=str,required=False, help='input background file name')
    parser.add_argument('--startIndex','-s', type=int,required=False, help='Index to begin with...', default=0)
    parser.add_argument('--endIndex','-e', type=int,required=False, help='Index to end with...', default=1e10)
    parser.add_argument('--plotData', action='store_true', help='plotting enabled', default=False)
    parser.add_argument('--outputFile','-o', type=str,required=False, help='output file',default=None)
    parser.add_argument('--integrationTime', '-t', type=int, required=False,
                        help='List of integration times', default=1)
    parser.add_argument('--gapTime', '-g', type=int, required=False, help='Gap Time ', default=3)
    parser.add_argument('--backgroundIntegration', '-b', type=int, required=False, help='Background integration time ',
                        default=30)


    args = parser.parse_args()

    intTime = args.integrationTime

    if intTime < 1:
        print 'ERROR: Integration Time of ' + intTime + ' is not valid...exiting'
        exit()


    gTime = args.gapTime
    if gTime < 1:
        print 'No valid gap time...exiting.'
        exit()

    bTime = args.backgroundIntegration
    if bTime < 1:
        print 'No valid background integration time...exiting.'
        exit()


    plotData = args.plotData

    if plotData:
        fig, ax1 = plt.subplots(1)

    LLD = 30
    R = injectionRun(args.filename, 3 * bTime)

    R.run(startIndex=args.startIndex)
    F = isoSNRFeature(1024, bTime, gTime, intTime)

    spec = R.nextSpectrum()
    spec = poisson(np.ones(len(F.bins))*5)

    x = np.arange(-10.0,10.0,0.1)

    H = []
    for i in range(len(F.bins)):
        H.append(np.zeros(len(x)))

    while spec is not None and R.BFile.getIndex() < args.endIndex:

        print "Index: " + str(R.BFile.getIndex())

        SNR = F.ingest(spec)
        spec = R.nextSpectrum()  # For next time around...
        spec = poisson(np.ones(len(F.bins))*5)

        if not R.cooldownActive():
            for i in range(len(SNR)):
                D = np.abs(SNR[i]-x)
                H[i][D==np.min(D)] = H[i][D==np.min(D)] + 1

    m1 = np.zeros(len(F.bins))
    m2 = np.zeros(len(F.bins))

    for i in range(len(F.bins)):
        p = H[i]/np.sum(H[i])
        m1[i] = np.sum(p * x)
        m2[i] = np.sum(p * np.power(x-m1[i],2.0))
        std = np.sqrt(m2)

    if args.outputFile is not None:
        with open(args.outputFile,'w') as Fout:
            pickle.dump(H,Fout)

    if plotData:
        ax1.plot(np.arange(len(m1)), m1, 'b')
        ax1.plot(np.arange(len(m1)), m1 + std, 'r--')
        ax1.plot(np.arange(len(m1)), m1 - std, 'r--')
        plt.show()








