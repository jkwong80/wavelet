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

def progress(iteration, total, prefix='', suffix='', decimals=1, barlength=50):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(barlength * iteration / float(total)))
    bar = '█' * filled_length + '-' * (barlength - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()



if __name__ == '__main__':

    import argparse
    from os import listdir
    from os.path import isfile, join

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename','-f', type=str,required=False, help='input background file name')
    parser.add_argument('--injects','-i', type=str,required=False, help='input inject file name', default=None)
    parser.add_argument('--outFile','-o', type=str,required=False, help='input inject file name', default=None)
    parser.add_argument('--startIndex','-s', type=int,required=False, help='Index to begin with...', default=None)
    parser.add_argument('--plotData', action='store_true', help='plotting enabled', default=False)
    parser.add_argument('--runningPlot', action='store_true', help='Running plot of results', default=False)
    parser.add_argument('--wtVectorFile','-w', type=str,required=False, help='input path with weights',default=None)


    args = parser.parse_args()

    plotData = args.plotData
    runningPlot = args.runningPlot

    w = waveletBinTree(1023)
    ind = w.flatList()
    print np.shape(ind)

    B = backgroundEstimation(30,5,len(ind))
    S = EWMA(3,len(ind))


    C = np.arange(len(ind))

    E = np.arange(1024)*3.0;
    LLD = 30  # keV
    FLLD = np.ones(len(E))
    FLLD[E<100] = lld(E[E<100],LLD,10)


    if runningPlot and plotData:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3)

    binnedSpec = np.zeros(len(ind))

    injectionStarts = {}


    maxInjectIndex = 0
    if args.injects is not None:
        Finj = h5py.File(args.injects, 'r')
        for key in Finj:
            sIndex = Finj[key].attrs['Start_Index']
            injectionStarts[sIndex] = key + '/Inj_' + key
            if sIndex > maxInjectIndex:
                maxInjectIndex = sIndex

    maxInjectIndex = maxInjectIndex + 100
    print 'Max Injection Index: ' + str(maxInjectIndex
                                        )
    if args.startIndex is None:
        startInd = 0
    else:
        startInd = args.startIndex


    injectSpectrum = None

    cooldownTimer = 90

    if args.wtVectorFile is not None:
        with open(args.wtVectorFile,'r') as win:
            reader = csv.reader(win)
            for row in reader:
                wt = [float(data) for data in row]
                print np.shape(wt)
    else:
        wt = np.zeros(len(ind))


    if args.outFile is not None:
        fout = open(args.outFile,'w')


    with h5py.File(args.filename,'r') as hf:
        spec = hf['sensor_90_degrees/spectrum']
        spectrum = spec[:]
        sh = np.shape(spectrum)

        Y = np.zeros(sh[0])
        for n in range(startInd,maxInjectIndex):

            if cooldownTimer > 0:
                cooldownTimer = cooldownTimer - 1

            s = spectrum[n][:]

            if n in injectionStarts and cooldownTimer == 0 and injectSpectrum is None:
                print 'Injecting: ' + injectionStarts[n] + ' at index: ' + str(n)
                injectSpectrum = Finj[injectionStarts[n]][:]
                shSpec = np.shape(injectSpectrum)
                midPoint = np.floor(shSpec[0]/2.0) + 1
                nInject = 0
                print 'Writing weighting vector out...'


            if injectSpectrum is not None:
                s = s + injectSpectrum[0,:]*FLLD
                nInject = nInject + 1

                if nInject == midPoint:
                    if args.wtVectorFile is None:
                        fout.write(str(1.0) + "," + ",".join(['%.3f' % num for num in wt]) + '\n')
                if np.shape(injectSpectrum)[0] > 1:
                    injectSpectrum = injectSpectrum[1:,:]
                else:
                    injectSpectrum = None
                    print 'Completed Injection...starting cooldown'
                    cooldownTimer = 90

            for i in range(len(ind)):
                binnedSpec[i] = sum(s[ind[i]])

            S.inject(binnedSpec)
            Sfil = S.get()
            B.ingest(binnedSpec)
            Bfil = np.array(B.Bfil.get())


            SNR = np.zeros(len(C))
            P = np.zeros(len(SNR))
            CDF = np.zeros(len(SNR))
            poiss_ind = (np.where(np.logical_and(Bfil < 10.0,Bfil > 0.0))[0]).astype(int)

            # if len(poiss_ind) > 0:
            #     P[poiss_ind] = np.exp(-1*Bfil[poiss_ind]) * np.power(Bfil[poiss_ind],binnedSpec[poiss_ind]) / factorial(binnedSpec[poiss_ind])

            gauss_ind = (np.where(Bfil >= 0.5)[0]).astype(int)

            if len(gauss_ind) > 0:
                # P[gauss_ind] = 1.0/np.sqrt(2*np.pi*Bfil[gauss_ind]) * np.exp(-1.0*np.power(binnedSpec[gauss_ind]-Bfil[gauss_ind],2.0)/(2.0*Bfil[gauss_ind]))
                # CDF[gauss_ind] = 0.5*(1.0 + erf((binnedSpec[gauss_ind]-Bfil[gauss_ind])/(np.sqrt(Bfil[gauss_ind])*np.sqrt(2.0))))
                SNR[gauss_ind] = np.abs(Sfil[gauss_ind]-Bfil[gauss_ind])/np.sqrt(Bfil[gauss_ind])

            if cooldownTimer == 0:
                # We are either in Background or Inject State
                if injectSpectrum is not None:
                    #Inject State
                    dout = 1.0
                    acc = 0.001
                else:
                    #Background State
                    dout = 0.0
                    acc = 0.001
                    if args.wtVectorFile is None:
                        fout.write(str(0.0) + "," + ",".join(['%.3f' % num for num in wt]) + '\n')

                SNRD = SNR
                yi = np.sum(wt * SNR)

                y = 1.0/(1 + np.exp(-1.0*(yi)))
                print '(' + str(n) + ') Current Output: ' + str(y)
                if args.wtVectorFile is None:
                    wt = wt + acc*(dout - y)*SNRD
                Y[n] = y

            # print binnedSpec[8736], Bfil[8736], P[8736], CDF[8736]
            if plotData:
                ax2.cla()
                ax2.plot(C[poiss_ind], SNR[poiss_ind],'g.')
                ax2.plot(C[gauss_ind],SNR[gauss_ind],'r.')
                ax2.set_ylim([0,10])

                ax1.cla()
                ax1.plot(C, Sfil)
                ax1.plot(C, Bfil)
                ax1.plot(C, np.abs(Bfil - Sfil))

                ax3.cla()
                if args.wtVectorFile is None:
                    ax3.plot(C,wt)
                else:
                    ax3.plot(np.arange(n),Y[:n])

                if cooldownTimer > 0:
                    plt.title('Cooldown State: ' + str(cooldownTimer))
                elif injectSpectrum is not None:
                    plt.title('Injection Active')
                else:
                    plt.title('Background: ' + str(n))

                plt.pause(0.01)



