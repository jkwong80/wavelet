#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from collections import deque
import sys
import os
from bins import *
import time

class EWMA:
    def __init__(self, k, nelem):
        self.basek = k
        self.k = np.array(np.ones(nelem)*k)
        self.accum = np.zeros(nelem)
        self.var = np.zeros(nelem)


    def setK(self, k):
        self.k = float(k)

    def getK(self):
        return self.k

    def resetK(self, vec):
        if len(vec) == len(self.k):
            va = self.k
            self.k[vec>0.0] = self.basek

    def inject(self, A):
        prior = self.accum
        self.accum = (1.0 - 1.0/self.k)*self.accum + A/self.k
        self.var = np.abs((1.0 - 1.0 / self.k) * self.var + np.power((A-prior), 2.) / self.k)

    def get(self):
        return self.accum

    def getVariance(self):
        return self.var

    def set(self, mu, var):
        self.accum = mu
        self.var = var

class delay:
    def __init__(self, kdelay):
        self.buffer = deque()
        self.delay = kdelay

    def push(self,inp):
        if len(self.buffer) <= self.delay:
            self.buffer.appendleft(inp)
            return None
        else:
            ret = self.buffer.pop()
            self.buffer.appendleft(inp)
            return ret

    def setDelay(self,d):
        self.delay = d

    def getDelay(self):
        return self.delay

    def clear(self):
        self.buffer.clear()

    def isFull(self):
        return len(self.buffer)==self.buffer.maxlen


class backgroundEstimation:

    def __init__(self,kb,kdelay,nelem):
        self.derivativeThresh = 5.0
        self.fastK = 2
        self.midK = 6
        self.longK = 12
        self.nelem = nelem
        self.fast = EWMA(self.fastK, 1)
        self.midDelay = delay(self.fastK)
        self.mid = EWMA(self.midK, 1)
        self.longDelay = delay(self.fastK + self.midK)
        self.long = EWMA(self.longK, 1)
        self.reset = True
        self.derivative = np.zeros(1)
        self.curvature = np.zeros(1)
        self.startupTimer = 120

        self.BDelay = delay(kdelay)
        self.Bfil = EWMA(kb,nelem)
        self.sourceStatus = False

    def ingest(self, Arr):

        # print '1: ' + str((time.time()*1000) % 1000)
        if Arr is None:
            return

        Arr = np.array(Arr)

        if not len(Arr) == self.nelem:
            print 'WARNING: input shape is not correct!'
            return

        gc = np.sum(Arr)
        # Reset Filters
        if self.reset:
            self.fast.set(gc, gc)
            self.mid.set(gc, gc)
            self.long.set(gc, gc)
            self.reset = False
        # print '2: ' + str((time.time()*1000) % 1000)

        self.fast.inject(gc)
        # print '3: ' + str((time.time()*1000) % 1000)

        midOut = self.midDelay.push(gc)
        if midOut is not None:
            self.mid.inject(midOut)
        longOut = self.longDelay.push(gc)
        if longOut is not None:
            self.long.inject(longOut)

        fast = np.array(self.fast.get())
        mid =  np.array(self.mid.get())
        long =  np.array(self.long.get())

        self.derivative = (fast-mid)/np.sqrt(3.0*self.fastK*fast + 3.0*self.midK*mid)
        self.curvature = (fast - 2*mid + long) /\
                              np.sqrt(3.0*self.fastK*fast + 3.0*self.midK*mid + 3.0*self.longK*long)

        # print '4: ' + str((time.time()*1000) % 1000)

        if np.abs(self.derivative) < 2.0 and self.startupTimer > 0:
            self.startupTimer -= 1

        if (not self.sourceStatus and np.abs(self.derivative) < 2.0) or self.startupTimer > 0:
            ret = self.BDelay.push(Arr)
            if ret is not None:
                self.Bfil.inject(ret)
        else:
            self.setReset()

        # print '5: ' + str((time.time()*1000) % 1000)


    def setReset(self):
        self.reset = True

    def setSourceStatus(self, status=True):
        self.sourceStatus = status



class spectralShapeChangeDetection:

    def __init__(self,kb,bdelay,ks,nelem):
        self.nelem = nelem
        self.bins = waveletBinTree(nelem)

        self.bEst = backgroundEstimation(kb,bdelay,len(self.bins.flatList()))
        self.sfil = EWMA(ks,len(self.bins.flatList()))
        self.reset = True

    def ingest(self,spec):
        if spec is None:
            return

        if not len(spec) == self.nelem:
            print 'WARNING: input shape is not correct!'
            return

        Arr = self.bins.binSpectrum(spec, normalized=True)
        Arr = np.array(Arr)

        # Reset Filters
        if self.reset:
            self.sfil.set(Arr, Arr)
            self.bEst.Bfil.set(Arr, Arr)
            self.bEst.BDelay.clear()
            self.reset = False

        self.sfil.inject(Arr)
        self.bEst.ingest(Arr)

    def getSource(self):
        return self.sfil.get()

    def getSourceVariance(self):
        return self.sfil.getVariance()

    def getBackgroundVariance(self):
        return self.bEst.Bfil.getVariance()

    def getBackground(self):
        return self.bEst.Bfil.get()

    def changeMetric(self):
        metric = np.zeros(len(self.bins.flatList()))

        D = (self.getSource()-self.getBackground())
        svar = self.getSourceVariance()
        bvar = self.getBackgroundVariance()

        ind = [i for i,j in enumerate(svar) if svar[i] > 0 or bvar[i] > 0]
        metric[ind] = D[ind]/np.sqrt(svar[ind] + bvar[ind])
        return metric


        # return pearsonr(self.sfil.get(), self.bEst.Bfil.get())


class spectralChangeDetection:

    def __init__(self,ks,kb,kdelay,nelem,backgroundInjectThresh):

        self.bfil = EWMA(kb,nelem)
        self.bdelay = delay(kdelay)
        self.sfil = EWMA(ks,nelem)
        self.reset = True
        self.nelem = nelem
        self.bthresh = backgroundInjectThresh

    def ingest(self, Arr):

        if Arr is None:
            return

        Arr = np.array(Arr)

        if not len(Arr) == self.nelem:
            print 'WARNING: input shape is not correct!'
            return

        # Reset Filters
        if self.reset:
            self.sfil.set(Arr, Arr)
            self.bfil.set(Arr, Arr)
            self.reset = False

        snr = np.array(self.getSNR())

        self.sfil.inject(Arr)
        ret = self.bdelay.push(Arr)
        if self.bdelay.isFull():
            if max(snr) < self.bthresh:
                self.bfil.inject(ret)

    def setReset(self):
        self.reset = True

    def getSourceFilter(self):
        f,v = self.sfil.get()
        return f

    def getBackgroundFilter(self):
        f,v = self.bfil.get()
        return f

    def getSNR(self):

        bf, bvar = self.bfil.get()
        sf, svar = self.sfil.get()

        snr = np.zeros(len(sf))
        ind = [i for i,j in enumerate(list(bf)) if bf[i]>0.0]

        # SNR = (N-B)/std(B)
        #Estimate EWMA filter as 3 x count rate x filter coef
        stdB = np.sqrt(3.0*self.bfil.k * bf)

        snr[ind] = (sf[ind]-bf[ind])/stdB[ind]
        return snr

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
    import os
    import cPickle as pickle
    import argparse
    import csv
    from bins import *
    import matplotlib.pyplot as plt



    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', type=str,required=False, help='input csv file')
    parser.add_argument('--output','-o', type=str, required=False, help='output csv file')
    parser.add_argument('--maxSNR','-m',action='store_true', help='Output maxSNR only',default=True)
    parser.add_argument('--plot','-p',action='store_true', help='Running plot of results',default=False)
    parser.add_argument('--plotInputCountRate',action='store_true', help='Running plot of results',default=False)

    parser.add_argument('-t', '--timeWindow', nargs='+', type=float, help='time window in sec (t1, t2)', required=False)
    args = parser.parse_args()


    if args.input is not None and not args.plotInputCountRate:

        writer = None
        if args.output is not None:
            fout = open(args.output, 'wb')
            writer = csv.writer(fout)

        print writer

        t1,t2 = [None,None]
        if args.timeWindow is not None:
            if len(args.timeWindow) == 2:
                t1,t2 = args.timeWindow
            else:
                print "ERROR: Time Window requires two values: [t1,t2]"
                exit()

        plotFlag = args.plot


        if args.plot:
            plt.ion()
            plt.figure()
            f, axarr = plt.subplots(4)

        wavelet = waveletBinTree(128)
        scd = spectralChangeDetection(8,40,8,len(wavelet.ind),1.0)
        fsize = os.path.getsize(args.input)

        currentTime = None
        with open(args.input,'rb') as f:
            reader = csv.reader(f)
            for row in reader:

                t = float(row[3])
                if t1 is not None:
                    if t < t1:
                        continue
                    elif t > t2:
                        exit()

                if currentTime is None:
                    currentTime = t-1.0
                if t > (currentTime+20.0):
                    scd.setReset()
                    print 'reset filters'
                currentTime = t

                s = np.array(list(map(int, row[4:])))
                w = wavelet.binSpectrum(s)

                scd.ingest(w)
                SNR = scd.getSNR()
                maxInd = [i for i,j in enumerate(SNR) if SNR[i] > 1.0]
                if len(maxInd) > 0:
                    data = [t] + [max(SNR),maxInd,SNR[maxInd]]
                    if writer is not None:
                        writer.writerow(data)
                        print data

                pos = f.tell()
                progress(pos,fsize,prefix='Read Progress: ', suffix='Complete', barlength=50)

                if args.plot:

                    x = range(len(w))
                    axarr[0].clear()
                    axarr[0].plot(x,w)
                    # axarr[0].set_title('Sharing X axis')

                    axarr[1].clear()
                    axarr[1].plot(x, scd.getSourceFilter(),color='r')
                    axarr[1].plot(x, scd.getBackgroundFilter(),color='k')

                    axarr[2].clear()
                    # axarr[2].plot(x, snrCNT,color='b')
                    axarr[2].plot(x, SNR, color='g')
                    for mx in maxInd:
                        axarr[2].plot([mx,mx],[0,max(SNR)],color='r')

                    axarr[3].clear()
                    # axarr[2].plot(x, snrCNT,color='b')
                    axarr[3].plot(x, scd.sfil.k, color='r')
                    axarr[3].plot(x, scd.bfil.k, color='k')


                    plt.draw()
                    plt.pause(0.1)

    elif args.output is not None and args.input is None:

        # General plotting of output...needs to change with different output formats!!!
        S = []
        t = []
        with open(args.output,'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                t.append(float(row[0]))
                S.append(float(row[1]))

        plt.figure()
        plt.plot(t,S)
        plt.show()

    elif args.plotInputCountRate and args.input is not None:

        # General plotting of input stuff...custom...

        R = []
        stamp = []
        dt = []
        cT = None

        t1, t2 = [None, None]
        if args.timeWindow is not None:
            if len(args.timeWindow) == 2:
                t1, t2 = args.timeWindow
            else:
                print "ERROR: Time Window requires two values: [t1,t2]"
                exit()

        with open(args.input, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                t = float(row[3])
                if cT is None:
                    cT = t
                dt.append(t-cT)
                cT = t

                if t1 is not None:
                    if t < t1:
                        continue
                    elif t > t2:
                        exit()
                stamp.append(t)
                s = np.array(list(map(int, row[4:])))
                R.append(np.sum(s))

        plt.figure()
        plt.plot(range(len(R)), R)
        # plt.plot(range(len(dt)),dt)
        plt.show()
