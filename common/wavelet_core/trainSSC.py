#! /usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np
from isoPerceptron import *
from inject import *


if __name__ == '__main__':

    import argparse
    from os import listdir
    from os.path import isfile, join

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename','-f', type=str,required=False, help='input background file name')
    parser.add_argument('--injectListFile','-i', type=str,required=False, help='input inject list file name', default=None)
    parser.add_argument('--startIndex','-s', type=int,required=False, help='Index to begin with...', default=0)
    parser.add_argument('--endIndex','-e', type=int,required=False, help='Index to end with...', default=1e10)
    parser.add_argument('--plotData', action='store_true', help='plotting enabled', default=False)
    parser.add_argument('--training', type=bool, help='training enabled', default=True)
    parser.add_argument('--wtVectorFile','-w', type=str,required=False, help='input path with weights',default=None)
    parser.add_argument('--wtVectorOutput','-o', type=str,required=False, help='output path with weights',default=None)
    parser.add_argument('--integrationTime','-t', type=int, required=False, help='List of integration times', default=1)
    parser.add_argument('--gapTime','-g', type=int,required=False, help='Gap Time ', default=3)
    parser.add_argument('--backgroundIntegration','-b', type=int,required=False, help='Background integration time ', default=30)
    parser.add_argument('--sourceAttenuation','-a', type=float,required=False, help='Attentuation factor for injection ', default=1.0)


    args = parser.parse_args()


    intTime = args.integrationTime
    # for it in intTimes:
    if intTime < 1:
        print 'Warning: Integration Time of ' + str(intTime) + ' is not valid.'
        exit()

    gTime = args.gapTime
    if gTime < 1:
        print 'No valid gap time...exiting.'
        exit()

    bTime = args.backgroundIntegration
    if bTime < 1:
        print 'No valid background integration time...exiting.'
        exit()

    aTime = args.sourceAttenuation
    if aTime > 1 or aTime <= 0.0:
        print 'Valid attentuation 0 < t <= 1..exiting.'
        exit()

    plotData = args.plotData

    if plotData:
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)


    trainingCooldown = 5*bTime

    LLD = 30
    R = injectionRun(args.filename,3*bTime,attenuation=aTime)

    R.run(startIndex=args.startIndex)
    F = isoSNRFeature(1024,bTime,gTime,intTime)


    iFiles = []
    P = {}
    I = {}
    Y = {}


    if args.injectListFile is not None:
        with open(args.injectListFile,'r') as ifile:
            for s in ifile:
                if len(s) > 0:
                    items = str.split(s)
                    iFiles.append((items[0],float(items[1])))
                    # Perceptron classifier
                    if items[0] not in P:
                        P[items[0]] = {}
                        Y[items[0]] = {}
                        P[items[0]] = iso_perceptron(1024)
                        Y[items[0]] = np.zeros(R.BFile.getSpecNumber())
                    R.addInjectFile(items[0],float(items[1]))

    if args.wtVectorFile is not None:
        with open(args.wtVectorFile, 'r') as wfile:
            reader = csv.reader(wfile)
            for row in reader:
                key = row[0]
                wt = [float(n) for n in row[1:]]
                if key in P:
                    P[key].setWt(wt)

    spec = R.nextSpectrum()

    cooldownFlag = False


    totalSpec = R.BFile.getSpecNumber()
    trainingCooldownTime = min(args.endIndex,totalSpec) - trainingCooldown
    print "Training Cooldown Time..." + str(trainingCooldownTime)
    while spec is not None and R.BFile.getIndex() < args.endIndex:

        print "Index: " + str(R.BFile.getIndex())

        SNR = F.ingest(spec)

        # Ensure no injects during cooldown and during end of training profile...
        if not R.cooldownActive() and R.BFile.getIndex() < trainingCooldownTime:
            injectKey = R.injectActive()

            if injectKey is None:
                injectKey = ''
            else:
                if cooldownFlag and args.wtVectorOutput is not None:
                    with open(args.wtVectorOutput,'w') as wtFile:
                        print 'Writing weights to: ' + args.wtVectorOutput
                        cooldownFlag = False

                        for key in P:
                            pwt = P[key].getWt()
                            wtFile.write(key + "," + ",".join(['%.6f' % num for num in pwt]) + '\n')

            for key in P:
                P[key].ingest(SNR)
                if args.training:
                    if injectKey == key:
                        P[key].train(1.0)
                    else:
                        P[key].train(0.0)

                Y[key][R.BFile.getIndex() - 1] = P[key].get()

        else:
            cooldownFlag = True

        spec = R.nextSpectrum()  # For next time around...

        if plotData:


            ax2.cla()
            Sfil = np.array(F.getS())
            Bfil = np.array(F.getB())

            C = np.arange(len(SNR))
            gauss_ind = (np.where(Bfil >= 0.5)[0]).astype(int)
            poiss_ind = (np.where(np.logical_and(Bfil < 10.0, Bfil > 0.0))[0]).astype(int)

            ax2.plot(C[poiss_ind], SNR[poiss_ind],'g.')
            ax2.plot(C[gauss_ind],SNR[gauss_ind],'r.')
            ax2.set_ylim([0,10])

            ax1.cla()
            ax1.plot(C, Sfil,'r.')
            ax1.plot(C, Bfil,'k.')
            ax1.plot(C, np.abs(Bfil - Sfil))

            ax3.cla()
            for key in P:
                ax3.plot(C,P[key].getWt())

            ax4.cla()
            for key in Y:
                if R.BFile.getIndex() > 301:
                    ind = np.arange(R.BFile.getIndex()-301,R.BFile.getIndex()-1)
                else:
                    ind =  np.arange(0,R.BFile.getIndex()-1)
                y = Y[key][ind]
                ax4.plot(ind,y)

                if R.BFile.getIndex() > 301:
                    ax4.set_xlim([np.min(ind),np.max(ind)])
                else:
                    ax4.set_xlim([0,300])

                ax4.set_ylim([0,1.0])
            plt.pause(0.01)



