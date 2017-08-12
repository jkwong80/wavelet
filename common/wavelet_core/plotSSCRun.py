#! /usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np
from isoPerceptron import *
from inject import *
import pickle
import matplotlib.cm as cm



if __name__ == '__main__':

    import argparse
    from os import listdir
    from os.path import isfile, join

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename','-f', type=str,required=True, help='output pickle file name')


    args = parser.parse_args()

    fname = args.filename


    with open(fname,'r') as fin:
        D = pickle.load(fin)
        T = D[0]
        Y = D[1]


    fig, (ax1,ax2) = plt.subplots(2)
    colors = cm.rainbow(np.linspace(0, 1, len(Y)))
    n = -1

    print Y

    for key in Y:
        n = n+1
        ind = np.arange(len(Y[key]))
        ax1.plot(ind,Y[key],color=colors[n])
        ax1.plot(ind,T[key],'--',color=colors[n])
    plt.show()







