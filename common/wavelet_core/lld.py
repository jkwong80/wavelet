#! /usr/bin/env python
# -*- coding: utf-8 -*-




import numpy as np

def lld(E, mid, width):

    a = 0
    b = 1.0
    c = mid
    r = 5*(b-a)/width
    f = (a*np.exp(c*r) + b*np.exp(r*E))/(np.exp(c*r) + np.exp(r*E))
    return f

