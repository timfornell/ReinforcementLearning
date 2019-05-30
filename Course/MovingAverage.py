#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:12:45 2019

@author: markusenberg
"""

import numpy as np


def MovingAverage(a, n=3):
    best = np.array(a)
    ret = np.zeros((best.size-n+1,1))
    for i in range(best.size-n+1):
        ret[i] = np.sum(best[i:i+n])/n
    return ret

if __name__=="__main__":
    a = np.arange(20)
    result_array = MovingAverage(a)
    a = np.array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
        12.,  13.,  14.,  15.,  16.,  17.,  18.])
    
    result_array2 = MovingAverage(a, n=4)
    correct_array = np.array([  2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,   9.5,
        10.5,  11.5,  12.5,  13.5,  14.5, 15.5, 16.5])