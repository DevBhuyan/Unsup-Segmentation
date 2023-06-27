#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 04:11:37 2023

@author: dev
"""

import numpy as np
from GLCM import GLCM
from datetime import datetime

def cslbp_val(x, y, f):
    cslbp_bin = np.zeros((3, 3), dtype=np.uint8)
    exp_mat = np.array([[8, 4, 2],
                        [0, 0, 1],
                        [0, 0, 0]], dtype=np.uint8)

    cslbp_bin[0, 0] = int(f[x-1, y-1] > f[x+1, y+1])
    cslbp_bin[0, 1] = int(f[x, y-1] > f[x, y+1])
    cslbp_bin[0, 2] = int(f[x+1, y-1] > f[x-1, y+1])
    cslbp_bin[1, 2] = int(f[x+1, y] > f[x-1, y])

    cslbp_mat = exp_mat * cslbp_bin
    cslbp_out = np.sum(cslbp_mat)

    return cslbp_out


def CSLBP(f):
    cslbp = np.zeros((f.shape[0]-2, f.shape[1]-2), dtype=np.uint8)

    for x in range(1, f.shape[0]-1):
        for y in range(1, f.shape[1]-1):
            cslbp[x-1, y-1] = cslbp_val(x, y, f)

    return cslbp


def hist(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    histogram = histogram.astype(float) / np.sum(histogram)
    feature_vector = histogram.flatten()
    return feature_vector


def vect(img):
    cslbp = CSLBP(img)
    glcm = GLCM(cslbp)
    return glcm