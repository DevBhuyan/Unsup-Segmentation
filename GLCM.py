#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 04:17:24 2023

@author: dev
"""

import numpy as np


def glcm_val0(i, j, f):
    indices_i = np.where(f[:-1, :-1] == i)
    indices_j = np.where(f[:-1, 1:] == j)
    common_indices = np.intersect1d(indices_i, indices_j)

    return len(common_indices)

def glcm_val45(i, j, f):
    indices_i = np.where(f[1:, :-1] == i)
    indices_j = np.where(f[:-1, 1:] == j)
    common_indices = np.intersect1d(indices_i, indices_j)

    return len(common_indices)

def glcm_val90(i, j, f):
    indices_i = np.where(f[1:, :] == i)
    indices_j = np.where(f[:-1, :] == j)
    common_indices = np.intersect1d(indices_i, indices_j)

    return len(common_indices)

def glcm_val135(i, j, f):
    indices_i = np.where(f[1:, 1:] == i)
    indices_j = np.where(f[:-1, :-1] == j)
    common_indices = np.intersect1d(indices_i, indices_j)

    return len(common_indices)

def GLCM(cslbp):
    max_val = 15
    glcm0 = np.zeros((max_val+1, max_val+1), dtype=np.uint32)
    glcm45 = np.zeros((max_val+1, max_val+1), dtype=np.uint32)
    glcm90 = np.zeros((max_val+1, max_val+1), dtype=np.uint32)
    glcm135 = np.zeros((max_val+1, max_val+1), dtype=np.uint32)

    for i in range(16):
        for j in range(16):
            glcm0[i, j] = glcm_val0(i, j, cslbp)    # FIXME costs about 0.43s
            glcm45[i, j] = glcm_val45(i, j, cslbp)
            glcm90[i, j] = glcm_val90(i, j, cslbp)
            glcm135[i, j] = glcm_val135(i, j, cslbp)

    glcm = np.concatenate((
        glcm0.flatten(),
        glcm45.flatten(),
        glcm90.flatten(),
        glcm135.flatten()
        ))
    return glcm