#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 20:14:22 2023

@author: dev
"""

import cv2
import os
from tqdm import tqdm
from bb_to_bw import bb_to_bw, bb_from_mask
from metrics import metrics
import pickle

def threshold(mask):
    for i in range(224):
        for j in range(224):
            if mask[i, j] <= 127:
                mask[i, j] = 0
            else:
                mask[i, j] = 255
    return mask

def bb():
    src = './ISIC 2016 for maskgen/MASKS/REFERENCE/'
    unf_src = './ISIC 2016 for maskgen/MASKS/UNFILTERED/'
    bw_dest = './ISIC 2016 for maskgen/MASKS/BW/'
    tolerance = 10
    if not os.path.exists(bw_dest):
        os.makedirs(bw_dest)
    bb = []
    if os.path.exists('test'):
        with open("test", "rb") as fp:
            bb = pickle.load(fp)
    else:
        for file in tqdm(os.listdir(src), total = 1279):
            mask = threshold(cv2.imread(src+file, cv2.IMREAD_GRAYSCALE))
            r0, r1, c0, c1 = bb_from_mask(mask)
            bb.append([file, r0-tolerance, r1+tolerance, c0-tolerance, c1+tolerance])
        with open("test", "wb") as fp:
            pickle.dump(bb, fp)
    bb_to_bw(bb, unf_src, bw_dest)

    metrics()
    return

if __name__ == "__main__":
    bb()