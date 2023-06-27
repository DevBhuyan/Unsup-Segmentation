#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:48:40 2023

@author: dev
"""

import os
from marginalize import check_num_regions
import cv2
from tqdm import tqdm

src = './ISIC 2016 for maskgen (backup)/MASKS/REFERENCE/'

def threshold(mask):
    for i in range(224):
        for j in range(224):
            if mask[i, j] > 127:
                mask[i, j] = 255
            else:
                mask[i, j] = 0
    return mask

n = 0
for file in tqdm(os.listdir(src), total=1279):
    img = cv2.imread(src+file, cv2.IMREAD_GRAYSCALE)
    num_regions, _ = check_num_regions(img, bw=True)
    if num_regions > n:
        n = num_regions
print('Max no. of lesions = ', n)