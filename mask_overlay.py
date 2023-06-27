#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:41:42 2023

@author: dev
"""

import os
import cv2
import numpy as np
from marginalize import boundary
from tqdm import tqdm

base_dir = './ISIC 2016 for maskgen/'
img_dir = base_dir+'IMAGES/'
gen_dir = base_dir+'MASKS/BW/'
gt_dir = base_dir+'MASKS/REFERENCE/'
unet_dir = base_dir+'MASKS/UNET MASKS/'
out_dir = base_dir+'OVERLAID/COMPARED/'
kern = 9

for file in tqdm(os.listdir(img_dir), total=1279):
    img = cv2.imread(img_dir+file)
    gt_mask = boundary(cv2.imread(gt_dir+file, cv2.IMREAD_GRAYSCALE), kern=kern)
    z = np.zeros_like(gt_mask)
    gt_mask = cv2.merge((z, z, gt_mask))
    
    gen_mask = boundary(cv2.imread(gen_dir+file, cv2.IMREAD_GRAYSCALE), kern=kern)
    gen_mask = cv2.merge((z, gen_mask, z))
    
    unet_mask = boundary(cv2.imread(unet_dir+file, cv2.IMREAD_GRAYSCALE), kern=kern)
    unet_mask = cv2.merge((unet_mask, unet_mask, unet_mask))
    
    overlay = cv2.addWeighted(img, 1, gt_mask, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 1, gen_mask, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 1, unet_mask, 0.3, 0)
    
    if not cv2.imwrite(out_dir+file, overlay):
        raise Exception('File not saved : ', file)
