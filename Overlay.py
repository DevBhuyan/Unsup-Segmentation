#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from tqdm import tqdm

base_dir = './ISIC 2016 for maskgen (backup)/'

# HACK
img_src = base_dir+'IMAGES/'
msk_src = base_dir+'MASKS/UNET MASKS/'
dest = base_dir+'SEGMENTED/'

for file in tqdm(os.listdir(img_src), total=1279):
    img = cv2.imread(img_src+file)
    mask = cv2.imread(msk_src+file, cv2.IMREAD_GRAYSCALE)
    (thresh, mask2) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img, img, mask = mask2)
    cv2.imwrite(dest+file, res)