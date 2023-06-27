#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 00:35:38 2023

@author: dev
"""

# TODO Write a function which recieves a bounding box and an image. The function must find all the regions in the image. Then for each mask, check if any pixel in that mask lies outside the bounding boxes. If yes, turn it black. At the end, whatever is not black, turn it white

import cv2
from marginalize import check_num_regions, fill_color, marginalize
from tqdm import tqdm
from color_clean import stabilize
import os

def bb_from_mask(mask):
    r0 = r1 = -1
    for i in range(224):
        if max(mask[i]) and r0 == -1:
            r0 = i
        if max(mask[223-i]) and r1 == -1:
            r1 = 223-i
    c0 = c1 = -1
    mask = mask.T
    for i in range(224):
        if max(mask[i]) and c0 == -1:
            c0 = i
        if max(mask[223-i]) and c1 == -1:
            c1 = 223-i
    return r0, r1, c0, c1

def check_if_inside(unfiltered, bb):
    r0, r1, c0, c1 = bb[1], bb[2], bb[3], bb[4]
    num_regions, masks = check_num_regions(unfiltered)
    for region in masks:
        x0, x1, y0, y1 = bb_from_mask(region)
        if x0 < r0 or x1 > r1 or y0 < c0 or y1 > c1:
            unfiltered = fill_color(unfiltered, [0, 0], [0, 0, 0], mask = region)
        else:
            unfiltered = fill_color(unfiltered, [0, 0], [255, 255, 255], mask = region)
    return unfiltered

def bb_to_bw(bb, unf_src, bw_dest):
    listdir = os.listdir(bw_dest)
    for row in tqdm(bb, total = len(bb)):
        file = row[0]
        if os.path.exists(unf_src+file):
            if file in listdir:
                continue
            unfiltered = stabilize(cv2.imread(unf_src+file, cv2.IMREAD_COLOR))
            thresh = 0.001
            while thresh <= 0.04:
                unfiltered = marginalize(unfiltered, thresh)
                thresh *= 2
            bw = check_if_inside(unfiltered, row)
            if not cv2.imwrite(bw_dest+file, bw):
                raise Exception('File not saved')
    return