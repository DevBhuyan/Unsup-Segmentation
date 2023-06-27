#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 23:44:13 2023

@author: dev
"""

import cv2
import os
import numpy as np
from texture_helpers import hist, vect
from color_clean import stabilize
from marginalize import marginalize, check_num_regions, boundary, fill_color
from datetime import datetime

def manhattan_distance(vector1, vector2):
    return np.sum(np.abs(vector1 - vector2))

def hdfv(img, mask):
    res = cv2.bitwise_and(img, img, mask = mask)
    b, g, r = cv2.split(res)
    print('Preparing histograms....')
    hdfv = [hist(b+g), hist(g+r), hist(r+b)]
    hdfv = np.array(hdfv).flatten()
    print('Histograms ready. Time = ', datetime.now())
    hdfv2 = [vect(b), vect(g), vect(r), vect(b+g), vect(g+r), vect(r+b)]
    hdfv2 = np.array(hdfv2).flatten()
    print('Vectors ready. Time = ', datetime.now())

    return np.concatenate((hdfv, hdfv2))

def merge(unfiltered, mask1, mask2):
    flag = 0
    for i in range(224):
        for j in range(224):
            if mask1[i, j]:
                start_pos = [i, j]
                flag = 1
                break
        if flag:
            break
    color = unfiltered[i, j]
    for i in range(224):
        for j in range(224):
            if mask2[i, j]:
                unfiltered[i, j] = color
    return unfiltered

def rm(unfiltered, img):
    num_regions, masks = check_num_regions(unfiltered)
    print('num regions: ', num_regions)
    adj_hdfvs = [hdfv(img, mask2) for mask2 in masks]
    while len(masks) > 2:
        mask_hdfv = adj_hdfvs[0]
        distances = [manhattan_distance(mask_hdfv, adj_hdfvs[j]) for j in range(1, len(adj_hdfvs))]
        print(distances)
        closest_index = distances.index(min(distances))
        unfiltered = merge(unfiltered, masks[closest_index+1], masks[0])
        new_mask = cv2.bitwise_or(masks[0], masks[closest_index+1])
        new_hdfv = hdfv(img, new_mask)
        masks.pop(0)
        adj_hdfvs.pop(0)
        masks.pop(closest_index)
        adj_hdfvs.pop(closest_index)
        masks.append(new_mask)
        adj_hdfvs.append(new_hdfv)
        print('len(new_hdfv)', len(new_hdfv))
        # cv2.imshow('merging regions', unfiltered)
        # cv2.waitKey(1)

    return unfiltered

img_src = './ISIC 2016 for maskgen/IMAGES/'
mask_src = './ISIC 2016 for maskgen/MASKS/UNFILTERED/'
filter_dest = './ISIC 2016 for maskgen/MASKS/FILTERED/'
bw_dest = './ISIC 2016 for maskgen/MASKS/BW/'

if __name__ == "__main__":
    for file in os.listdir(img_src):
        img = cv2.imread(img_src+file, cv2.IMREAD_COLOR)
        unfiltered = cv2.imread(mask_src+file, cv2.IMREAD_COLOR)
        unfiltered = stabilize(unfiltered)
        thresh = 0.001
        while thresh <= 0.032:
            unfiltered = marginalize(unfiltered, thresh)
            # cv2.imshow('marginalizing', unfiltered)
            # cv2.waitKey(1)
            thresh *= 2
            print(thresh)
        # REGION MERGING
        # cv2.imshow('before rm', unfiltered)
        # cv2.waitKey(1)
        filtered = rm(unfiltered, img)
        cv2.imwrite(filter_dest+file, filtered)
        pixels = {}

        for i in range(224):
            for j in range(224):
                if not i or not j or i == 223 or j == 223:
                    try:
                        pixels[(filtered[i, j])[0]] += 1
                    except:
                        pixels[(filtered[i, j])[0]] = 1
        b = 0
        for color in pixels:
            if pixels[color] > b:
                b = pixels[color]
                c = color
        bg_color = [c, c, c]
        for i in range(224):
            for j in range(224):
                if (filtered[i, j] == [0, 0, 0]).all() or (filtered[i, j] == [255, 255, 255]).all():
                    continue
                elif (filtered[i, j] == bg_color).all():
                    filtered[i, j] = [0, 0, 0]
                else:
                    filtered[i, j] = [255, 255, 255]
        break
        cv2.imwrite(bw_dest+file, filtered)