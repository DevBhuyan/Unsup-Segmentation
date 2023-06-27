#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:24:47 2023

@author: dev
"""

import os
import cv2
from marginalize import area, check_num_regions, fill_color, boundary
import numpy as np


def convert(img, dst, file):
    h, w = img.shape[:2]
    if tuple(img[h//2, w//2].tolist()) == tuple(img[0, 0].tolist()) or tuple(img[h//2, w//2].tolist()) == tuple(img[h-1, w-1].tolist()) or not all(x == y == z == a for x, y, z, a in zip(img[0, 0], img[h-1, w-1], img[0, w-1], img[h-1, 0])):
        # print(img[h//2, w//2])
        # print(img[0, 0])
        # print(img[0, w-1])
        # print(img[h-1, 0])
        # print(img[h-1, w-1])
        print('Image may not have been properly cleaned....')
    num_regions, final_masks = check_num_regions(img)
    if num_regions == 2:
        bgmask, _ = area(img, [0, 0])
        lsmask = 255*np.ones((224, 224), dtype=np.uint8)-bgmask
        out = lsmask
    else:
        pixels = img.reshape(-1, img.shape[-1])
        unique_colors = np.unique(pixels, axis=0)
        if len(unique_colors) == 2:
            bgmask, _ = area(img, [0, 0])
            lsmask = 255*np.ones((224, 224), dtype=np.uint8)-bgmask
            out = lsmask
        else:
            while num_regions > 2:
                print('entered while loop')
                num_regions, final_masks = check_num_regions(img)
                mx = boundary(final_masks[1], True)
                index = 1
                for i, mask in enumerate(final_masks):
                    if not i:
                        continue
                    if boundary(mask, True) > mx:
                        mx = boundary(mask, True)
                        index = i
                assert max([boundary(mask, True) for mask in final_masks]) == boundary(final_masks[index], True)
                print('max([boundary(mask, True) for mask in final_masks])', max([boundary(mask, True) for mask in final_masks]), 'boundary(final_masks[index], True)', boundary(final_masks[index], True))
                flag = 1
                for i in range(224):
                    for j in range(224):
                        if final_masks[index][i, j] != 0:
                            start_pos = [i, j]
                            flag = 0
                            break
                    if not flag:
                        break 
                assert final_masks[index][i, j] == 255
                print('final_masks[index][i, j]', final_masks[index][i, j], '[start pos: ]', [i, j])                         
                img = fill_color(img, start_pos, [0, 0, 0], final_masks[index])
                num_regions, _ = check_num_regions(img)
            bgmask, _ = area(img, [0, 0])
            lsmask = 255*np.ones((224, 224), dtype=np.uint8)-bgmask
            out = lsmask
    
    cv2.imwrite(dst+file, out)
    return out


def bw(src='./ISIC 2016 for maskgen/MASKS/FILTERED/', dst='./ISIC 2016 for maskgen/MASKS/BW/', file=None):
    
    if isinstance(src, str):
        for file in os.listdir(src):
            img = cv2.imread(src+file)
            out = convert(img, dst, file)
    else:
        out = convert(src, dst, file)
    
    return out

'''IF A REGION IS ALREADY BLACK, DO NOT DO ANYTHING WITH IT'''