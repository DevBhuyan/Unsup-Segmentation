#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 07:17:16 2023

@author: dev
"""
import cv2
import os

def permute(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    combs = {
        'blue-green': b+g, 
        'green-red': g+r, 
        'blue-red': b+r, 
        'bgr': b+g+r
        }
    for comb in combs:
        cv2.imshow(str(comb), combs[comb])
        cv2.waitKey(1)
        
        
if __name__ == "__main__":
    base_dir = './ISIC 2016 for maskgen/IMAGES/'
    for file in os.listdir(base_dir):
        permute(base_dir+file)
        x = input()