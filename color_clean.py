#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 01:04:48 2023

@author: dev
"""
import os
import cv2
import numpy as np
from marginalize import marginalize
from color_to_bw import bw
import pandas as pd
from metrics import metrics
from datetime import datetime


def stabilize(img):
    colors = np.array([
        [96, 96, 96],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [224, 224, 224],
        [160, 160, 160],
        [128, 128, 128],
        [192, 192, 192]
    ])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j].astype(int)
            distances = np.sum((colors - pixel)**2, axis=1)
            closest_color_index = np.argmin(distances)
            closest_color = colors[closest_color_index]
            img[i, j] = closest_color

    return img


def color_clean(src = './ISIC 2016 for maskgen/MASKS/UNFILTERED/', file=None, dst = './ISIC 2016 for maskgen/MASKS/FILTERED/'):

    if isinstance(src, str):
        print('src : ', src)
        print('dst : ', dst)

        df = pd.read_csv('./nomans favourite variable.csv')
        listdir = os.listdir(dst)

        # for file in df['Part 4']:
        for file in os.listdir(src):
            start = datetime.now()
            if file in listdir:
                continue
            try:
                img = cv2.imread(src+file)
            except:
                continue

            print('FILENAME: ', file)

            img = stabilize(img)

            thresh = 0.001
            l = 5
            run = 0

            while(l>2 and thresh < 0.05):
                print('marginalizing run: ', run+1, 'threshold percent: ', thresh)
                pixels = img.reshape(-1, img.shape[-1])
                unique_colors = np.unique(pixels, axis=0)
                counts = np.array([np.count_nonzero(np.all(img == color, axis=-1)) for color in unique_colors])
                perc_px = [i/sum(counts) for i in counts]

                for i in range(len(unique_colors)):
                    print(f"Color: {unique_colors[i]}, Percentage: {perc_px[i]}")

                img = marginalize(img, thresh)
                thresh*=2
                l = len(unique_colors)
                run += 1
                # x = input('Press Return to go to next run')

            cv2.imwrite(dst+file, img)
            bw(img, file=file)
            print('latency for image: ', datetime.now()-start)
        metrics()

    else:

        print('FILENAME: ', file)

        img = stabilize(img)

        thresh = 0.001

        pixels = img.reshape(-1, img.shape[-1])
        unique_colors = np.unique(pixels, axis=0)
        counts = np.array([np.count_nonzero(np.all(img == color, axis=-1)) for color in unique_colors])
        perc_px = [i/sum(counts) for i in counts]

        for i in range(len(unique_colors)):
            print(f"Color: {unique_colors[i]}, Percentage: {perc_px[i]}")

        while len(unique_colors) > 2:
            img = marginalize(img, thresh*2)
            pixels = img.reshape(-1, img.shape[-1])
            unique_colors = np.unique(pixels, axis=0)

        cv2.imwrite(dst+file, img)
        img = bw(img, dst=dst[:-9]+'BW/', file=file)



if __name__ == "__main__":
    color_clean()