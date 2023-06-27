#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 01:54:48 2023

@author: dev
"""

import os
import cv2
from sklearn.metrics import precision_score, jaccard_score, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm


def threshold(img, threshold=128):
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img


def metrics(img_name='hello'):
    gt_path = './ISIC 2016 for maskgen (backup)/MASKS/REFERENCE/'
    gen_path = './ISIC 2016 for maskgen/MASKS/BW/'

    if not isinstance(img_name, str):
        gt_mask = cv2.imread(gt_path+img_name, 0).astype(np.uint8)
        gen_mask = cv2.imread(gen_path+img_name, 0).astype(np.uint8)

        gt_mask = threshold(gt_mask)
        gen_mask = threshold(gen_mask)

        gt_mask = (gt_mask > 0).astype(np.uint8)
        gen_mask = (gen_mask > 0).astype(np.uint8)

        acc = accuracy_score(gt_mask.ravel(), gen_mask.ravel())
        precision = precision_score(gt_mask.ravel(), gen_mask.ravel(), zero_division=0)
        jaccard = jaccard_score(gt_mask.ravel(), gen_mask.ravel(), zero_division=0)
        dice = f1_score(gt_mask.ravel(), gen_mask.ravel(), zero_division=0)

        return acc, precision, jaccard, dice

    images = os.listdir(gen_path)

    precision_scores = []
    jaccard_scores = []
    dice_scores = []
    accuracy_scores = []

    ls = []

    for img_name in tqdm(images, total=len(images)):
        # print(img_name)
        gt_mask = cv2.imread(gt_path+img_name, 0).astype(np.uint8)
        gen_mask = cv2.imread(gen_path+img_name, 0).astype(np.uint8)

        gt_mask = threshold(gt_mask)
        gen_mask = threshold(gen_mask)

        gt_mask = (gt_mask > 0).astype(np.uint8)
        gen_mask = (gen_mask > 0).astype(np.uint8)

        acc = accuracy_score(gt_mask.ravel(), gen_mask.ravel())
        precision = precision_score(gt_mask.ravel(), gen_mask.ravel(), zero_division=0)
        jaccard = jaccard_score(gt_mask.ravel(), gen_mask.ravel(), zero_division=0)
        dice = f1_score(gt_mask.ravel(), gen_mask.ravel(), zero_division=0)

        if dice <= 0.0:
            # print('accuracy: ', acc, 'precision: ', precision, 'jaccard: ', jaccard, 'dice: ', dice)
            # print('Check image: ', img_name)
            ls.append(img_name)
        else:
            accuracy_scores.append(acc)
            precision_scores.append(precision)
            jaccard_scores.append(jaccard)
            dice_scores.append(dice)


    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_acc = sum(accuracy_scores)/len(accuracy_scores)

    print(f"Average Accuracy: {avg_acc}")
    print(f"Average precision: {avg_precision}")
    print(f"Average Jaccard score: {avg_jaccard}")
    print(f"Average Dice score: {avg_dice}")

    return ls

if __name__ == "__main__":
    ls = metrics()