#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:48:46 2023

@author: dev
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def threshold_mask(mask):
    _, thresholded = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return thresholded

def combine_masks(mask1, mask2):
    combined = np.maximum(mask1, mask2)
    return combined

# Path to the folders containing the masks
mask_folder1 = './ISIC 2016 maskgen outputs for UNET/MASKS/BW/'
mask_folder2 = './ISIC 2016 maskgen outputs for UNET/MASKS/REFERENCE/'

# Get the list of mask file names from folder1
mask_files = os.listdir(mask_folder1)

# Iterate over each mask file
for file in tqdm(mask_files, total=1279):
    # Load masks from both folders
    mask1_path = os.path.join(mask_folder1, file)
    mask2_path = os.path.join(mask_folder2, file)
    
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform thresholding
    mask1 = threshold_mask(mask1)
    mask2 = threshold_mask(mask2)
    
    # Combine the masks
    combined_mask = combine_masks(mask1, mask2)
    
    # Save the resulting combined mask
    output_path = './ISIC 2016 maskgen outputs for UNET/MASKS/' + file
    if not cv2.imwrite(output_path, combined_mask):
        raise Exception('Image not saved')
