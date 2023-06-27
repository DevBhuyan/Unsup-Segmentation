#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:27:40 2023

@author: dev
"""

import cv2

image = cv2.imread('ISIC_0000003.jpg', cv2.IMREAD_COLOR)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
lab[:,:,0] = clahe.apply(lab[:,:,0])

clahe_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
cv2.imwrite('CLAHE_ISIC_0000003.jpg', clahe_image)
